"""
Quality Evaluation for LongLive Optimizations.

Compares visual quality between baseline and optimized pipelines using:
1. PSNR - Peak Signal-to-Noise Ratio
2. SSIM - Structural Similarity Index
3. LPIPS - Learned Perceptual Image Patch Similarity
4. CLIP Score - Prompt adherence via CLIP embeddings

Usage:
    # Generate comparison videos
    python benchmarks/quality_eval.py \
        --config configs/longlive_inference.yaml \
        --preset balanced \
        --output results/quality

    # Quick visual check
    python benchmarks/quality_eval.py \
        --config configs/longlive_inference.yaml \
        --preset balanced \
        --quick
"""

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))


# Test prompts for quality evaluation
EVAL_PROMPTS = [
    "A panda walking through a bamboo forest",
    "Ocean waves crashing on rocks at sunset",
    "A car driving through city streets at night",
    "Fireworks exploding over a cityscape",
    "A person dancing in the rain",
]

# Sample sizes
FRAMES_PER_PROMPT = 120  # 120 frames per prompt for evaluation
QUICK_FRAMES = 30  # Quick mode


@dataclass
class QualityMetrics:
    """Quality metrics for a single comparison."""
    prompt: str = ""

    # Frame-level metrics (mean across frames)
    psnr_mean: float = 0.0
    psnr_std: float = 0.0
    ssim_mean: float = 0.0
    ssim_std: float = 0.0
    lpips_mean: float = 0.0
    lpips_std: float = 0.0

    # CLIP scores
    clip_baseline: float = 0.0
    clip_optimized: float = 0.0
    clip_delta: float = 0.0

    # Temporal consistency
    flow_consistency: float = 0.0


@dataclass
class QualityReport:
    """Full quality evaluation report."""
    preset: str = ""
    num_prompts: int = 0
    frames_per_prompt: int = 0

    # Aggregate metrics (mean across all prompts)
    psnr_overall: float = 0.0
    ssim_overall: float = 0.0
    lpips_overall: float = 0.0
    clip_delta_overall: float = 0.0

    # Per-prompt results
    per_prompt: List[Dict] = field(default_factory=list)

    # Quality verdict
    acceptable: bool = True
    notes: str = ""


def compute_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    Compute PSNR between two images.

    Args:
        img1, img2: Tensors of shape [C, H, W] or [B, C, H, W], values in [0, 1]

    Returns:
        PSNR in dB (higher is better, >30dB is good)
    """
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    return 10 * torch.log10(1.0 / mse).item()


def compute_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> float:
    """
    Compute SSIM between two images.

    Args:
        img1, img2: Tensors of shape [C, H, W] or [B, C, H, W], values in [0, 1]

    Returns:
        SSIM value (higher is better, >0.9 is good)
    """
    # Simple SSIM implementation
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Ensure 4D
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    # Convert to grayscale for simplicity
    img1_gray = 0.299 * img1[:, 0] + 0.587 * img1[:, 1] + 0.114 * img1[:, 2]
    img2_gray = 0.299 * img2[:, 0] + 0.587 * img2[:, 1] + 0.114 * img2[:, 2]

    mu1 = F.avg_pool2d(img1_gray.unsqueeze(1), window_size, stride=1, padding=window_size // 2)
    mu2 = F.avg_pool2d(img2_gray.unsqueeze(1), window_size, stride=1, padding=window_size // 2)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(img1_gray.unsqueeze(1) ** 2, window_size, stride=1, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2_gray.unsqueeze(1) ** 2, window_size, stride=1, padding=window_size // 2) - mu2_sq
    sigma12 = F.avg_pool2d(img1_gray.unsqueeze(1) * img2_gray.unsqueeze(1), window_size, stride=1, padding=window_size // 2) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean().item()


class LPIPSMetric:
    """LPIPS perceptual similarity metric."""

    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._model = None

    def _load_model(self):
        """Lazy load LPIPS model."""
        if self._model is None:
            try:
                import lpips
                self._model = lpips.LPIPS(net='alex').to(self.device)
                self._model.eval()
            except ImportError:
                print("Warning: lpips not installed, using fallback")
                self._model = "fallback"

    def compute(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """
        Compute LPIPS between two images.

        Args:
            img1, img2: Tensors of shape [C, H, W] or [B, C, H, W], values in [0, 1]

        Returns:
            LPIPS value (lower is better, <0.1 is good)
        """
        self._load_model()

        if self._model == "fallback":
            # Fallback: use L2 distance in normalized space
            return F.mse_loss(img1, img2).item()

        # Ensure 4D and on correct device
        if img1.dim() == 3:
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)

        img1 = img1.to(self.device)
        img2 = img2.to(self.device)

        # LPIPS expects [-1, 1]
        img1 = img1 * 2 - 1
        img2 = img2 * 2 - 1

        with torch.no_grad():
            return self._model(img1, img2).item()


class CLIPScore:
    """CLIP-based prompt adherence scorer."""

    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._model = None
        self._preprocess = None

    def _load_model(self):
        """Lazy load CLIP model."""
        if self._model is None:
            try:
                import clip
                self._model, self._preprocess = clip.load("ViT-B/32", device=self.device)
            except ImportError:
                print("Warning: clip not installed, using fallback")
                self._model = "fallback"

    def compute(self, image: torch.Tensor, prompt: str) -> float:
        """
        Compute CLIP similarity between image and prompt.

        Args:
            image: Tensor of shape [C, H, W], values in [0, 1]
            prompt: Text prompt

        Returns:
            CLIP similarity score (higher is better)
        """
        self._load_model()

        if self._model == "fallback":
            return 0.0

        import clip

        # Preprocess image
        if image.dim() == 3:
            image = image.unsqueeze(0)

        # Convert to PIL and preprocess
        from torchvision.transforms.functional import to_pil_image
        pil_image = to_pil_image(image[0])
        image_input = self._preprocess(pil_image).unsqueeze(0).to(self.device)

        # Tokenize text
        text_input = clip.tokenize([prompt]).to(self.device)

        with torch.no_grad():
            image_features = self._model.encode_image(image_input)
            text_features = self._model.encode_text(text_input)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            similarity = (image_features @ text_features.T).item()

        return similarity


def evaluate_prompt(
    baseline_pipeline,
    optimized_pipeline,
    prompt: str,
    num_frames: int,
    lpips_metric: LPIPSMetric,
    clip_scorer: CLIPScore,
    device: torch.device,
) -> QualityMetrics:
    """
    Evaluate quality for a single prompt.

    Generates videos from both pipelines and compares.
    """
    print(f"    Evaluating: '{prompt[:40]}...'")

    metrics = QualityMetrics(prompt=prompt)

    # Generate sample input
    noise = torch.randn(1, 16, 3, 60, 104, device=device, dtype=torch.bfloat16)

    # Generate baseline video
    with torch.no_grad():
        baseline_video = baseline_pipeline.inference(noise.clone(), [prompt], num_frames=num_frames)

    # Generate optimized video
    with torch.no_grad():
        optimized_video = optimized_pipeline.inference(noise.clone(), [prompt], num_frames=num_frames)

    # Ensure same shape
    assert baseline_video.shape == optimized_video.shape, \
        f"Shape mismatch: {baseline_video.shape} vs {optimized_video.shape}"

    # Compute frame-level metrics
    psnr_values = []
    ssim_values = []
    lpips_values = []

    for i in range(min(num_frames, baseline_video.shape[0])):
        base_frame = baseline_video[i]
        opt_frame = optimized_video[i]

        psnr_values.append(compute_psnr(base_frame, opt_frame))
        ssim_values.append(compute_ssim(base_frame, opt_frame))
        lpips_values.append(lpips_metric.compute(base_frame, opt_frame))

    metrics.psnr_mean = float(np.mean(psnr_values))
    metrics.psnr_std = float(np.std(psnr_values))
    metrics.ssim_mean = float(np.mean(ssim_values))
    metrics.ssim_std = float(np.std(ssim_values))
    metrics.lpips_mean = float(np.mean(lpips_values))
    metrics.lpips_std = float(np.std(lpips_values))

    # CLIP scores (use middle frame)
    mid_idx = num_frames // 2
    metrics.clip_baseline = clip_scorer.compute(baseline_video[mid_idx], prompt)
    metrics.clip_optimized = clip_scorer.compute(optimized_video[mid_idx], prompt)
    metrics.clip_delta = metrics.clip_optimized - metrics.clip_baseline

    return metrics


def run_quality_evaluation(
    config_path: str,
    preset: str,
    output_dir: str,
    quick: bool = False,
) -> QualityReport:
    """
    Run full quality evaluation.

    Args:
        config_path: Path to LongLive config
        preset: Optimization preset
        output_dir: Directory for output
        quick: Quick mode with fewer frames

    Returns:
        QualityReport with all metrics
    """
    from benchmark_suite import load_pipeline

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_frames = QUICK_FRAMES if quick else FRAMES_PER_PROMPT

    print(f"\n{'='*60}")
    print("Quality Evaluation")
    print(f"Preset: {preset}")
    print(f"Frames per prompt: {num_frames}")
    print(f"{'='*60}\n")

    # Load pipelines
    print("Loading baseline pipeline...")
    baseline = load_pipeline(config_path, optimized=False)

    print("Loading optimized pipeline...")
    optimized = load_pipeline(config_path, optimized=True, preset=preset)

    # Initialize metrics
    lpips_metric = LPIPSMetric(device)
    clip_scorer = CLIPScore(device)

    # Evaluate each prompt
    report = QualityReport(
        preset=preset,
        num_prompts=len(EVAL_PROMPTS),
        frames_per_prompt=num_frames,
    )

    all_psnr = []
    all_ssim = []
    all_lpips = []
    all_clip_delta = []

    for prompt in EVAL_PROMPTS:
        metrics = evaluate_prompt(
            baseline, optimized, prompt, num_frames,
            lpips_metric, clip_scorer, device
        )

        report.per_prompt.append(asdict(metrics))

        all_psnr.append(metrics.psnr_mean)
        all_ssim.append(metrics.ssim_mean)
        all_lpips.append(metrics.lpips_mean)
        all_clip_delta.append(metrics.clip_delta)

    # Aggregate
    report.psnr_overall = float(np.mean(all_psnr))
    report.ssim_overall = float(np.mean(all_ssim))
    report.lpips_overall = float(np.mean(all_lpips))
    report.clip_delta_overall = float(np.mean(all_clip_delta))

    # Quality verdict
    # Thresholds: PSNR > 30dB, SSIM > 0.9, LPIPS < 0.1, CLIP delta < 2%
    if report.psnr_overall < 30:
        report.acceptable = False
        report.notes += "PSNR below threshold. "
    if report.ssim_overall < 0.9:
        report.acceptable = False
        report.notes += "SSIM below threshold. "
    if report.lpips_overall > 0.1:
        report.acceptable = False
        report.notes += "LPIPS above threshold. "
    if abs(report.clip_delta_overall) > 0.02:
        report.acceptable = False
        report.notes += "CLIP delta above threshold. "

    if report.acceptable:
        report.notes = "All metrics within acceptable range."

    # Print summary
    print(f"\n{'='*60}")
    print("Quality Results")
    print(f"{'='*60}")
    print(f"  PSNR:  {report.psnr_overall:.2f} dB (threshold: >30)")
    print(f"  SSIM:  {report.ssim_overall:.4f} (threshold: >0.9)")
    print(f"  LPIPS: {report.lpips_overall:.4f} (threshold: <0.1)")
    print(f"  CLIP Delta: {report.clip_delta_overall:+.4f} (threshold: <2%)")
    print(f"\n  Verdict: {'ACCEPTABLE' if report.acceptable else 'DEGRADED'}")
    print(f"  Notes: {report.notes}")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/quality_{preset}.json", 'w') as f:
        json.dump(asdict(report), f, indent=2)
    print(f"\nSaved: {output_dir}/quality_{preset}.json")

    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='LongLive config')
    parser.add_argument('--preset', type=str, default='balanced', choices=['quality', 'balanced', 'speed'])
    parser.add_argument('--output', type=str, default='results/quality', help='Output directory')
    parser.add_argument('--quick', action='store_true', help='Quick mode (fewer frames)')

    args = parser.parse_args()

    run_quality_evaluation(
        config_path=args.config,
        preset=args.preset,
        output_dir=args.output,
        quick=args.quick,
    )


if __name__ == '__main__':
    main()
