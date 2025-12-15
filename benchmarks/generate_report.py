"""
Generate comparison report from benchmark results.

Usage:
    python benchmarks/generate_report.py \
        --input-dir results/ \
        --output results/comparison_report.md
"""

import argparse
import json
from pathlib import Path
from datetime import datetime


def load_results(input_dir: Path) -> dict:
    """Load all JSON result files from directory."""
    results = {}
    for json_file in input_dir.glob("*.json"):
        with open(json_file) as f:
            data = json.load(f)
            name = json_file.stem
            results[name] = data
    return results


def calculate_improvement(baseline: float, optimized: float) -> str:
    """Calculate improvement percentage."""
    if baseline == 0:
        return "N/A"
    improvement = (baseline - optimized) / baseline * 100
    if improvement > 0:
        return f"-{improvement:.1f}%"
    else:
        return f"+{abs(improvement):.1f}%"


def generate_markdown_report(results: dict, output_path: Path):
    """Generate markdown comparison report."""

    baseline = results.get("baseline", {})
    presets = ["quality", "balanced", "speed"]

    report = []
    report.append("# LongLive Optimization Benchmark Report\n")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    if baseline:
        report.append(f"Device: {baseline.get('device', 'Unknown')}\n")

    # Executive Summary
    report.append("\n## Executive Summary\n")

    best_preset = None
    best_improvement = 0
    for preset in presets:
        key = f"optimized_{preset}"
        if key in results and baseline:
            ss_max_imp = (baseline.get('ss_max', 0) - results[key].get('ss_max', 0))
            if ss_max_imp > best_improvement:
                best_improvement = ss_max_imp
                best_preset = preset

    if best_preset:
        report.append(f"**Recommended Preset**: {best_preset.capitalize()}\n")
        key = f"optimized_{best_preset}"
        report.append(f"- Steady-state max latency: {results[key].get('ss_max', 0):.1f}ms ")
        report.append(f"({calculate_improvement(baseline.get('ss_max', 0), results[key].get('ss_max', 0))} improvement)\n")
        report.append(f"- 40ms Target: {'PASS' if results[key].get('meets_40ms', False) else 'FAIL'}\n")

    # Comparison Table
    report.append("\n## Latency Comparison\n")
    report.append("| Metric | Baseline |")
    for preset in presets:
        if f"optimized_{preset}" in results:
            report.append(f" {preset.capitalize()} |")
    report.append("\n")

    report.append("|--------|----------|")
    for preset in presets:
        if f"optimized_{preset}" in results:
            report.append("----------|")
    report.append("\n")

    # Rows
    metrics = [
        ("ss_mean", "SS Mean (ms)"),
        ("ss_p99", "SS P99 (ms)"),
        ("ss_max", "SS Max (ms)"),
        ("ps_mean", "PS Mean (ms)"),
        ("ps_max", "PS Max (ms)"),
        ("fps", "Throughput (FPS)"),
        ("peak_memory_gb", "Memory (GB)"),
    ]

    for key, label in metrics:
        row = f"| {label} | "
        base_val = baseline.get(key, 0)
        row += f"{base_val:.1f} |"

        for preset in presets:
            result_key = f"optimized_{preset}"
            if result_key in results:
                opt_val = results[result_key].get(key, 0)
                imp = calculate_improvement(base_val, opt_val)
                if key == "fps":
                    # For FPS, higher is better
                    fps_imp = (opt_val - base_val) / base_val * 100 if base_val > 0 else 0
                    imp = f"+{fps_imp:.1f}%" if fps_imp > 0 else f"{fps_imp:.1f}%"
                row += f" {opt_val:.1f} ({imp}) |"
        report.append(row + "\n")

    # Target Pass/Fail
    report.append("\n| 40ms Target |")
    if baseline:
        report.append(f" {'PASS' if baseline.get('ss_max', 999) <= 40 else 'FAIL'} |")
    for preset in presets:
        key = f"optimized_{preset}"
        if key in results:
            report.append(f" {'PASS' if results[key].get('meets_40ms', False) else 'FAIL'} |")
    report.append("\n")

    # Detailed Analysis
    report.append("\n## Detailed Analysis\n")

    # Baseline
    if baseline:
        report.append("### Baseline (No Optimizations)\n")
        report.append(f"- Steady-State: mean={baseline.get('ss_mean', 0):.1f}ms, ")
        report.append(f"p99={baseline.get('ss_p99', 0):.1f}ms, max={baseline.get('ss_max', 0):.1f}ms\n")
        report.append(f"- Prompt Switch: mean={baseline.get('ps_mean', 0):.1f}ms, max={baseline.get('ps_max', 0):.1f}ms\n")
        report.append(f"- Throughput: {baseline.get('fps', 0):.1f} FPS\n")
        report.append(f"- Memory: {baseline.get('peak_memory_gb', 0):.2f} GB\n")

    # Each preset
    for preset in presets:
        key = f"optimized_{preset}"
        if key in results:
            r = results[key]
            report.append(f"\n### {preset.capitalize()} Preset\n")
            report.append(f"- Steady-State: mean={r.get('ss_mean', 0):.1f}ms, ")
            report.append(f"p99={r.get('ss_p99', 0):.1f}ms, max={r.get('ss_max', 0):.1f}ms\n")
            report.append(f"- Prompt Switch: mean={r.get('ps_mean', 0):.1f}ms, max={r.get('ps_max', 0):.1f}ms\n")
            report.append(f"- Throughput: {r.get('fps', 0):.1f} FPS\n")
            report.append(f"- Memory: {r.get('peak_memory_gb', 0):.2f} GB\n")
            report.append(f"- 40ms Target: {'**PASS**' if r.get('meets_40ms', False) else 'FAIL'}\n")

    # Recommendations
    report.append("\n## Recommendations\n")
    report.append("1. **Use Balanced Preset** for most production deployments\n")
    report.append("2. **Use Speed Preset** if latency is critical and slight quality loss is acceptable\n")
    report.append("3. **Use Quality Preset** for offline rendering or when quality is paramount\n")
    report.append("4. **Pre-warm prompt cache** with common prompts before interactive sessions\n")
    report.append("5. **Monitor P99 latency** in production to catch spikes\n")

    # Write report
    with open(output_path, 'w') as f:
        f.writelines(report)

    print(f"Report generated: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, default='results', help='Directory with JSON results')
    parser.add_argument('--output', type=str, default='results/comparison_report.md', help='Output markdown file')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)

    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return

    results = load_results(input_dir)
    if not results:
        print(f"Error: No JSON result files found in {input_dir}")
        return

    print(f"Found {len(results)} result files: {list(results.keys())}")

    generate_markdown_report(results, output_path)


if __name__ == '__main__':
    main()
