"""
Prompt Embedding Cache for LongLive.

Caches text encoder outputs to eliminate prompt encoding latency during
inference, especially useful for prompt switches to previously seen prompts.

Key features:
1. LRU eviction policy
2. Hash-based key lookup
3. GPU tensor caching
4. Pre-warming capability for common prompts
"""

import torch
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union
import hashlib


class PromptEmbeddingCache:
    """
    Cache for text encoder embeddings with LRU eviction.

    Eliminates redundant text encoding when:
    1. Same prompt is used repeatedly
    2. Switching between previously seen prompts
    3. Interactive applications with common prompt patterns

    Usage:
        cache = PromptEmbeddingCache(text_encoder, max_size=100)

        # First call - computes and caches
        embeds = cache.get_embeddings(["A panda walking"])

        # Second call - returns cached (near-zero latency)
        embeds = cache.get_embeddings(["A panda walking"])

        # Pre-warm with common prompts
        cache.prewarm(["prompt1", "prompt2", ...])
    """

    def __init__(
        self,
        text_encoder,
        max_cache_size: int = 100,
        device: torch.device = None,
    ):
        """
        Initialize prompt embedding cache.

        Args:
            text_encoder: The text encoder model (WanTextEncoder)
            max_cache_size: Maximum number of prompts to cache
            device: Device for cached tensors
        """
        self.text_encoder = text_encoder
        self.max_cache_size = max_cache_size
        self.device = device or torch.device('cuda')

        # LRU cache: key -> embeddings dict
        self._cache: OrderedDict[str, Dict[str, torch.Tensor]] = OrderedDict()

        # Statistics
        self.hits = 0
        self.misses = 0

    def _hash_prompt(self, prompt: str) -> str:
        """Create a hash key for a prompt."""
        return hashlib.md5(prompt.encode()).hexdigest()

    def _hash_prompts(self, prompts: List[str]) -> str:
        """Create a hash key for a list of prompts."""
        combined = "||".join(prompts)
        return hashlib.md5(combined.encode()).hexdigest()

    def get_embeddings(
        self,
        text_prompts: List[str],
        force_recompute: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Get embeddings for prompts, using cache when available.

        Args:
            text_prompts: List of text prompts
            force_recompute: If True, bypass cache and recompute

        Returns:
            Dictionary containing prompt embeddings (same format as text_encoder output)
        """
        cache_key = self._hash_prompts(text_prompts)

        if not force_recompute and cache_key in self._cache:
            # Cache hit - move to end (most recently used)
            self._cache.move_to_end(cache_key)
            self.hits += 1
            return self._cache[cache_key]

        # Cache miss - compute embeddings
        self.misses += 1

        with torch.no_grad():
            embeddings = self.text_encoder(text_prompts=text_prompts)

        # Clone tensors to avoid issues with in-place modifications
        cached_embeddings = {}
        for key, value in embeddings.items():
            if isinstance(value, torch.Tensor):
                cached_embeddings[key] = value.clone()
            else:
                cached_embeddings[key] = value

        # Add to cache
        self._cache[cache_key] = cached_embeddings

        # Evict oldest if over capacity
        while len(self._cache) > self.max_cache_size:
            self._cache.popitem(last=False)

        return embeddings

    def get_single_embedding(self, prompt: str) -> Dict[str, torch.Tensor]:
        """Convenience method for single prompt."""
        return self.get_embeddings([prompt])

    def prewarm(self, prompts: List[str]):
        """
        Pre-compute and cache embeddings for a list of prompts.

        Useful for warming up cache with common prompts before
        interactive sessions.

        Args:
            prompts: List of prompts to pre-cache
        """
        print(f"Pre-warming cache with {len(prompts)} prompts...")
        for prompt in prompts:
            _ = self.get_embeddings([prompt])
        print(f"Cache pre-warmed. Size: {len(self._cache)}/{self.max_cache_size}")

    def contains(self, text_prompts: List[str]) -> bool:
        """Check if prompts are in cache."""
        cache_key = self._hash_prompts(text_prompts)
        return cache_key in self._cache

    def store(self, text_prompts: List[str], embeddings: Dict[str, torch.Tensor]):
        """
        Manually store embeddings in cache.

        Used when embeddings are computed externally (e.g., by original text encoder)
        and we want to cache the result.

        Args:
            text_prompts: List of text prompts (used as key)
            embeddings: Dictionary containing prompt embeddings
        """
        cache_key = self._hash_prompts(text_prompts)

        # Clone tensors to avoid issues with in-place modifications
        cached_embeddings = {}
        for key, value in embeddings.items():
            if isinstance(value, torch.Tensor):
                cached_embeddings[key] = value.clone()
            else:
                cached_embeddings[key] = value

        # Add to cache
        self._cache[cache_key] = cached_embeddings

        # Evict oldest if over capacity
        while len(self._cache) > self.max_cache_size:
            self._cache.popitem(last=False)

    def clear(self):
        """Clear all cached embeddings."""
        self._cache.clear()
        self.hits = 0
        self.misses = 0

    def get_stats(self) -> Dict[str, Union[int, float]]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0

        return {
            'size': len(self._cache),
            'max_size': self.max_cache_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
        }

    def __len__(self) -> int:
        return len(self._cache)

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"PromptEmbeddingCache(\n"
            f"  size={stats['size']}/{stats['max_size']},\n"
            f"  hits={stats['hits']},\n"
            f"  misses={stats['misses']},\n"
            f"  hit_rate={stats['hit_rate']:.2%}\n"
            f")"
        )


class AsyncPromptCache(PromptEmbeddingCache):
    """
    Async-aware prompt cache with speculative pre-computation.

    Can speculatively compute embeddings for likely next prompts
    while current generation is running.
    """

    def __init__(
        self,
        text_encoder,
        max_cache_size: int = 100,
        device: torch.device = None,
    ):
        super().__init__(text_encoder, max_cache_size, device)

        # Stream for async embedding computation
        self._compute_stream = torch.cuda.Stream() if torch.cuda.is_available() else None

        # Pending async computations
        self._pending: Dict[str, torch.cuda.Event] = {}

    def prefetch_async(self, text_prompts: List[str]):
        """
        Start async computation of embeddings.

        Does not block - computation happens on separate CUDA stream.
        Call get_embeddings() later to wait and retrieve results.
        """
        cache_key = self._hash_prompts(text_prompts)

        if cache_key in self._cache or cache_key in self._pending:
            return  # Already cached or computing

        if self._compute_stream is None:
            # No CUDA, fallback to sync
            self.get_embeddings(text_prompts)
            return

        # Start async computation
        event = torch.cuda.Event()

        with torch.cuda.stream(self._compute_stream):
            with torch.no_grad():
                embeddings = self.text_encoder(text_prompts=text_prompts)

                # Clone to separate memory
                cached_embeddings = {}
                for key, value in embeddings.items():
                    if isinstance(value, torch.Tensor):
                        cached_embeddings[key] = value.clone()
                    else:
                        cached_embeddings[key] = value

                self._cache[cache_key] = cached_embeddings

            event.record()

        self._pending[cache_key] = event

    def get_embeddings(
        self,
        text_prompts: List[str],
        force_recompute: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Get embeddings, waiting for async computation if needed."""
        cache_key = self._hash_prompts(text_prompts)

        # Check if pending async computation
        if cache_key in self._pending:
            event = self._pending[cache_key]
            event.synchronize()
            del self._pending[cache_key]

        return super().get_embeddings(text_prompts, force_recompute)
