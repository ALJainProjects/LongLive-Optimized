"""
Tests for PromptEmbeddingCache

Tests LRU cache behavior, hit/miss tracking, and memory management.
"""

import pytest
import torch

from optimizations.prompt_cache import PromptEmbeddingCache, AsyncPromptCache


class MockTextEncoder:
    """Mock text encoder for testing."""

    def __init__(self, embedding_dim: int = 4096, seq_len: int = 77):
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len
        self.call_count = 0

    def __call__(self, text_prompts: list) -> dict:
        """Generate deterministic embeddings based on prompt."""
        self.call_count += 1
        batch_size = len(text_prompts)

        # Generate deterministic embeddings based on prompt hash
        embeddings = []
        for prompt in text_prompts:
            # Use hash to generate reproducible "embedding"
            seed = hash(prompt) % (2**32)
            torch.manual_seed(seed)
            emb = torch.randn(1, self.seq_len, self.embedding_dim)
            embeddings.append(emb)

        # Return dict like real text encoder
        return {
            'prompt_embeds': torch.cat(embeddings, dim=0),
            'prompt_attention_mask': torch.ones(batch_size, self.seq_len),
        }


class TestPromptEmbeddingCache:
    """Tests for PromptEmbeddingCache class."""

    @pytest.fixture
    def encoder(self):
        """Create mock encoder."""
        return MockTextEncoder()

    @pytest.fixture
    def cache(self, encoder):
        """Create cache with mock encoder."""
        return PromptEmbeddingCache(
            text_encoder=encoder,
            max_cache_size=10,
        )

    def test_initialization(self, cache):
        """Test cache initialization."""
        assert cache.max_cache_size == 10
        assert len(cache) == 0
        assert cache.hits == 0
        assert cache.misses == 0

    def test_cache_miss(self, cache, encoder):
        """Test cache miss triggers encoding."""
        prompt = "A panda walking through bamboo"
        initial_calls = encoder.call_count

        embedding = cache.get_embeddings([prompt])

        assert embedding is not None
        assert 'prompt_embeds' in embedding
        assert encoder.call_count == initial_calls + 1
        assert cache.misses == 1
        assert cache.hits == 0

    def test_cache_hit(self, cache, encoder):
        """Test cache hit returns cached value without encoding."""
        prompt = "A panda walking through bamboo"

        # First call - miss
        embedding1 = cache.get_embeddings([prompt])
        calls_after_first = encoder.call_count

        # Second call - hit
        embedding2 = cache.get_embeddings([prompt])

        assert encoder.call_count == calls_after_first  # No new encoding
        assert cache.hits == 1
        assert torch.allclose(
            embedding1['prompt_embeds'],
            embedding2['prompt_embeds']
        )

    def test_different_prompts(self, cache, encoder):
        """Test different prompts are cached separately."""
        prompt1 = "A cat sleeping"
        prompt2 = "A dog running"

        embedding1 = cache.get_embeddings([prompt1])
        embedding2 = cache.get_embeddings([prompt2])

        # Both should be encoded
        assert cache.misses == 2
        # Embeddings should be different
        assert not torch.allclose(
            embedding1['prompt_embeds'],
            embedding2['prompt_embeds']
        )

    def test_lru_eviction(self, cache, encoder):
        """Test LRU eviction when cache is full."""
        # Fill cache
        for i in range(10):
            cache.get_embeddings([f"prompt_{i}"])

        assert len(cache) == 10

        # Access prompt_0 to make it recently used
        cache.get_embeddings(["prompt_0"])

        # Add new prompt, should evict prompt_1 (least recently used)
        cache.get_embeddings(["new_prompt"])

        assert len(cache) == 10
        # prompt_0 should still be cached (was recently accessed)
        assert cache.contains(["prompt_0"])

    def test_clear(self, cache):
        """Test cache clearing."""
        # Add some prompts
        for i in range(5):
            cache.get_embeddings([f"prompt_{i}"])

        cache.clear()

        assert len(cache) == 0
        assert cache.hits == 0
        assert cache.misses == 0

    def test_get_stats(self, cache):
        """Test statistics."""
        # All misses
        for i in range(5):
            cache.get_embeddings([f"prompt_{i}"])

        stats = cache.get_stats()
        assert stats['hit_rate'] == 0.0
        assert stats['misses'] == 5

        # All hits
        for i in range(5):
            cache.get_embeddings([f"prompt_{i}"])

        stats = cache.get_stats()
        assert stats['hit_rate'] == 0.5  # 5 hits / 10 total
        assert stats['hits'] == 5

    def test_embedding_shape(self, cache, encoder):
        """Test returned embeddings have correct shape."""
        prompt = "Test prompt"
        embedding = cache.get_embeddings([prompt])

        assert embedding['prompt_embeds'].shape == (1, encoder.seq_len, encoder.embedding_dim)

    def test_batch_prompts(self, cache, encoder):
        """Test caching with batch of prompts."""
        prompts = ["prompt_a", "prompt_b", "prompt_c"]

        embeddings = cache.get_embeddings(prompts)

        assert embeddings['prompt_embeds'].shape[0] == len(prompts)

    def test_contains(self, cache):
        """Test contains method."""
        prompt = "Test prompt"

        assert not cache.contains([prompt])

        cache.get_embeddings([prompt])

        assert cache.contains([prompt])

    def test_deterministic_embeddings(self, cache):
        """Test same prompt always returns same embedding."""
        prompt = "A consistent prompt"

        embedding1 = cache.get_embeddings([prompt])
        cache.clear()
        embedding2 = cache.get_embeddings([prompt])

        assert torch.allclose(
            embedding1['prompt_embeds'],
            embedding2['prompt_embeds']
        )

    def test_prewarm(self, cache):
        """Test pre-warming cache."""
        prompts = ["prompt_1", "prompt_2", "prompt_3"]

        cache.prewarm(prompts)

        # All should be cached
        for prompt in prompts:
            assert cache.contains([prompt])

    def test_force_recompute(self, cache, encoder):
        """Test force recompute bypasses cache."""
        prompt = "Test prompt"

        # Cache it
        cache.get_embeddings([prompt])
        calls_before = encoder.call_count

        # Force recompute
        cache.get_embeddings([prompt], force_recompute=True)

        # Should have called encoder again
        assert encoder.call_count == calls_before + 1


class TestAsyncPromptCache:
    """Tests for AsyncPromptCache."""

    @pytest.fixture
    def encoder(self):
        return MockTextEncoder()

    @pytest.fixture
    def cache(self, encoder):
        return AsyncPromptCache(
            text_encoder=encoder,
            max_cache_size=10,
        )

    def test_inherits_from_prompt_cache(self, cache):
        """Test AsyncPromptCache inherits from PromptEmbeddingCache."""
        assert isinstance(cache, PromptEmbeddingCache)

    def test_sync_fallback(self, cache, encoder):
        """Test sync fallback when no CUDA."""
        prompt = "Test prompt"

        # Should work even without CUDA
        embedding = cache.get_embeddings([prompt])
        assert embedding is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_prefetch_async(self, cache, encoder):
        """Test async prefetch."""
        prompt = "Test prompt"

        # Start async prefetch
        cache.prefetch_async([prompt])

        # Should be able to get embeddings (may wait for async)
        embedding = cache.get_embeddings([prompt])
        assert embedding is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_prefetch_already_cached(self, cache):
        """Test prefetch skips already cached prompts."""
        prompt = "Test prompt"

        # Cache it first (this is a miss, not a hit)
        cache.get_embeddings([prompt])
        assert cache.misses == 1
        assert cache.hits == 0

        # Prefetch should do nothing (already cached)
        cache.prefetch_async([prompt])

        # Stats should not change since prefetch skipped
        assert cache.misses == 1
        assert cache.hits == 0
