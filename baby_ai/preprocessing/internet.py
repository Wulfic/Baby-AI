"""
Internet data preprocessing — curated API access and snippet parsing.

Implements sandboxed, rate-limited access to curated APIs
and converts retrieved data into compact embeddings.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Optional

import torch

from baby_ai.config import NETWORK_STORAGE
from baby_ai.utils.logging import get_logger

log = get_logger("internet_preproc")


# Curated allowed domains — only these are accessible
ALLOWED_DOMAINS = frozenset({
    "api.wikipedia.org",
    "api.stackexchange.com",
    "api.github.com",
    "api.weather.gov",
})


class InternetPreprocessor:
    """
    Sandboxed internet access for curated API data.

    Fetches small JSON snippets from whitelisted APIs,
    extracts key text, and converts to embeddings via
    simple bag-of-characters encoding (no tokenizer needed).

    Args:
        max_snippet_chars: Maximum characters to retain per fetch.
        cache_dir: Directory to cache fetched data.
        rate_limit_per_min: Maximum API calls per minute.
        char_embed_dim: Dimension of character-level embeddings.
    """

    def __init__(
        self,
        max_snippet_chars: int = 1024,
        cache_dir: Optional[Path] = None,
        rate_limit_per_min: int = 10,
        char_embed_dim: int = 128,
    ):
        self.max_snippet_chars = max_snippet_chars
        self.cache_dir = cache_dir or (NETWORK_STORAGE / "api_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit = rate_limit_per_min
        self.char_embed_dim = char_embed_dim

        self._call_times: list[float] = []

    def _check_domain(self, url: str) -> bool:
        """Verify URL is from an allowed domain."""
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.hostname in ALLOWED_DOMAINS

    def _rate_limit_ok(self) -> bool:
        """Check if we're within rate limits."""
        now = time.time()
        self._call_times = [t for t in self._call_times if now - t < 60]
        return len(self._call_times) < self.rate_limit

    def _cache_key(self, url: str) -> str:
        return hashlib.md5(url.encode()).hexdigest()

    def _get_cached(self, url: str) -> Optional[str]:
        """Check disk cache for a previous fetch."""
        cache_path = self.cache_dir / f"{self._cache_key(url)}.json"
        if cache_path.exists():
            with open(cache_path, "r") as f:
                data = json.load(f)
            return data.get("text", "")
        return None

    def _save_cache(self, url: str, text: str) -> None:
        cache_path = self.cache_dir / f"{self._cache_key(url)}.json"
        with open(cache_path, "w") as f:
            json.dump({"url": url, "text": text, "time": time.time()}, f)

    async def fetch(self, url: str) -> Optional[str]:
        """
        Fetch a snippet from a curated API.

        Returns None if blocked by safety checks or rate limits.
        """
        if not self._check_domain(url):
            log.warning("Blocked: domain not in allowlist — %s", url)
            return None

        if not self._rate_limit_ok():
            log.warning("Rate limit exceeded, skipping fetch.")
            return None

        # Check cache first
        cached = self._get_cached(url)
        if cached is not None:
            return cached

        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        return None
                    text = await resp.text()
                    text = text[:self.max_snippet_chars]
                    self._save_cache(url, text)
                    self._call_times.append(time.time())
                    return text
        except Exception as e:
            log.error("Fetch error: %s", e)
            return None

    def text_to_embedding(self, text: str) -> torch.Tensor:
        """
        Convert text snippet to a fixed-size embedding.

        Uses a simple bag-of-characters approach — no tokenizer.
        This is intentionally primitive; the model learns to
        extract useful features via the encoder stack.

        Returns:
            (char_embed_dim,) float tensor.
        """
        # Character frequency histogram (ASCII printable range 32-126)
        hist = torch.zeros(95, dtype=torch.float32)
        for ch in text:
            idx = ord(ch) - 32
            if 0 <= idx < 95:
                hist[idx] += 1

        # Normalize
        hist = hist / (hist.sum() + 1e-8)

        # Project to embed_dim via simple hashing
        # Deterministic random projection
        rng = torch.Generator().manual_seed(42)
        proj = torch.randn(95, self.char_embed_dim, generator=rng)
        embedding = hist @ proj
        embedding = embedding / (embedding.norm() + 1e-8)

        return embedding
