"""
Internet sandbox — restricts network access to curated APIs only.

Wraps aiohttp to enforce domain whitelisting, rate limiting,
and response size limits.
"""

from __future__ import annotations

import time
from typing import Optional, Set

from baby_ai.utils.logging import get_logger

log = get_logger("sandbox", log_file="safety.log")


# Default allowed domains
DEFAULT_ALLOWED_DOMAINS: Set[str] = {
    "api.wikipedia.org",
    "api.stackexchange.com",
    "api.github.com",
    "api.weather.gov",
}


class InternetSandbox:
    """
    Network access sandbox.

    All outgoing HTTP requests must pass through this sandbox,
    which enforces:
    - Domain whitelist
    - Rate limiting
    - Response size limits
    - Request logging

    Args:
        allowed_domains: Set of allowed hostnames.
        rate_limit_per_min: Max requests per minute.
        max_response_bytes: Max response body size.
    """

    def __init__(
        self,
        allowed_domains: Optional[Set[str]] = None,
        rate_limit_per_min: int = 10,
        max_response_bytes: int = 1024 * 100,  # 100 KB
    ):
        self.allowed_domains = allowed_domains or DEFAULT_ALLOWED_DOMAINS
        self.rate_limit_per_min = rate_limit_per_min
        self.max_response_bytes = max_response_bytes
        self._call_times: list[float] = []
        self._total_requests = 0
        self._blocked_requests = 0

    def is_allowed(self, url: str) -> bool:
        """Check if a URL is from an allowed domain."""
        from urllib.parse import urlparse
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        allowed = hostname in self.allowed_domains
        if not allowed:
            self._blocked_requests += 1
            log.warning("BLOCKED domain: %s (url: %s)", hostname, url)
        return allowed

    def check_rate(self) -> bool:
        """Check if within rate limit."""
        now = time.time()
        self._call_times = [t for t in self._call_times if now - t < 60]
        ok = len(self._call_times) < self.rate_limit_per_min
        if not ok:
            log.warning("Rate limit exceeded: %d/%d per minute",
                       len(self._call_times), self.rate_limit_per_min)
        return ok

    async def fetch(self, url: str) -> Optional[str]:
        """
        Sandboxed HTTP GET.

        Returns response text if allowed, None otherwise.
        """
        if not self.is_allowed(url):
            return None
        if not self.check_rate():
            return None

        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status != 200:
                        log.warning("HTTP %d from %s", resp.status, url)
                        return None

                    body = await resp.content.read(self.max_response_bytes)
                    self._call_times.append(time.time())
                    self._total_requests += 1

                    text = body.decode("utf-8", errors="replace")
                    log.info("Fetched %d bytes from %s", len(body), url)
                    return text

        except Exception as e:
            log.error("Sandbox fetch error: %s", e)
            return None

    @property
    def stats(self) -> dict:
        return {
            "total_requests": self._total_requests,
            "blocked_requests": self._blocked_requests,
            "allowed_domains": len(self.allowed_domains),
        }
