import time
import random
import threading
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ProxyState:
    url: str
    fail_count: int = 0
    success_count: int = 0
    banned_until: float = 0.0
    last_used: float = 0.0
    total_requests: int = 0

    @property
    def is_banned(self) -> bool:
        return time.monotonic() < self.banned_until

    @property
    def failure_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.fail_count / self.total_requests


@dataclass
class ProxyPoolConfig:
    ban_threshold: int = 5
    ban_duration_sec: float = 300.0
    ban_escalation_factor: float = 2.0
    max_ban_duration_sec: float = 3600.0
    cooldown_sec: float = 2.0
    max_failure_rate: float = 0.8


class ProxyPool:
    def __init__(self, proxy_urls: list[str],
                 config: ProxyPoolConfig | None = None):
        self._config = config or ProxyPoolConfig()
        self._lock = threading.Lock()
        self._proxies: list[ProxyState] = [
            ProxyState(url=url) for url in proxy_urls
        ]
        self._index = 0

        if not proxy_urls:
            logger.warning("proxy pool empty, using direct connections")

    @property
    def size(self) -> int:
        return len(self._proxies)

    @property
    def available_count(self) -> int:
        with self._lock:
            return sum(1 for p in self._proxies if not p.is_banned)

    def get_next(self) -> str | None:
        if not self._proxies:
            return None

        with self._lock:
            now = time.monotonic()
            n = len(self._proxies)

            for _ in range(n):
                proxy = self._proxies[self._index]
                self._index = (self._index + 1) % n

                if proxy.is_banned:
                    continue

                elapsed = now - proxy.last_used
                if elapsed < self._config.cooldown_sec:
                    continue

                proxy.last_used = now
                proxy.total_requests += 1

                logger.debug("selected proxy %s (ok=%d, fail=%d, rate=%.2f)",
                             proxy.url, proxy.success_count, proxy.fail_count,
                             proxy.failure_rate)
                return proxy.url

            unbanned = [p for p in self._proxies if not p.is_banned]
            if unbanned:
                proxy = min(unbanned, key=lambda p: p.last_used)
                proxy.last_used = now
                proxy.total_requests += 1
                return proxy.url

            earliest_unban = min(p.banned_until for p in self._proxies)
            wait = earliest_unban - now
            logger.warning("all proxies banned, earliest unban in %.1fs", wait)
            return None

    def report_success(self, proxy_url: str) -> None:
        with self._lock:
            proxy = self._find(proxy_url)
            if proxy:
                proxy.success_count += 1
                proxy.fail_count = max(0, proxy.fail_count - 1)

    def report_failure(self, proxy_url: str, reason: str = "") -> None:
        with self._lock:
            proxy = self._find(proxy_url)
            if not proxy:
                return

            proxy.fail_count += 1

            if proxy.fail_count >= self._config.ban_threshold:
                times_banned = proxy.fail_count // self._config.ban_threshold
                duration = min(
                    self._config.ban_duration_sec * (self._config.ban_escalation_factor ** (times_banned - 1)),
                    self._config.max_ban_duration_sec,
                )
                proxy.banned_until = time.monotonic() + duration
                logger.warning("banned proxy %s for %.0fs (fails=%d, reason=%s)",
                               proxy.url, duration, proxy.fail_count, reason or "unknown")

    def report_rate_limited(self, proxy_url: str) -> None:
        self.report_failure(proxy_url, reason="rate_limited")

    def get_stats(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "url": p.url,
                    "success": p.success_count,
                    "fail": p.fail_count,
                    "total": p.total_requests,
                    "failure_rate": round(p.failure_rate, 3),
                    "banned": p.is_banned,
                    "banned_remaining_sec": max(0, round(p.banned_until - time.monotonic(), 1)),
                }
                for p in self._proxies
            ]

    def _find(self, url: str) -> ProxyState | None:
        for p in self._proxies:
            if p.url == url:
                return p
        return None
