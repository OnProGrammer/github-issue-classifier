import time
import random
import logging
from dataclasses import dataclass

import requests

from scraper.net.proxy_pool import ProxyPool

logger = logging.getLogger(__name__)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.5; rv:126.0) Gecko/20100101 Firefox/126.0",
]

ACCEPT_LANGUAGES = [
    "en-US,en;q=0.9",
    "en-GB,en;q=0.9",
    "en-US,en;q=0.9,de;q=0.8",
    "en,en-US;q=0.9,fr;q=0.7",
]


@dataclass
class RetryConfig:
    max_retries: int = 4
    base_delay_sec: float = 2.0
    max_delay_sec: float = 60.0
    jitter_range: float = 0.5
    retry_on_statuses: tuple[int, ...] = (429, 500, 502, 503, 504)


@dataclass
class RateLimitConfig:
    min_delay_sec: float = 1.0
    max_delay_sec: float = 3.0


class HttpClient:
    def __init__(self, proxy_pool: ProxyPool | None = None,
                 retry_config: RetryConfig | None = None,
                 rate_limit_config: RateLimitConfig | None = None,
                 timeout: float = 30.0):
        self._proxy_pool = proxy_pool
        self._retry = retry_config or RetryConfig()
        self._rate_limit = rate_limit_config or RateLimitConfig()
        self._timeout = timeout

        self._request_count = 0
        self._success_count = 0
        self._failure_count = 0

    def get(self, url: str, params: dict | None = None) -> requests.Response | None:
        self._rate_limit_delay()

        for attempt in range(1, self._retry.max_retries + 1):
            proxy_url = self._proxy_pool.get_next() if self._proxy_pool else None

            if self._proxy_pool and self._proxy_pool.size > 0 and proxy_url is None:
                logger.error("no available proxies, waiting for unban")
                time.sleep(30)
                proxy_url = self._proxy_pool.get_next() if self._proxy_pool else None
                if proxy_url is None:
                    return None

            session = self._build_session(proxy_url)

            try:
                self._request_count += 1
                resp = session.get(url, params=params, timeout=self._timeout)
            except requests.RequestException as exc:
                self._failure_count += 1
                logger.warning("request failed (attempt %d/%d): %s - %s",
                               attempt, self._retry.max_retries, url, exc)
                if self._proxy_pool and proxy_url:
                    self._proxy_pool.report_failure(proxy_url, reason=type(exc).__name__)

                self._backoff(attempt)
                continue

            if resp.status_code == 429:
                logger.warning("rate limited on %s via proxy %s (attempt %d/%d)",
                               url, proxy_url or "direct", attempt, self._retry.max_retries)
                if self._proxy_pool and proxy_url:
                    self._proxy_pool.report_rate_limited(proxy_url)

                self._backoff(attempt)
                continue

            if resp.status_code in self._retry.retry_on_statuses:
                logger.warning("retryable status %d on %s (attempt %d/%d)",
                               resp.status_code, url, attempt, self._retry.max_retries)
                if self._proxy_pool and proxy_url:
                    self._proxy_pool.report_failure(proxy_url, reason=f"http_{resp.status_code}")

                self._backoff(attempt)
                continue

            if resp.status_code != 200:
                self._failure_count += 1
                logger.warning("non-retryable status %d on %s", resp.status_code, url)
                if self._proxy_pool and proxy_url:
                    self._proxy_pool.report_failure(proxy_url, reason=f"http_{resp.status_code}")
                return None

            self._success_count += 1
            if self._proxy_pool and proxy_url:
                self._proxy_pool.report_success(proxy_url)

            return resp

        self._failure_count += 1
        logger.error("exhausted all retries for %s", url)
        return None

    def get_stats(self) -> dict:
        return {
            "total_requests": self._request_count,
            "success": self._success_count,
            "failure": self._failure_count,
            "success_rate": round(self._success_count / max(1, self._request_count), 3),
        }

    def _build_session(self, proxy_url: str | None) -> requests.Session:
        session = requests.Session()
        session.headers.update({
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": random.choice(ACCEPT_LANGUAGES),
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Cache-Control": "max-age=0",
        })
        if proxy_url:
            session.proxies.update({"http": proxy_url, "https": proxy_url})
        return session

    def _rate_limit_delay(self) -> None:
        delay = random.uniform(self._rate_limit.min_delay_sec,
                               self._rate_limit.max_delay_sec)
        time.sleep(delay)

    def _backoff(self, attempt: int) -> None:
        delay = min(
            self._retry.base_delay_sec * (2 ** (attempt - 1)),
            self._retry.max_delay_sec,
        )
        jitter = delay * random.uniform(-self._retry.jitter_range, self._retry.jitter_range)
        total = max(0, delay + jitter)
        logger.info("backing off %.1fs before retry", total)
        time.sleep(total)
