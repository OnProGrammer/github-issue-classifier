import signal
import logging

from scraper.config import ScraperConfig, LABEL_GROUPS
from scraper.net.proxy_pool import ProxyPool, ProxyPoolConfig
from scraper.net.http_client import HttpClient, RetryConfig, RateLimitConfig
from scraper.storage.sqlite_storage import IssueStorage
from scraper.storage.checkpoint import ScrapeProgress
from scraper.parsing.github_parser import (
    has_no_results,
    parse_issue_list,
    parse_issue_body,
    parse_issue_meta,
    build_text_for_model,
    classify_target,
)

logger = logging.getLogger(__name__)


class IssueScraper:
    def __init__(self, config: ScraperConfig):
        self._config = config
        self._shutdown_requested = False

        self._proxy_pool = ProxyPool(
            proxy_urls=config.proxies,
            config=ProxyPoolConfig(
                ban_threshold=config.proxy_ban_threshold,
                ban_duration_sec=config.proxy_ban_duration_sec,
                cooldown_sec=config.proxy_cooldown_sec,
            ),
        ) if config.proxies else None

        self._client = HttpClient(
            proxy_pool=self._proxy_pool,
            retry_config=RetryConfig(
                max_retries=config.retry_max,
                base_delay_sec=config.retry_base_delay_sec,
                max_delay_sec=config.retry_max_delay_sec,
            ),
            rate_limit_config=RateLimitConfig(
                min_delay_sec=config.rate_limit_min_sec,
                max_delay_sec=config.rate_limit_max_sec,
            ),
            timeout=config.request_timeout_sec,
        )

        self._storage = IssueStorage(config.db_paths)
        self._progress = ScrapeProgress()

    def run(self) -> None:
        self._install_signal_handlers()
        self._progress = ScrapeProgress.load(self._config.checkpoint_path)

        with self._storage:
            existing_ids = self._storage.load_existing_ids(self._config.repo)

            logger.info("starting scrape: targets=%s, pages=%d-%d, proxies=%d",
                        self._config.targets,
                        self._config.start_page,
                        self._config.start_page + self._config.num_pages - 1,
                        len(self._config.proxies))

            db_counts = self._storage.count_by_target()
            for target, count in db_counts.items():
                logger.info("  %s: %d existing issues in db", target, count)

            try:
                self._scrape_all(existing_ids)
            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt received")
            finally:
                self._progress.save(self._config.checkpoint_path)
                self._log_final_stats()

    def _scrape_all(self, existing_ids: set[int]) -> None:
        last_page = self._config.start_page + self._config.num_pages - 1
        base_url = f"https://github.com/{self._config.repo}/issues"

        for target in self._config.targets:
            if self._shutdown_requested:
                break

            label_suffixes = LABEL_GROUPS.get(target, [])
            if not label_suffixes:
                logger.warning("no label suffixes for target %s, skipping", target)
                continue

            logger.info("=== scraping target: %s (labels: %s) ===", target, label_suffixes)

            for label_suffix in label_suffixes:
                if self._shutdown_requested:
                    break

                for page in range(self._config.start_page, last_page + 1):
                    if self._shutdown_requested:
                        break

                    if self._progress.is_done(target, label_suffix, page):
                        logger.debug("skipping %s/%s page %d (checkpoint)",
                                     target, label_suffix, page)
                        continue

                    issues_saved = self._scrape_page(
                        base_url, target, label_suffix, page, existing_ids
                    )

                    self._progress.mark_done(target, label_suffix, page)

                    if issues_saved is None:
                        break

                    if self._progress.scraped_issue_count % 50 == 0 and self._progress.scraped_issue_count > 0:
                        self._progress.save(self._config.checkpoint_path)
                        self._log_progress()

    def _scrape_page(self, base_url: str, target: str,
                     label_suffix: str, page: int,
                     existing_ids: set[int]) -> int | None:
        q_str = f"is:issue state:closed sort:created-desc label:kind/{label_suffix}"
        params = {"q": q_str, "page": str(page)}

        logger.info("fetching %s kind/%s page %d", target, label_suffix, page)

        resp = self._client.get(base_url, params=params)
        if resp is None:
            logger.warning("failed to fetch list page %s/%s/%d", target, label_suffix, page)
            return 0

        html = resp.content.decode("utf-8", errors="ignore")

        if has_no_results(html):
            logger.info("no more results for %s/kind/%s at page %d", target, label_suffix, page)
            return None

        issues_on_page = parse_issue_list(html, self._config.repo)
        if not issues_on_page:
            logger.debug("no issues parsed from page %d", page)
            return 0

        saved = 0
        for meta in issues_on_page:
            if self._shutdown_requested:
                break

            issue_id = meta["issue_id"]
            if issue_id in existing_ids:
                continue

            classified = classify_target(meta["labels_raw"])
            if classified != target:
                continue

            issue_row = self._fetch_and_parse_issue(meta, target)
            if issue_row is None:
                continue

            self._storage.upsert(target, issue_row)
            existing_ids.add(issue_id)
            self._progress.scraped_issue_count += 1
            saved += 1

            logger.info("saved issue #%d [%s], total: %d",
                        issue_id, target, self._progress.scraped_issue_count)

        return saved

    def _fetch_and_parse_issue(self, meta: dict, target: str) -> dict | None:
        resp = self._client.get(meta["url"])
        if resp is None:
            logger.warning("failed to fetch issue #%d", meta["issue_id"])
            return None

        html = resp.content.decode("utf-8", errors="ignore")
        body_raw = parse_issue_body(html)
        page_meta = parse_issue_meta(html)

        return {
            "repo": self._config.repo,
            "issue_id": meta["issue_id"],
            "url": meta["url"],
            "target": target,
            "title_raw": meta["title_raw"],
            "body_raw": body_raw,
            "text_for_model": build_text_for_model(meta["title_raw"], body_raw),
            "labels_raw": meta["labels_raw"],
            "created_at": page_meta["created_at"],
            "state": page_meta["state"] or "closed",
        }

    def _install_signal_handlers(self) -> None:
        def handler(signum, frame):
            sig_name = signal.Signals(signum).name
            logger.warning("received %s, finishing current task and saving checkpoint", sig_name)
            self._shutdown_requested = True

        signal.signal(signal.SIGINT, handler)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, handler)

    def _log_progress(self) -> None:
        client_stats = self._client.get_stats()
        logger.info(
            "progress: %d issues scraped | http: %d requests, %.1f%% success",
            self._progress.scraped_issue_count,
            client_stats["total_requests"],
            client_stats["success_rate"] * 100,
        )

        if self._proxy_pool:
            proxy_stats = self._proxy_pool.get_stats()
            for ps in proxy_stats:
                status = "BANNED" if ps["banned"] else "OK"
                logger.info(
                    "  proxy %s: %s (ok=%d, fail=%d, rate=%.1f%%)",
                    ps["url"][:30] + "...", status,
                    ps["success"], ps["fail"],
                    (1 - ps["failure_rate"]) * 100,
                )

    def _log_final_stats(self) -> None:
        logger.info("=" * 60)
        logger.info("scrape finished%s", " (interrupted)" if self._shutdown_requested else "")
        logger.info("total issues scraped this run: %d", self._progress.scraped_issue_count)
        logger.info("completed pages: %d", len(self._progress.completed))

        client_stats = self._client.get_stats()
        logger.info("http stats: %s", client_stats)

        db_counts = self._storage.count_by_target()
        for target, count in db_counts.items():
            logger.info("  db %s: %d issues", target, count)

        logger.info("=" * 60)
