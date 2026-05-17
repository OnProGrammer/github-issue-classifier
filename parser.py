import argparse
import os
import sys

from scraper.config import ScraperConfig
from scraper.net.logging_config import setup_logging
from scraper.issue_scraper import IssueScraper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape GitHub issues with proxy rotation, retry, and checkpointing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    parser.add_argument("--dump-config", type=str, metavar="PATH",
                        help="Generate a YAML config template and exit")

    scrape_group = parser.add_argument_group("scraping")
    scrape_group.add_argument("--repo", type=str, help="GitHub repo (owner/name)")
    scrape_group.add_argument("--data-dir", type=str, help="Directory for DBs and raw data")
    scrape_group.add_argument("--targets", nargs="+",
                              choices=["BUG", "FEATURE", "DOCS", "SUPPORT"],
                              help="Target classes to scrape")
    scrape_group.add_argument("--start-page", type=int)
    scrape_group.add_argument("--num-pages", type=int)

    proxy_group = parser.add_argument_group("proxy")
    proxy_group.add_argument("--proxies", nargs="*", help="Proxy URLs")
    proxy_group.add_argument("--proxy-file", type=str,
                             help="File with proxy URLs, one per line")

    log_group = parser.add_argument_group("logging")
    log_group.add_argument("--log-level", type=str,
                           choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    log_group.add_argument("--log-json", action="store_true",
                           help="Output logs in JSON format")

    parser.add_argument("--fresh", action="store_true",
                        help="Ignore existing checkpoint and start from scratch")

    return parser.parse_args()


def load_proxies_from_file(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def build_config(args: argparse.Namespace) -> ScraperConfig:
    if args.config:
        config = ScraperConfig.from_yaml(args.config)
    else:
        config = ScraperConfig()

    if args.repo:
        config.repo = args.repo
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.targets:
        config.targets = args.targets
    if args.start_page is not None:
        config.start_page = args.start_page
    if args.num_pages is not None:
        config.num_pages = args.num_pages
    if args.log_level:
        config.log_level = args.log_level
    if args.log_json:
        config.log_json = True

    if args.proxy_file:
        config.proxies = load_proxies_from_file(args.proxy_file)
    elif args.proxies:
        config.proxies = args.proxies

    return config


def main() -> None:
    args = parse_args()

    if args.dump_config:
        config = ScraperConfig()
        config.to_yaml(args.dump_config)
        print(f"Config template written to {args.dump_config}")
        sys.exit(0)

    config = build_config(args)
    setup_logging(level=config.log_level, json_output=config.log_json)

    if args.fresh:
        progress_path = config.checkpoint_path
        if os.path.exists(progress_path):
            os.remove(progress_path)

    scraper = IssueScraper(config)
    scraper.run()


if __name__ == "__main__":
    main()
