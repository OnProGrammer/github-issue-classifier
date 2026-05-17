import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


LABEL_GROUPS: dict[str, list[str]] = {
    "BUG": ["bug", "regression", "failing-test", "flake"],
    "FEATURE": ["feature", "api-change", "deprecation"],
    "DOCS": ["documentation"],
    "SUPPORT": ["support"],
}

ALL_LABEL_SUFFIXES: dict[str, str] = {}
for _target, _suffixes in LABEL_GROUPS.items():
    for _s in _suffixes:
        ALL_LABEL_SUFFIXES[_s] = _target


@dataclass
class ScraperConfig:
    repo: str = "kubernetes/kubernetes"
    data_dir: str = "data"
    targets: list[str] = field(default_factory=lambda: ["BUG", "FEATURE", "DOCS", "SUPPORT"])
    start_page: int = 1
    num_pages: int = 10

    proxies: list[str] = field(default_factory=list)

    proxy_ban_threshold: int = 5
    proxy_ban_duration_sec: float = 300.0
    proxy_cooldown_sec: float = 2.0

    retry_max: int = 4
    retry_base_delay_sec: float = 2.0
    retry_max_delay_sec: float = 60.0

    rate_limit_min_sec: float = 1.0
    rate_limit_max_sec: float = 3.0

    request_timeout_sec: float = 30.0

    log_level: str = "INFO"
    log_json: bool = False

    checkpoint_file: str = ""

    @property
    def db_paths(self) -> dict[str, str]:
        return {
            t: os.path.join(self.data_dir, f"issues_{t.lower()}.db")
            for t in self.targets
        }

    @property
    def checkpoint_path(self) -> str:
        if self.checkpoint_file:
            return self.checkpoint_file
        return os.path.join(self.data_dir, "checkpoint.json")

    @classmethod
    def from_yaml(cls, path: str) -> "ScraperConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_env(cls) -> "ScraperConfig":
        kwargs: dict = {}

        if v := os.environ.get("SCRAPER_REPO"):
            kwargs["repo"] = v
        if v := os.environ.get("SCRAPER_DATA_DIR"):
            kwargs["data_dir"] = v
        if v := os.environ.get("SCRAPER_TARGETS"):
            kwargs["targets"] = [t.strip() for t in v.split(",")]
        if v := os.environ.get("SCRAPER_PROXIES"):
            kwargs["proxies"] = [p.strip() for p in v.split(",") if p.strip()]
        if v := os.environ.get("SCRAPER_LOG_LEVEL"):
            kwargs["log_level"] = v
        if os.environ.get("SCRAPER_LOG_JSON", "").lower() in ("1", "true", "yes"):
            kwargs["log_json"] = True

        return cls(**kwargs)

    def to_yaml(self, path: str) -> None:
        data = {
            "repo": self.repo,
            "data_dir": self.data_dir,
            "targets": self.targets,
            "start_page": self.start_page,
            "num_pages": self.num_pages,
            "proxies": self.proxies,
            "proxy_ban_threshold": self.proxy_ban_threshold,
            "proxy_ban_duration_sec": self.proxy_ban_duration_sec,
            "proxy_cooldown_sec": self.proxy_cooldown_sec,
            "retry_max": self.retry_max,
            "retry_base_delay_sec": self.retry_base_delay_sec,
            "retry_max_delay_sec": self.retry_max_delay_sec,
            "rate_limit_min_sec": self.rate_limit_min_sec,
            "rate_limit_max_sec": self.rate_limit_max_sec,
            "request_timeout_sec": self.request_timeout_sec,
            "log_level": self.log_level,
            "log_json": self.log_json,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
