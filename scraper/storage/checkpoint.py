import json
import os
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ScrapeProgress:
    completed: set[str] = field(default_factory=set)
    scraped_issue_count: int = 0

    def mark_done(self, target: str, label_suffix: str, page: int) -> None:
        key = f"{target}:{label_suffix}:{page}"
        self.completed.add(key)

    def is_done(self, target: str, label_suffix: str, page: int) -> bool:
        key = f"{target}:{label_suffix}:{page}"
        return key in self.completed

    def save(self, path: str) -> None:
        data = {
            "completed": sorted(self.completed),
            "scraped_issue_count": self.scraped_issue_count,
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)

        logger.debug("checkpoint saved: %d pages done, %d issues",
                     len(self.completed), self.scraped_issue_count)

    @classmethod
    def load(cls, path: str) -> "ScrapeProgress":
        if not os.path.exists(path):
            logger.info("no checkpoint at %s, starting fresh", path)
            return cls()

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        progress = cls(
            completed=set(data.get("completed", [])),
            scraped_issue_count=data.get("scraped_issue_count", 0),
        )
        logger.info("resumed from checkpoint: %d pages done, %d issues scraped",
                     len(progress.completed), progress.scraped_issue_count)
        return progress

    def clear(self, path: str) -> None:
        self.completed.clear()
        self.scraped_issue_count = 0
        if os.path.exists(path):
            os.remove(path)
            logger.info("checkpoint cleared")
