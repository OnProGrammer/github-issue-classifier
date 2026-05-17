import sqlite3
import json
import os
import logging

logger = logging.getLogger(__name__)


class IssueStorage:
    def __init__(self, db_paths: dict[str, str]):
        self._db_paths = db_paths
        self._connections: dict[str, sqlite3.Connection] = {}

    def open(self) -> None:
        for target, path in self._db_paths.items():
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            conn = sqlite3.connect(path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS issues (
                    repo TEXT NOT NULL,
                    issue_id INTEGER NOT NULL,
                    url TEXT NOT NULL,
                    target TEXT NOT NULL,
                    title_raw TEXT NOT NULL,
                    body_raw TEXT NOT NULL,
                    text_for_model TEXT NOT NULL,
                    labels_raw TEXT NOT NULL,
                    created_at TEXT,
                    state TEXT,
                    PRIMARY KEY (repo, issue_id)
                );
            """)
            conn.commit()
            self._connections[target] = conn
            logger.debug("opened db for %s at %s", target, path)

    def close(self) -> None:
        for target, conn in self._connections.items():
            conn.close()
        self._connections.clear()

    def __enter__(self) -> "IssueStorage":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def upsert(self, target: str, issue: dict) -> None:
        conn = self._connections.get(target)
        if conn is None:
            raise ValueError(f"no db connection for target '{target}'")

        labels_json = json.dumps(issue["labels_raw"], ensure_ascii=False)
        conn.execute(
            """
            INSERT INTO issues (repo, issue_id, url, target, title_raw, body_raw,
                                text_for_model, labels_raw, created_at, state)
            VALUES (:repo, :issue_id, :url, :target, :title_raw, :body_raw,
                    :text_for_model, :labels_raw, :created_at, :state)
            ON CONFLICT(repo, issue_id) DO UPDATE SET
                url = excluded.url,
                target = excluded.target,
                title_raw = excluded.title_raw,
                body_raw = excluded.body_raw,
                text_for_model = excluded.text_for_model,
                labels_raw = excluded.labels_raw,
                created_at = excluded.created_at,
                state = excluded.state;
            """,
            {
                "repo": issue["repo"],
                "issue_id": issue["issue_id"],
                "url": issue["url"],
                "target": issue["target"],
                "title_raw": issue["title_raw"],
                "body_raw": issue["body_raw"],
                "text_for_model": issue["text_for_model"],
                "labels_raw": labels_json,
                "created_at": issue.get("created_at"),
                "state": issue.get("state"),
            },
        )
        conn.commit()

    def load_existing_ids(self, repo: str) -> set[int]:
        ids: set[int] = set()
        for conn in self._connections.values():
            rows = conn.execute(
                "SELECT issue_id FROM issues WHERE repo = ?", (repo,)
            ).fetchall()
            ids.update(row[0] for row in rows)
        logger.info("loaded %d existing issue ids from db", len(ids))
        return ids

    def count_by_target(self) -> dict[str, int]:
        counts = {}
        for target, conn in self._connections.items():
            row = conn.execute("SELECT COUNT(*) FROM issues").fetchone()
            counts[target] = row[0] if row else 0
        return counts
