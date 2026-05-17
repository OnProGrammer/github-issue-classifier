import os
import sqlite3
import json
import random
import argparse

import pandas as pd


LABEL2ID: dict[str, int] = {
    "BUG": 0,
    "FEATURE": 1,
    "DOCS": 2,
    "SUPPORT": 3,
}


class IssuesDatasetBuilder:
    def __init__(self, db_paths: dict[str, str], output_csv: str,
                 train_fraction: float = 0.8, seed: int = 42):
        self.db_paths = db_paths
        self.output_csv = output_csv
        self.train_fraction = train_fraction
        self.seed = seed

        random.seed(self.seed)
        self.df: pd.DataFrame | None = None

    def _load_from_db(self, db_path: str, expected_target: str) -> list[dict]:
        if not os.path.exists(db_path):
            return []

        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(
            "SELECT repo, issue_id, url, target, title_raw, body_raw, "
            "text_for_model, labels_raw, created_at, state FROM issues"
        )
        rows = cur.fetchall()
        conn.close()

        items = []
        for row in rows:
            text_for_model = row[6]
            if not text_for_model or not text_for_model.strip():
                continue

            try:
                labels_list = json.loads(row[7])
            except (json.JSONDecodeError, TypeError):
                labels_list = []

            items.append({
                "repo": row[0],
                "issue_id": row[1],
                "url": row[2],
                "target": expected_target,
                "title_raw": row[4],
                "body_raw": row[5],
                "text_for_model": text_for_model.strip(),
                "labels_raw": labels_list,
                "created_at": row[8],
                "state": row[9],
            })

        return items

    def build(self) -> pd.DataFrame:
        all_items: list[dict] = []
        for target, db_path in self.db_paths.items():
            all_items.extend(self._load_from_db(db_path, target))

        if not all_items:
            self.df = pd.DataFrame()
            return self.df

        records = []
        for item in all_items:
            target = item["target"]
            if target not in LABEL2ID:
                continue
            records.append({
                "issue_id": item["issue_id"],
                "repo": item["repo"],
                "url": item["url"],
                "text": item["text_for_model"].replace("\r\n", "\n").strip(),
                "label": LABEL2ID[target],
                "label_name": target,
                "created_at": item["created_at"],
                "state": item["state"],
            })

        random.shuffle(records)
        n_train = int(self.train_fraction * len(records))
        for i, rec in enumerate(records):
            rec["split"] = "train" if i < n_train else "test"

        self.df = pd.DataFrame(records)
        return self.df

    def save(self) -> None:
        if self.df is None or self.df.empty:
            return
        os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)
        self.df.to_csv(self.output_csv, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ML dataset from scraped issue DBs")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory with .db files")
    parser.add_argument("--output", type=str, default="out/issues.csv", help="Output CSV path")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    db_paths = {
        "BUG": os.path.join(args.data_dir, "issues_bug.db"),
        "FEATURE": os.path.join(args.data_dir, "issues_feature.db"),
        "DOCS": os.path.join(args.data_dir, "issues_docs.db"),
        "SUPPORT": os.path.join(args.data_dir, "issues_support.db"),
    }

    builder = IssuesDatasetBuilder(
        db_paths=db_paths,
        output_csv=args.output,
        train_fraction=args.train_fraction,
        seed=args.seed,
    )
    df = builder.build()
    builder.save()

    print(f"Dataset: {df.shape[0]} samples, {df['label_name'].nunique()} classes")
    print(f"Train: {(df['split'] == 'train').sum()}, Test: {(df['split'] == 'test').sum()}")
    print(df["label_name"].value_counts().to_string())
