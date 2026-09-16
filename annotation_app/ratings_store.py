"""
Persistent storage for human annotation ratings, one CSV file per
rater, so multiple raters' sessions never collide and each rater's
raw input is independently auditable.
"""
import csv
import os
from dataclasses import dataclass

import numpy as np

RATINGS_DIR = os.path.join(os.path.dirname(__file__), "data", "ratings")
FIELDNAMES = ["sample_id", "fairness_score", "soundness_score", "notes"]


@dataclass
class Rating:
    sample_id: str
    fairness_score: int
    soundness_score: int
    notes: str = ""


def rater_csv_path(rater_name: str) -> str:
    os.makedirs(RATINGS_DIR, exist_ok=True)
    safe_name = "".join(c for c in rater_name if c.isalnum() or c in ("-", "_")).lower()
    return os.path.join(RATINGS_DIR, f"{safe_name}.csv")


def load_ratings(rater_name: str) -> dict[str, Rating]:
    path = rater_csv_path(rater_name)
    if not os.path.exists(path):
        return {}
    ratings = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ratings[row["sample_id"]] = Rating(
                sample_id=row["sample_id"],
                fairness_score=int(row["fairness_score"]),
                soundness_score=int(row["soundness_score"]),
                notes=row.get("notes", ""),
            )
    return ratings


def save_rating(rater_name: str, rating: Rating) -> None:
    """Upsert one rating into this rater's CSV (overwrites if sample_id already rated)."""
    existing = load_ratings(rater_name)
    existing[rating.sample_id] = rating

    path = rater_csv_path(rater_name)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for r in existing.values():
            writer.writerow(
                {
                    "sample_id": r.sample_id,
                    "fairness_score": r.fairness_score,
                    "soundness_score": r.soundness_score,
                    "notes": r.notes,
                }
            )


def list_raters() -> list[str]:
    if not os.path.exists(RATINGS_DIR):
        return []
    return [f[:-4] for f in os.listdir(RATINGS_DIR) if f.endswith(".csv")]


def build_ratings_matrix(rater_names: list[str], sample_ids: list[str], score_field: str) -> np.ndarray:
    """
    Build a (n_raters, n_items) matrix suitable for
    evaluation/agreement_stats.py, with np.nan for any sample a given
    rater has not yet scored.
    """
    matrix = np.full((len(rater_names), len(sample_ids)), np.nan)
    for i, rater in enumerate(rater_names):
        ratings = load_ratings(rater)
        for j, sid in enumerate(sample_ids):
            if sid in ratings:
                matrix[i, j] = getattr(ratings[sid], score_field)
    return matrix
