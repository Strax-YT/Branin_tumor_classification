"""
Persistent prediction history, SQLite-backed.

Deliberately minimal: one table, no ORM, one short-lived connection per
call. This is a single-user demo API, not a production service under
concurrent write load -- this is proportionate, not a placeholder.

Privacy: only prediction metadata is stored. No image bytes, filename,
or uploader identity ever touches this table.
"""

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.environ.get("NEUROSIGHT_DB_PATH", str(PROJECT_ROOT / "neurosight.db")))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    predicted_class TEXT NOT NULL,
    confidence REAL NOT NULL,
    entropy REAL NOT NULL,
    top_two_gap REAL NOT NULL,
    reliability_score REAL NOT NULL,
    reliability_tier TEXT NOT NULL,
    review_recommended INTEGER NOT NULL,
    quality_score REAL NOT NULL,
    quality_status TEXT NOT NULL,
    inference_time_ms REAL NOT NULL,
    model_version TEXT NOT NULL
);
"""


def _connect():
    return sqlite3.connect(DB_PATH)


def init_db():
    with _connect() as conn:
        conn.execute(_SCHEMA)


def record_prediction(row):
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO predictions (
                timestamp, predicted_class, confidence, entropy, top_two_gap,
                reliability_score, reliability_tier, review_recommended,
                quality_score, quality_status, inference_time_ms, model_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                row["predicted_class"],
                row["confidence"],
                row["entropy"],
                row["top_two_gap"],
                row["reliability_score"],
                row["reliability_tier"],
                int(row["review_recommended"]),
                row["quality_score"],
                row["quality_status"],
                row["inference_time_ms"],
                row["model_version"],
            ),
        )


def get_history(limit=50, offset=0):
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            """
            SELECT id, timestamp, predicted_class, confidence, reliability_score,
                   reliability_tier, review_recommended, inference_time_ms
            FROM predictions
            ORDER BY id DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        )
        rows = [dict(r) for r in cursor.fetchall()]
        for r in rows:
            r["review_recommended"] = bool(r["review_recommended"])
        return rows


def get_metrics():
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        total = conn.execute("SELECT COUNT(*) AS n FROM predictions").fetchone()["n"]

        if total == 0:
            return {
                "total_predictions": 0,
                "average_inference_time_ms": 0.0,
                "average_confidence": 0.0,
                "class_distribution": {},
                "review_recommended_count": 0,
                "poor_quality_count": 0,
            }

        aggregates = conn.execute(
            """
            SELECT AVG(inference_time_ms) AS avg_time,
                   AVG(confidence) AS avg_confidence,
                   SUM(review_recommended) AS review_count,
                   SUM(CASE WHEN quality_status = 'Poor' THEN 1 ELSE 0 END) AS poor_count
            FROM predictions
            """
        ).fetchone()

        distribution = conn.execute(
            "SELECT predicted_class, COUNT(*) AS n FROM predictions GROUP BY predicted_class"
        ).fetchall()

        return {
            "total_predictions": total,
            "average_inference_time_ms": round(aggregates["avg_time"], 1),
            "average_confidence": round(aggregates["avg_confidence"], 4),
            "class_distribution": {r["predicted_class"]: r["n"] for r in distribution},
            "review_recommended_count": aggregates["review_count"],
            "poor_quality_count": aggregates["poor_count"],
        }
