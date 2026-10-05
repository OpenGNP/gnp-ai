"""All SQL the pipeline runs. No other module talks to Postgres.

The pipeline is the second writer on a database the server owns, so every
statement here is deliberately narrow:

  * it only inserts points for answers that have none;
  * it only clusters points that carry an embedding, which the seeded demo
    points do not — so seeded rows and their topic assignments are never
    disturbed;
  * it only deletes canonical_topics that a previous pipeline run created,
    identified by the PIPELINE_TAG marker in topic_versions.model_version.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from . import config


def as_array(value) -> np.ndarray:
    """pgvector hands back a Vector object (or a list, depending on version);
    every consumer here wants plain float32."""
    if isinstance(value, np.ndarray):
        return value.astype(np.float32)
    if hasattr(value, "to_numpy"):
        return value.to_numpy().astype(np.float32)
    return np.asarray(value, dtype=np.float32)


@dataclass
class ClusterablePoint:
    id: int
    text: str
    embedding: np.ndarray
    sentiment_label: str | None
    is_severe: bool | None
    created_at: datetime


def connect() -> psycopg.Connection:
    """autocommit is on deliberately.

    With autocommit off, the first read opens an implicit transaction, and a
    later `with conn.transaction()` degrades to a savepoint that never commits
    on its own — the writes are then discarded when the connection closes.
    With autocommit on, reads stand alone and every `with conn.transaction()`
    block is a real atomic unit, which is what the cluster stage needs.
    """
    conn = psycopg.connect(config.require_database_url(), autocommit=True)
    register_vector(conn)
    return conn


# ── ingest ─────────────────────────────────────────────────────────────

def pending_answers(conn: psycopg.Connection, limit: int | None = None) -> list[tuple[int, str]]:
    """Free-text answers with no points row yet.

    Radio/checkbox answers store answer_option_id and leave answer_text NULL,
    so the NOT NULL filter skips them without needing to join form_fields.
    """
    sql = """
        SELECT a.id, a.answer_text
        FROM answers a
        LEFT JOIN points p ON p.answer_id = a.id
        WHERE a.answer_text IS NOT NULL AND btrim(a.answer_text) <> '' AND p.id IS NULL
        ORDER BY a.id
    """
    if limit:
        sql += f" LIMIT {int(limit)}"
    with conn.cursor() as cur:
        cur.execute(sql)
        return [(row[0], row[1]) for row in cur.fetchall()]


def insert_points(conn: psycopg.Connection, rows: list[dict]) -> int:
    """processing_status is constrained by chk_processing to
    pending | assigned | unknown | clustered, and sentiment_label by
    chk_sentiment to positive | neutral | negative (NULL allowed)."""
    if not rows:
        return 0
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO points
                (answer_id, point_text, embedding, sentiment_label, is_severe, processing_status)
            VALUES (%(answer_id)s, %(point_text)s, %(embedding)s, %(sentiment_label)s,
                    %(is_severe)s, %(processing_status)s)
            """,
            rows,
        )
    return len(rows)


def points_missing_features(conn: psycopg.Connection) -> list[tuple[int, str]]:
    """Existing points with no embedding — the seeded rows. Only touched by --backfill."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, point_text FROM points
            WHERE embedding IS NULL AND point_text IS NOT NULL
            ORDER BY id
            """
        )
        return [(row[0], row[1]) for row in cur.fetchall()]


def points_for_relabel(conn: psycopg.Connection) -> list[tuple[int, str]]:
    """Points the pipeline owns (it is the only writer of embeddings), for
    re-running the classifiers after a model change. Seeded points have no
    embedding and keep their original labels."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, point_text FROM points
            WHERE embedding IS NOT NULL AND point_text IS NOT NULL
            ORDER BY id
            """
        )
        return [(row[0], row[1]) for row in cur.fetchall()]


def update_point_labels(conn: psycopg.Connection, rows: list[dict]) -> int:
    if not rows:
        return 0
    with conn.cursor() as cur:
        cur.executemany(
            """
            UPDATE points
            SET sentiment_label = %(sentiment_label)s, is_severe = %(is_severe)s
            WHERE id = %(id)s
            """,
            rows,
        )
    return len(rows)


def update_point_features(conn: psycopg.Connection, rows: list[dict]) -> int:
    if not rows:
        return 0
    with conn.cursor() as cur:
        cur.executemany(
            """
            UPDATE points
            SET embedding = %(embedding)s,
                sentiment_label = COALESCE(%(sentiment_label)s, sentiment_label),
                is_severe = COALESCE(%(is_severe)s, is_severe),
                processing_status = %(processing_status)s
            WHERE id = %(id)s
            """,
            rows,
        )
    return len(rows)


# ── clustering ─────────────────────────────────────────────────────────

def clusterable_points(conn: psycopg.Connection) -> list[ClusterablePoint]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, point_text, embedding, sentiment_label, is_severe, created_at
            FROM points
            WHERE embedding IS NOT NULL AND point_text IS NOT NULL
            ORDER BY id
            """
        )
        return [ClusterablePoint(*row) for row in cur.fetchall()]


def pipeline_topic_ids(conn: psycopg.Connection) -> list[int]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT canonical_topic_id FROM topic_versions WHERE model_version = %s",
            (config.PIPELINE_TAG,),
        )
        return [row[0] for row in cur.fetchall()]


def delete_topics(conn: psycopg.Connection, topic_ids: list[int]) -> int:
    """Deleting a topic cascades to topic_trends and topic_versions, and sets
    points.canonical_topic_id back to NULL (fk_point_topic is ON DELETE SET NULL)."""
    if not topic_ids:
        return 0
    with conn.cursor() as cur:
        cur.execute("DELETE FROM canonical_topics WHERE id = ANY(%s)", (topic_ids,))
        return cur.rowcount


def insert_topic(
    conn: psycopg.Connection,
    *,
    name: str,
    summary: str | None,
    keywords: str,
    centroid: np.ndarray,
    size: int,
    first_seen: datetime,
    last_seen: datetime,
) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO canonical_topics
                (canonical_name, canonical_summary, representative_keywords, centroid_vector,
                 topic_size, status, first_detected_at, last_updated_at)
            VALUES (%s, %s, %s, %s, %s, 'active', %s, %s)
            RETURNING id
            """,
            (name, summary, keywords, centroid, size, first_seen, last_seen),
        )
        topic_id = cur.fetchone()[0]
        # The ownership marker. Without it a later run cannot tell this topic
        # apart from one the server seeded, and would have to leave it alone.
        cur.execute(
            """
            INSERT INTO topic_versions
                (canonical_topic_id, generated_title, generated_summary,
                 representative_keywords, model_version)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (topic_id, name, summary, keywords, config.PIPELINE_TAG),
        )
    return topic_id


def assign_points(conn: psycopg.Connection, rows: list[dict]) -> int:
    if not rows:
        return 0
    with conn.cursor() as cur:
        cur.executemany(
            """
            UPDATE points
            SET canonical_topic_id = %(topic_id)s,
                assignment_confidence = %(confidence)s,
                processing_status = %(processing_status)s
            WHERE id = %(id)s
            """,
            rows,
        )
    return len(rows)


def insert_trend(
    conn: psycopg.Connection,
    *,
    topic_id: int,
    period_start: datetime,
    period_end: datetime,
    counts: dict[str, int],
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO topic_trends
                (canonical_topic_id, period_start, period_end, feedback_count,
                 positive_count, neutral_count, negative_count, severe_count)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                topic_id, period_start, period_end, counts["total"],
                counts["positive"], counts["neutral"], counts["negative"], counts["severe"],
            ),
        )


def clear_unassigned(conn: psycopg.Connection, point_ids: list[int]) -> int:
    if not point_ids:
        return 0
    with conn.cursor() as cur:
        cur.execute("DELETE FROM unassigned_points WHERE point_id = ANY(%s)", (point_ids,))
        return cur.rowcount


def insert_unassigned(conn: psycopg.Connection, rows: list[dict]) -> int:
    if not rows:
        return 0
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO unassigned_points (point_id, embedding, reason)
            VALUES (%(point_id)s, %(embedding)s, %(reason)s)
            """,
            rows,
        )
    return len(rows)


def record_run(
    conn: psycopg.Connection,
    *,
    parameters: str,
    started_at: datetime,
    completed_at: datetime,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO ai_model_runs
                (model_name, model_version, embedding_model, clustering_algorithm,
                 parameters, started_at, completed_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                config.DECOMPOSE_MODEL, config.DECOMPOSE_PROMPT, config.EMBEDDING_MODEL,
                "BERTopic+HDBSCAN", parameters, started_at, completed_at,
            ),
        )
