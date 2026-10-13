"""All SQL the pipeline runs. No other module talks to Postgres.

The pipeline is the second writer on a database the server owns, so every
statement here is deliberately narrow:

  * it only inserts points for answers that have none;
  * it only clusters points that carry an embedding, which the seeded demo
    points do not — so seeded rows and their topic assignments are never
    disturbed;
  * it only touches canonical_topics belonging to the form it is processing,
    which canonical_topics.form_id makes explicit.
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
    # The topic this point belonged to before this run. Carrying it into memory
    # is what lets a re-cluster recognise an old topic by its membership.
    canonical_topic_id: int | None = None


def _dial() -> psycopg.Connection:
    """autocommit is on deliberately.

    With autocommit off, the first read opens an implicit transaction, and a
    later `with conn.transaction()` degrades to a savepoint that never commits
    on its own — the writes are then discarded when the connection closes.
    With autocommit on, reads stand alone and every `with conn.transaction()`
    block is a real atomic unit, which is what the cluster stage needs.
    """
    conn = psycopg.connect(config.require_database_url(), autocommit=True)
    register_vector(conn)
    with conn.cursor() as cur:
        # public stays on the path behind the app schema: the pgvector extension
        # lives there, so dropping it hides vector_dims() and the <=> operators
        # even though the column type still resolves.
        cur.execute(f'SET search_path TO "{config.DB_SCHEMA}", public')
    return conn


class Connection:
    """A psycopg connection that re-dials when the network has dropped it.

    The job spends minutes at a time inside Ollama and the CPU classifiers
    without touching Postgres. A connection left idle that long over a VPN
    does not reliably survive — it surfaces as

        OperationalError: could not receive data from server

    on the next statement, which is nowhere near the code that could recover.
    Re-dialling here means every caller keeps the connection it was handed.

    It only ever re-dials between transactions. A connection that dies inside
    one has to fail loudly: replacing it would silently drop the statements
    already issued and leave the rest to run against a fresh, empty
    transaction.
    """

    def __init__(self) -> None:
        self._conn = _dial()

    def _live(self) -> psycopg.Connection:
        if not self._conn.closed:
            if self._conn.info.transaction_status != psycopg.pq.TransactionStatus.IDLE:
                return self._conn  # mid-transaction, or already broken inside one
            try:
                with self._conn.cursor() as cur:
                    cur.execute("SELECT 1")
                return self._conn
            except psycopg.Error:
                pass
        try:
            self._conn.close()
        except psycopg.Error:
            pass
        self._conn = _dial()
        return self._conn

    def cursor(self, *args, **kwargs):
        return self._live().cursor(*args, **kwargs)

    def transaction(self, *args, **kwargs):
        return self._live().transaction(*args, **kwargs)

    def close(self) -> None:
        self._conn.close()


def connect() -> Connection:
    return Connection()


def embedding_dims(conn: Connection) -> int | None:
    """Dimension the points.embedding column is declared with, or None when the
    column is unconstrained. pgvector rejects any other width, so the encoder
    output has to be fitted to this."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT format_type(a.atttypid, a.atttypmod)
            FROM pg_attribute a
            JOIN pg_class c ON c.oid = a.attrelid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = %s AND c.relname = 'points' AND a.attname = 'embedding'
            """,
            (config.DB_SCHEMA,),
        )
        row = cur.fetchone()

    if not row or "(" not in row[0]:
        return None
    return int(row[0].split("(")[1].rstrip(")"))


# ── ingest ─────────────────────────────────────────────────────────────

def analysable_fields(conn: Connection) -> list[tuple[int, int, str]]:
    """(form_id, field_id, label) for every field the form marked for analysis.

    `analyze_with_ai` replaces the old guess of "any field with free text":
    the form author now says which questions the pipeline should read. This is
    the unit of the ingest stage, because an answer belongs to one field.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT f.form_id, f.id, f.field_label
            FROM form_fields f
            WHERE f.analyze_with_ai IS TRUE AND f.deleted_at IS NULL
            ORDER BY f.form_id, f.id
            """
        )
        return [(row[0], row[1], row[2]) for row in cur.fetchall()]


def analysable_forms(conn: Connection) -> list[tuple[int, str]]:
    """(form_id, form_title) for every form with at least one analysed field.

    This is the unit of the cluster stage. A form is one survey, so the answers
    to all of its analysed questions are one body of feedback: a complaint
    about room temperature is the same issue whether it arrived through the
    curriculum question or the facilities one. Clustering per form also gives
    HDBSCAN the corpus the dashboard displays, which is what the parameters
    that scale with corpus size are tuned against.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT fo.id, fo.form_title
            FROM forms fo
            JOIN form_fields f ON f.form_id = fo.id
            WHERE f.analyze_with_ai IS TRUE AND f.deleted_at IS NULL
            ORDER BY fo.id
            """
        )
        return [(row[0], row[1]) for row in cur.fetchall()]


def pending_answers(
    conn: Connection, field_id: int, limit: int | None = None
) -> list[tuple[int, str]]:
    """Free-text answers with no points row yet.

    Radio/checkbox answers store answer_option_id and leave answer_text NULL,
    so the NOT NULL filter skips them without needing to join form_fields.
    """
    sql = """
        SELECT a.id, a.answer_text
        FROM answers a
        LEFT JOIN points p ON p.answer_id = a.id
        WHERE a.field_id = %s
          AND a.answer_text IS NOT NULL AND btrim(a.answer_text) <> ''
          AND p.id IS NULL
        ORDER BY a.id
    """
    if limit:
        sql += f" LIMIT {int(limit)}"
    with conn.cursor() as cur:
        cur.execute(sql, (field_id,))
        return [(row[0], row[1]) for row in cur.fetchall()]


def insert_points(conn: Connection, rows: list[dict]) -> int:
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


# Points belong to a form only transitively: points -> answers -> submissions -> forms.
_FORM_FILTER = """
    AND p.answer_id IN (
        SELECT a.id FROM answers a
        JOIN submissions s ON s.id = a.submission_id
        WHERE s.form_id = ANY(%s)
    )
"""


def points_missing_features(
    conn: Connection, form_ids: list[int] | None = None
) -> list[tuple[int, str]]:
    """Existing points with no embedding — the seeded rows. Only touched by --backfill."""
    sql = """
        SELECT p.id, p.point_text FROM points p
        WHERE p.embedding IS NULL AND p.point_text IS NOT NULL
    """
    params: list = []
    if form_ids:
        sql += _FORM_FILTER
        params.append(form_ids)
    sql += " ORDER BY p.id"

    with conn.cursor() as cur:
        cur.execute(sql, params)
        return [(row[0], row[1]) for row in cur.fetchall()]


def points_for_relabel(
    conn: Connection, form_ids: list[int] | None = None
) -> list[tuple[int, str]]:
    """Points the pipeline owns (it is the only writer of embeddings), for
    re-running the classifiers after a model change. Seeded points have no
    embedding and keep their original labels."""
    sql = """
        SELECT p.id, p.point_text FROM points p
        WHERE p.embedding IS NOT NULL AND p.point_text IS NOT NULL
    """
    params: list = []
    if form_ids:
        sql += _FORM_FILTER
        params.append(form_ids)
    sql += " ORDER BY p.id"

    with conn.cursor() as cur:
        cur.execute(sql, params)
        return [(row[0], row[1]) for row in cur.fetchall()]


def update_point_labels(conn: Connection, rows: list[dict]) -> int:
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


def update_point_features(conn: Connection, rows: list[dict]) -> int:
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

def clusterable_points(conn: Connection, form_id: int) -> list[ClusterablePoint]:
    """Every embedded point of one form, across all of its analysed fields.

    A point does not store its form, so the link runs
    points -> answers -> submissions -> forms. Fields the author left out of
    the analysis never produced points, so no filter on analyze_with_ai is
    needed here.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT p.id, p.point_text, p.embedding, p.sentiment_label, p.is_severe,
                   p.created_at, p.canonical_topic_id
            FROM points p
            JOIN answers a ON a.id = p.answer_id
            JOIN submissions s ON s.id = a.submission_id
            WHERE s.form_id = %s
              AND p.embedding IS NOT NULL AND p.point_text IS NOT NULL
            ORDER BY p.id
            """,
            (form_id,),
        )
        return [ClusterablePoint(*row) for row in cur.fetchall()]


def has_unclustered(conn: Connection, form_id: int) -> bool:
    """Is there a point in this form that no clustering run has seen yet?

    processing_status is what separates one from a point the last run did look
    at and deliberately left out: an outlier is 'unknown' and carries no
    topic, which is a decision, while 'pending' means no run has considered it.
    Asking `canonical_topic_id IS NULL` instead would count every outlier as
    outstanding work and re-cluster on every pass for ever.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT EXISTS (
                SELECT 1 FROM points p
                JOIN answers a ON a.id = p.answer_id
                JOIN submissions s ON s.id = a.submission_id
                WHERE s.form_id = %s
                  AND p.embedding IS NOT NULL AND p.point_text IS NOT NULL
                  AND COALESCE(p.processing_status, 'pending') = 'pending'
            )
            """,
            (form_id,),
        )
        return cur.fetchone()[0]


def existing_topics(conn: Connection, form_id: int) -> dict[int, str]:
    """Live topics of this form, as {id: name}.

    There is no ownership marker any more: `canonical_topics.form_id` scopes a
    run to its own form, so a run can only ever replace topics it is
    responsible for. Topics already retired — absorbed by a merge, or archived
    when nothing claimed them — are left out: a new group can only inherit
    from a topic that is still live.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, canonical_name FROM canonical_topics
            WHERE form_id = %s
              AND COALESCE(status, 'active') NOT IN ('merged', 'archived')
            """,
            (form_id,),
        )
        return {row[0]: row[1] for row in cur.fetchall()}


def archive_topics(conn: Connection, topic_ids: list[int]) -> int:
    """Retire a topic no new group claimed, without destroying it.

    The dashboard only reads status='active', so archiving hides the topic
    exactly as deleting did — but the row survives, and with it the name the
    issue was reported under and the created_at that says when it first
    appeared. A delete threw both away for nothing: the points are reassigned
    in the same transaction either way, so nothing is freed.

    Unlike a merge there is no survivor to point at: merged_into_id stays NULL
    because the members did not go anywhere identifiable.
    """
    if not topic_ids:
        return 0
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE canonical_topics
            SET status = 'archived', updated_at = now()
            WHERE id = ANY(%s)
            """,
            (topic_ids,),
        )
        return cur.rowcount


def insert_topic(
    conn: Connection,
    *,
    form_id: int,
    name: str,
    summary: str | None,
    keywords: str,
    centroid: np.ndarray,
    size: int,
    field_id: int | None = None,
) -> int:
    """field_id stays NULL for a topic the pipeline creates: its members come
    from every analysed question of the form, so naming one of them would be
    wrong. The column is nullable and the server only ever filters on form_id
    (idx_canonical_topics_form_status), so nothing downstream depends on it."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO canonical_topics
                (form_id, field_id, canonical_name, canonical_summary,
                 representative_keywords, centroid_vector, topic_size, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, 'active')
            RETURNING id
            """,
            (form_id, field_id, name, summary, keywords, centroid, size),
        )
        return cur.fetchone()[0]


def update_topic(
    conn: Connection,
    topic_id: int,
    *,
    name: str,
    summary: str | None,
    keywords: str,
    centroid: np.ndarray,
    size: int,
) -> None:
    """Carry an existing topic forward. created_at is left alone — it marks when
    the issue first appeared, and the dashboard's trend depends on the id and
    that date staying put."""
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE canonical_topics
            SET canonical_name = %s,
                canonical_summary = COALESCE(%s, canonical_summary),
                representative_keywords = %s, centroid_vector = %s,
                topic_size = %s, status = 'active', updated_at = now()
            WHERE id = %s
            """,
            (name, summary, keywords, centroid, size, topic_id),
        )


def mark_topics_merged(conn: Connection, absorbed: dict[int, int]) -> int:
    """A topic absorbed by another is retired, not deleted, and now records
    where it went: merged_into_id is the column the schema added for exactly
    this. Deleting it would take its points' history with it."""
    if not absorbed:
        return 0
    with conn.cursor() as cur:
        cur.executemany(
            """
            UPDATE canonical_topics
            SET status = 'merged', merged_into_id = %(into)s, updated_at = now()
            WHERE id = %(id)s
            """,
            [{"id": old, "into": new} for old, new in absorbed.items()],
        )
        return len(absorbed)


def assign_points(conn: Connection, rows: list[dict]) -> int:
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


# topic_trends and unassigned_points were dropped from the schema. Trends are
# derived from points.created_at at read time, and an unclustered point is
# simply one with canonical_topic_id IS NULL.


def record_run(
    conn: Connection,
    *,
    form_id: int,
    parameters: str,
    started_at: datetime,
    completed_at: datetime,
    status: str = "completed",
    error: str | None = None,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO ai_model_runs
                (form_id, model_name, model_version, embedding_model,
                 clustering_algorithm, parameters, status, error,
                 started_at, completed_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                form_id, config.DECOMPOSE_MODEL, config.DECOMPOSE_PROMPT,
                config.EMBEDDING_MODEL, "BERTopic+HDBSCAN", parameters,
                status, error, started_at, completed_at,
            ),
        )
