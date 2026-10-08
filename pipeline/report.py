"""Read-only inspection of what each stage produced.

Every stage persists its output, so nothing needs re-running to analyse it:

    decompose   answers.answer_text -> the points.point_text rows under it
    sentiment   points.sentiment_label
    violation   points.is_severe
    embedding   points.embedding (reported as dimensions, not dumped)
    clustering  points.canonical_topic_id + assignment_confidence

Writes a readable report to data/reports/ (same convention as the experiment
scripts) and a flat CSV for spreadsheet or pandas analysis.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
import psycopg

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REPORTS = DATA / "reports"

# One row per point, with the answer it came from and the topic it landed in.
POINTS_SQL = """
    SELECT p.id            AS point_id,
           p.answer_id,
           a.answer_text,
           p.point_text,
           p.sentiment_label,
           p.is_severe,
           p.canonical_topic_id AS topic_id,
           c.canonical_name     AS topic_name,
           p.assignment_confidence,
           p.processing_status,
           vector_dims(p.embedding) AS embedding_dims,
           p.created_at
    FROM points p
    JOIN answers a ON a.id = p.answer_id
    LEFT JOIN canonical_topics c ON c.id = p.canonical_topic_id
    WHERE p.embedding IS NOT NULL
    ORDER BY p.answer_id, p.id
"""

# Counted from the points themselves. topic_trends used to cache this per
# period and no longer exists — the counts were always derivable, since every
# point carries its topic, its sentiment and the date it was submitted.
TOPICS_SQL = """
    SELECT c.id   AS topic_id,
           c.form_id,
           c.field_id,
           c.canonical_name          AS topic_name,
           c.representative_keywords AS keywords,
           count(p.id)                                                   AS feedback_count,
           count(*) FILTER (WHERE p.sentiment_label = 'positive')        AS positive_count,
           count(*) FILTER (WHERE p.sentiment_label = 'neutral')         AS neutral_count,
           count(*) FILTER (WHERE p.sentiment_label = 'negative')        AS negative_count,
           count(*) FILTER (WHERE p.is_severe)                           AS severe_count
    FROM canonical_topics c
    LEFT JOIN points p ON p.canonical_topic_id = c.id
    WHERE COALESCE(c.status, 'active') NOT IN ('merged', 'archived')
    GROUP BY c.id, c.form_id, c.field_id, c.canonical_name, c.representative_keywords
"""


def _frame(conn: psycopg.Connection, sql: str, params=None) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql, params or ())
        columns = [d.name for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=columns)


def rank(trends: pd.DataFrame) -> pd.DataFrame:
    """Biggest topics first. Ranking for the dashboard is the frontend's job."""
    return trends if trends.empty else trends.sort_values("feedback_count", ascending=False)


def build(conn: psycopg.Connection, *, examples: int = 3) -> tuple[str, pd.DataFrame]:
    points = _frame(conn, POINTS_SQL)
    trends = rank(_frame(conn, TOPICS_SQL))

    lines: list[str] = []
    add = lines.append

    add("=" * 78)
    add(f"pipeline report  {datetime.now():%Y-%m-%d %H:%M:%S}")
    add("=" * 78)

    if points.empty:
        add("\nno pipeline-owned points yet (nothing has an embedding)")
        return "\n".join(lines), points

    per_answer = points.groupby("answer_id").size()
    add("")
    add(f"answers processed     {len(per_answer)}")
    add(f"points produced       {len(points)}")
    add(f"points per answer     {per_answer.mean():.2f} mean, {per_answer.max()} max")
    add(f"embedding dimensions  {sorted(points.embedding_dims.unique())}")
    add(f"topics                {points.topic_id.nunique(dropna=True)}")
    add(f"unclustered points    {int(points.topic_id.isna().sum())}")

    sentiment = Counter(points.sentiment_label.fillna("NULL"))
    add(f"sentiment             {dict(sentiment)}")
    add(f"violation flagged     {int(points.is_severe.fillna(False).sum())}")

    # ── decompose ──
    add("")
    add("-" * 78)
    add("DECOMPOSE — one answer, the points it was split into")
    add("-" * 78)
    for answer_id, group in points.groupby("answer_id"):
        add("")
        add(f"answer {answer_id}  ->  {len(group)} point(s)")
        add(f"  raw: {group.answer_text.iloc[0]}")
        for row in group.itertuples():
            flag = " [FLAGGED]" if row.is_severe else ""
            add(f"   - [{row.sentiment_label or 'NULL':8s}]{flag} {row.point_text}")

    # ── clustering ──
    add("")
    add("-" * 78)
    add("TOPICS — largest first")
    add("-" * 78)
    for row in trends.itertuples():
        add("")
        add(f"{row.topic_name}   ({row.feedback_count} points)")
        add(f"  keywords: {row.keywords}")
        add(
            f"  +{row.positive_count} / ~{row.neutral_count} / -{row.negative_count}"
            f"   severe {row.severe_count}"
        )
        members = points[points.topic_id == row.topic_id].nlargest(
            examples, "assignment_confidence"
        )
        for member in members.itertuples():
            add(f"   {member.assignment_confidence:.2f}  {member.point_text[:88]}")

    orphans = points[points.topic_id.isna()]
    if not orphans.empty:
        add("")
        add("-" * 78)
        add(f"OUTLIERS — {len(orphans)} point(s) no topic claimed")
        add("-" * 78)
        for row in orphans.itertuples():
            add(f"  {row.point_text}")

    return "\n".join(lines), points


def save(text: str, points: pd.DataFrame) -> tuple[Path, Path]:
    REPORTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    report_path = REPORTS / f"pipeline_n{len(points)}_{stamp}.txt"
    report_path.write_text(text)

    csv_path = DATA / "pipeline_points.csv"
    points.drop(columns=["answer_text"]).to_csv(csv_path, index=False)
    return report_path, csv_path
