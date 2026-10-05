"""Read-only inspection of what each stage produced.

Every stage persists its output, so nothing needs re-running to analyse it:

    decompose   answers.answer_text -> the points.point_text rows under it
    sentiment   points.sentiment_label
    violation   points.is_severe
    embedding   points.embedding (reported as dimensions, not dumped)
    clustering  points.canonical_topic_id + assignment_confidence
    priority    computed here from topic_trends, since it is not stored

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

# Latest trend row per topic — the inputs to the priority formula.
TRENDS_SQL = """
    SELECT DISTINCT ON (t.canonical_topic_id)
           t.canonical_topic_id AS topic_id,
           c.canonical_name     AS topic_name,
           c.representative_keywords AS keywords,
           t.feedback_count, t.positive_count, t.neutral_count,
           t.negative_count, t.severe_count
    FROM topic_trends t
    JOIN canonical_topics c ON c.id = t.canonical_topic_id
    JOIN topic_versions v ON v.canonical_topic_id = c.id AND v.model_version = %s
    ORDER BY t.canonical_topic_id, t.period_end DESC
"""


def _frame(conn: psycopg.Connection, sql: str, params=None) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql, params or ())
        columns = [d.name for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=columns)


def priority(trends: pd.DataFrame) -> pd.DataFrame:
    """0.4*reach + 0.4*negative + 0.2*severity — see docs/pipeline.md.

    Reach is relative to the largest topic so one dominant topic cannot flatten
    the rest; the other two are shares of the topic's own volume.
    """
    if trends.empty:
        return trends

    out = trends.copy()
    biggest = out.feedback_count.max() or 1
    volume = out.feedback_count.replace(0, pd.NA)

    out["reach"] = out.feedback_count / biggest
    out["negative_ratio"] = (out.negative_count / volume).fillna(0.0)
    out["severity_ratio"] = (out.severe_count / volume).fillna(0.0)
    out["priority"] = 0.4 * out.reach + 0.4 * out.negative_ratio + 0.2 * out.severity_ratio
    return out.sort_values("priority", ascending=False)


def build(conn: psycopg.Connection, pipeline_tag: str, *, examples: int = 3) -> tuple[str, pd.DataFrame]:
    points = _frame(conn, POINTS_SQL)
    trends = priority(_frame(conn, TRENDS_SQL, (pipeline_tag,)))

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

    # ── clustering + priority ──
    add("")
    add("-" * 78)
    add("TOPICS — ranked by priority (0.4*reach + 0.4*negative + 0.2*severity)")
    add("-" * 78)
    for row in trends.itertuples():
        add("")
        add(f"[{row.priority:.3f}] {row.topic_name}   ({row.feedback_count} points)")
        add(f"  keywords: {row.keywords}")
        add(
            f"  reach {row.reach:.2f} · negative {row.negative_ratio:.2f} "
            f"· severity {row.severity_ratio:.2f}"
            f"  (+{row.positive_count}/~{row.neutral_count}/-{row.negative_count})"
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
