"""Batch job: turn form answers into points, topics and trends.

Two halves, as the schema implies:

  ingest   per answer   decompose -> embed -> sentiment/violation   writes points
  cluster  whole corpus HDBSCAN/BERTopic                  writes canonical_topics,
                                                          topic_trends, topic_versions,
                                                          unassigned_points, ai_model_runs

The server and client are never touched; they read these tables already.

    python -m pipeline.run --once            # both halves, one pass
    python -m pipeline.run --ingest-only     # new answers only
    python -m pipeline.run --cluster-only    # re-cluster what is already in points
    python -m pipeline.run --watch 60        # poll every 60s (demo stand-in for cron)
    python -m pipeline.run --once --dry-run  # report what would happen, write nothing
    python -m pipeline.run --report          # what every stage produced, read-only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone

import numpy as np

from . import cluster as cluster_mod
from . import config, db, decompose, embed, llm, report, sentiment, violation


def _log(message: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


# ── stage: ingest ──────────────────────────────────────────────────────

def ingest(conn, *, limit: int | None, dry_run: bool, skip_sentiment: bool) -> int:
    pending = db.pending_answers(conn, limit)
    if not pending:
        _log("ingest: no unprocessed answers")
        return 0

    _log(f"ingest: {len(pending)} unprocessed answer(s)")
    if dry_run:
        _log("ingest: dry run, nothing written")
        return 0

    # Decomposition is one slow local-LLM call per answer; the two classifiers
    # are fast CPU models, so they run once over the whole batch at the end.
    staged: list[tuple[int, str]] = []
    with llm.client() as http:
        for n, (answer_id, text) in enumerate(pending, 1):
            points, status = decompose.decompose(http, text)
            staged.extend((answer_id, point) for point in points)
            _log(f"  [{n}/{len(pending)}] answer {answer_id} -> {len(points)} point(s) [{status}]")

    texts = [text for _, text in staged]
    vectors = embed.encode(texts)
    labels = [None] * len(texts) if skip_sentiment else sentiment.classify(texts)
    severities = [None] * len(texts) if skip_sentiment else violation.flag(texts)

    rows = [
        {
            "answer_id": answer_id,
            "point_text": text,
            "embedding": vector,
            "sentiment_label": label,
            "is_severe": severe,
            "processing_status": "pending",
        }
        for (answer_id, text), vector, label, severe in zip(staged, vectors, labels, severities)
    ]

    with conn.transaction():
        db.insert_points(conn, rows)
    _log(f"ingest: wrote {len(rows)} point(s)")
    return len(rows)


def backfill(conn, *, dry_run: bool, skip_sentiment: bool) -> int:
    """Give pre-existing points (the seeded ones) an embedding so they join the
    corpus. Their topic assignment will then be replaced by pipeline topics."""
    missing = db.points_missing_features(conn)
    if not missing:
        _log("backfill: every point already has an embedding")
        return 0

    _log(f"backfill: {len(missing)} point(s) without an embedding")
    if dry_run:
        _log("backfill: dry run, nothing written")
        return 0

    texts = [text for _, text in missing]
    vectors = embed.encode(texts)

    labels = [None] * len(texts) if skip_sentiment else sentiment.classify(texts)
    severities = [None] * len(texts) if skip_sentiment else violation.flag(texts)

    rows = [
        {
            "id": point_id,
            "embedding": vector,
            "sentiment_label": label,
            "is_severe": severe,
            "processing_status": "pending",
        }
        for (point_id, _), vector, label, severe in zip(missing, vectors, labels, severities)
    ]

    with conn.transaction():
        db.update_point_features(conn, rows)
    _log(f"backfill: updated {len(rows)} point(s)")
    return len(rows)


def relabel(conn, *, dry_run: bool) -> int:
    """Re-run the classifiers over points the pipeline owns. Needed whenever
    SENTIMENT_MODEL, VIOLATION_MODEL or VIOLATION_THRESHOLD changes."""
    targets = db.points_for_relabel(conn)
    if not targets:
        _log("relabel: no pipeline-owned points")
        return 0

    _log(f"relabel: {len(targets)} point(s)")
    if dry_run:
        _log("relabel: dry run, nothing written")
        return 0

    texts = [text for _, text in targets]
    labels = sentiment.classify(texts)
    severities = violation.flag(texts)

    rows = [
        {"id": point_id, "sentiment_label": label, "is_severe": severe}
        for (point_id, _), label, severe in zip(targets, labels, severities)
    ]
    with conn.transaction():
        db.update_point_labels(conn, rows)

    _log(f"relabel: updated {len(rows)} point(s), {sum(1 for r in rows if r['is_severe'])} flagged")
    return len(rows)


# ── stage: cluster ─────────────────────────────────────────────────────

def recluster(conn, *, dry_run: bool) -> int:
    started = datetime.now(timezone.utc)
    points = db.clusterable_points(conn)
    _log(f"cluster: {len(points)} embedded point(s) in scope")

    result = cluster_mod.cluster(points)
    _log(
        f"cluster: {len(result.topics)} topic(s), {len(result.outliers)} outlier(s) "
        f"(mcs={result.min_cluster_size}, split-large={result.split_large}, "
        f"outlier-threshold={config.OUTLIER_THRESHOLD})"
    )
    for topic in result.topics:
        _log(f"  {topic.name}  ({topic.counts['total']} points)")

    stale = db.pipeline_topic_ids(conn)
    if dry_run:
        _log(f"cluster: dry run — would replace {len(stale)} pipeline topic(s), nothing written")
        return 0

    with conn.transaction():
        # Only topics a previous pipeline run created. Seeded topics carry no
        # PIPELINE_TAG marker and survive untouched.
        removed = db.delete_topics(conn, stale)

        assignments: list[dict] = []
        for topic in result.topics:
            topic_id = db.insert_topic(
                conn,
                name=topic.name,
                summary=None,  # auto-generated summaries are still an open decision
                keywords=topic.keywords,
                centroid=topic.centroid,
                size=topic.counts["total"],
                first_seen=topic.first_seen,
                last_seen=topic.last_seen,
            )
            db.insert_trend(
                conn,
                topic_id=topic_id,
                period_start=topic.first_seen,
                period_end=topic.last_seen,
                counts=topic.counts,
            )
            assignments.extend(
                {
                    "id": point_id,
                    "topic_id": topic_id,
                    "confidence": confidence,
                    "processing_status": "assigned",
                }
                for point_id, confidence in topic.members
            )

        db.assign_points(conn, assignments)

        outlier_ids = [p.id for p in result.outliers]
        db.assign_points(
            conn,
            [
                {"id": pid, "topic_id": None, "confidence": None, "processing_status": "unknown"}
                for pid in outlier_ids
            ],
        )
        db.clear_unassigned(conn, outlier_ids)
        db.insert_unassigned(
            conn,
            [
                {
                    "point_id": p.id,
                    "embedding": db.as_array(p.embedding),
                    "reason": f"below_threshold_{config.OUTLIER_THRESHOLD}",
                }
                for p in result.outliers
            ],
        )

        db.record_run(
            conn,
            parameters=json.dumps({
                "min_cluster_size": result.min_cluster_size,
                "split_large": result.split_large,
                "outlier_threshold": config.OUTLIER_THRESHOLD,
                "n_points": len(points),
                "n_topics": len(result.topics),
            }),
            started_at=started,
            completed_at=datetime.now(timezone.utc),
        )

    _log(f"cluster: replaced {removed} old topic(s), wrote {len(result.topics)} new")
    return len(result.topics)


# ── CLI ────────────────────────────────────────────────────────────────

def pass_once(conn, args) -> None:
    if args.report:
        text, points = report.build(conn, config.PIPELINE_TAG)
        print(text)
        report_path, csv_path = report.save(text, points)
        _log(f"report: {report_path}")
        _log(f"csv:    {csv_path}")
        return

    if args.relabel:
        relabel(conn, dry_run=args.dry_run)

    if args.backfill:
        backfill(conn, dry_run=args.dry_run, skip_sentiment=args.skip_sentiment)

    new_points = 0
    if not args.cluster_only:
        new_points = ingest(
            conn, limit=args.limit, dry_run=args.dry_run, skip_sentiment=args.skip_sentiment
        )

    if args.ingest_only:
        return
    if args.watch and not new_points and not args.backfill:
        _log("cluster: skipped, no new points this pass")
        return

    recluster(conn, dry_run=args.dry_run)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--once", action="store_true", help="single pass (default)")
    ap.add_argument("--watch", type=int, metavar="SECONDS",
                    help="poll forever, every SECONDS; cron replaces this in production")
    ap.add_argument("--ingest-only", action="store_true", help="decompose/embed/sentiment only")
    ap.add_argument("--cluster-only", action="store_true", help="re-cluster existing points only")
    ap.add_argument("--backfill", action="store_true",
                    help="embed points that have none (the seeded rows) so they join the corpus; "
                         "their topic assignment is then replaced by pipeline topics")
    ap.add_argument("--report", action="store_true",
                    help="print what every stage produced, save it to data/reports/ "
                         "and dump data/pipeline_points.csv; writes nothing to the database")
    ap.add_argument("--relabel", action="store_true",
                    help="re-run the sentiment and violation classifiers over existing "
                         "pipeline-owned points (after changing either model)")
    ap.add_argument("--limit", type=int, metavar="N", help="ingest at most N answers this pass")
    ap.add_argument("--skip-sentiment", action="store_true",
                    help="skip the sentiment and violation classifiers, leaving those columns NULL")
    ap.add_argument("--dry-run", action="store_true", help="report counts, write nothing")
    args = ap.parse_args()

    if args.ingest_only and args.cluster_only:
        ap.error("--ingest-only and --cluster-only are mutually exclusive")

    conn = db.connect()
    try:
        if args.watch:
            _log(f"watching every {args.watch}s — ctrl-c to stop")
            while True:
                pass_once(conn, args)
                time.sleep(args.watch)
        else:
            pass_once(conn, args)
    except KeyboardInterrupt:
        _log("stopped")
        sys.exit(0)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
