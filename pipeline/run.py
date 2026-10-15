"""Batch job: turn form answers into points, topics and trends.

Two halves, as the schema implies:

  ingest   per answer   decompose -> embed -> sentiment/violation   writes points
  cluster  per form     HDBSCAN/BERTopic                   writes canonical_topics,
                                                           ai_model_runs

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
import psycopg

from . import cluster as cluster_mod
from . import config, db, decompose, embed, identity, llm, naming, report, sentiment, violation


def _log(message: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


# ── stage: ingest ──────────────────────────────────────────────────────

# Answers decomposed before each write. Decomposition is one local-LLM call
# per answer and touches no database, so a whole field's worth of them leaves
# the connection idle for as long as it takes — twenty minutes for 212 answers,
# which is long enough for the network to drop it and lose every answer
# processed so far. Flushing every CHUNK answers keeps the idle gap to about a
# minute and caps what a failure costs; pending_answers() is the work queue, so
# a rerun picks up exactly what did not land.
CHUNK = 20


def ingest(conn, *, limit: int | None, dry_run: bool, skip_sentiment: bool) -> int:
    """Decompose every answer the form asked to have analysed.

    Scope comes from form_fields.analyze_with_ai, so a form's demographic
    questions and anything the author excluded are never read.
    """
    fields = db.analysable_fields(conn)
    if not fields:
        _log("ingest: no field is marked analyze_with_ai")
        return 0

    dims = db.embedding_dims(conn)
    written = 0
    for form_id, field_id, label in fields:
        pending = db.pending_answers(conn, field_id, limit)
        if not pending:
            continue

        _log(f"ingest: form {form_id} · {label} — {len(pending)} unprocessed answer(s)")
        if dry_run:
            continue

        done = 0
        field_points = 0
        for start in range(0, len(pending), CHUNK):
            batch = pending[start : start + CHUNK]

            # The classifiers are fast CPU models that prefer a batch, so they
            # run once per chunk rather than once per answer.
            staged: list[tuple[int, str]] = []
            with llm.client() as http:
                for answer_id, text in batch:
                    points, status = decompose.decompose(http, text)
                    staged.extend((answer_id, point) for point in points)
                    done += 1
                    _log(f"  [{done}/{len(pending)}] answer {answer_id} "
                         f"-> {len(points)} point(s) [{status}]")

            texts = [text for _, text in staged]
            vectors = embed.encode(texts, dims)
            labels = [None] * len(texts) if skip_sentiment else sentiment.classify(texts)
            violations = [None] * len(texts) if skip_sentiment else violation.flag_zeroshot(texts)

            rows = [
                {
                    "answer_id": answer_id,
                    "point_text": text,
                    "embedding": vector,
                    "sentiment_label": sentiment_label,
                    "is_violation": violation,
                    "processing_status": "pending",
                }
                for (answer_id, text), vector, sentiment_label, violation
                in zip(staged, vectors, labels, violations)
            ]

            # db.Connection re-dials between transactions, but one that dies
            # mid-write has to fail — so retry the chunk once on a fresh
            # connection rather than lose the decomposition behind it.
            # Every answer in the chunk is marked, including any the model
            # found no point in, so the next run does not read it again.
            answer_ids = [answer_id for answer_id, _ in batch]
            try:
                with conn.transaction():
                    db.insert_points(conn, rows)
                    db.mark_decomposed(conn, answer_ids)
            except psycopg.OperationalError as exc:
                _log(f"  write failed ({exc.__class__.__name__}), retrying on a new connection")
                with conn.transaction():
                    db.insert_points(conn, rows)
                    db.mark_decomposed(conn, answer_ids)

            written += len(rows)
            field_points += len(rows)

        _log(f"ingest: form {form_id} · {label} — wrote {field_points} point(s)")

    if dry_run:
        _log("ingest: dry run, nothing written")
    elif not written:
        _log("ingest: no unprocessed answers")
    return written


def backfill(conn, *, dry_run: bool, skip_sentiment: bool, form_ids: list[int] | None = None) -> int:
    """Give pre-existing points (the seeded ones) an embedding so they join the
    corpus. Their topic assignment will then be replaced by pipeline topics."""
    missing = db.points_missing_features(conn, form_ids)
    if not missing:
        _log("backfill: every point already has an embedding")
        return 0

    _log(f"backfill: {len(missing)} point(s) without an embedding")
    if dry_run:
        _log("backfill: dry run, nothing written")
        return 0

    texts = [text for _, text in missing]
    vectors = embed.encode(texts, db.embedding_dims(conn))

    labels = [None] * len(texts) if skip_sentiment else sentiment.classify(texts)
    violations = [None] * len(texts) if skip_sentiment else violation.flag_zeroshot(texts)

    rows = [
        {
            "id": point_id,
            "embedding": vector,
            "sentiment_label": label,
            "is_violation": violation,
            "processing_status": "pending",
        }
        for (point_id, _), vector, label, violation in zip(missing, vectors, labels, violations)
    ]

    with conn.transaction():
        db.update_point_features(conn, rows)
    _log(f"backfill: updated {len(rows)} point(s)")
    return len(rows)


def relabel(conn, *, dry_run: bool, form_ids: list[int] | None = None) -> int:
    """Re-run the classifiers over points the pipeline owns. Needed whenever
    SENTIMENT_MODEL, VIOLATION_MODEL or VIOLATION_THRESHOLD changes."""
    targets = db.points_for_relabel(conn, form_ids)
    if not targets:
        _log("relabel: no pipeline-owned points")
        return 0

    _log(f"relabel: {len(targets)} point(s)")
    if dry_run:
        _log("relabel: dry run, nothing written")
        return 0

    texts = [text for _, text in targets]
    labels = sentiment.classify(texts)
    violations = violation.flag_zeroshot(texts)

    rows = [
        {"id": point_id, "sentiment_label": label, "is_violation": violation}
        for (point_id, _), label, violation in zip(targets, labels, violations)
    ]
    with conn.transaction():
        db.update_point_labels(conn, rows)

    _log(f"relabel: updated {len(rows)} point(s), {sum(1 for r in rows if r['is_violation'])} flagged")
    return len(rows)


# ── stage: cluster ─────────────────────────────────────────────────────

def recluster(conn, *, dry_run: bool, name_topics: bool = True, force: bool = False) -> int:
    """Cluster each form on its own.

    One form is one body of feedback: a topic spans every question the author
    marked for analysis, because the same complaint can arrive through any of
    them. canonical_topics.form_id makes that scope explicit, and the
    trade-off is deliberate — the same issue raised in two different surveys
    becomes two topics and cannot be compared across them.

    A form with nothing new is left alone. Clustering has to look at the whole
    corpus at once, so unlike ingest it cannot do only the new part — but
    running it over unchanged points is minutes of work to arrive back where
    it started, which a nightly cron would repeat for ever. `force` overrides
    that, for when the parameters changed rather than the data.
    """
    forms = db.analysable_forms(conn)
    if not forms:
        _log("cluster: no field is marked analyze_with_ai")
        return 0

    total_topics = 0
    for form_id, form_title in forms:
        if not force and not db.has_unclustered(conn, form_id):
            _log(f"cluster: form {form_id} · {form_title} — no new points, skipped")
            continue

        started = datetime.now(timezone.utc)
        points = db.clusterable_points(conn, form_id)
        if len(points) < cluster_mod.MIN_POINTS:
            # Too few to cluster, but they have been considered, and leaving
            # them 'pending' makes the dashboard report the form as never
            # analysed and show nothing at all. Marked 'unknown' they appear
            # as uncategorised, which is what they are. They are not lost to
            # clustering: clusterable_points reads every embedded point
            # whatever its status, so once enough arrive these join the run.
            if not dry_run and points:
                with conn.transaction():
                    db.assign_points(
                        conn,
                        [
                            {"id": p.id, "topic_id": None, "confidence": None,
                             "processing_status": "unknown"}
                            for p in points
                        ],
                    )
            _log(f"cluster: form {form_id} · {form_title} — {len(points)} point(s), "
                 "too few to cluster, left uncategorised")
            continue

        _log(f"cluster: form {form_id} · {form_title} — {len(points)} embedded point(s)")
        result = cluster_mod.cluster(points)
        _log(
            f"  {len(result.topics)} topic(s), {len(result.outliers)} outlier(s) "
            f"(mcs={result.min_cluster_size})"
        )

        # Which of these groups are topics we already know? Resolved from
        # membership before anything is written, because the first write would
        # overwrite the very assignments the answer depends on.
        known = db.existing_topics(conn, form_id)
        plan = identity.resolve(points, result, known)
        counts = plan.summary()
        _log(
            f"  identity: carried {counts['carried']} · new {counts['new']} · "
            f"split {counts['split']} · merged {counts['merged']} · "
            f"dissolved {counts['dissolved']}"
        )
        for event in plan.events[:8]:
            _log(f"    {event}")

        # A carried topic keeps the title the dashboard already shows: the
        # same issue with more people reporting it is still the same issue,
        # and renaming it every run would make the trend unreadable.
        #
        # Absorbing another topic is the exception. The scope genuinely
        # widened, so the old title stops covering what is in the group —
        # "Inadequate Monitor Equipment" kept its name after swallowing "LX
        # Building Facilities" and growing from 16 points to 144, by which
        # point it was about room equipment generally. The merges already
        # recorded say which topics those are, so no new threshold is needed.
        absorbed = plan.widened()
        titles: dict[str, tuple[str | None, str | None]] = {}
        unnamed = [t for t in result.topics if t.key not in plan.reuse or t.key in absorbed]
        if name_topics and not dry_run and unnamed:
            renamed = sum(1 for t in unnamed if t.key in absorbed)
            _log(f"  naming {len(unnamed) - renamed} new topic(s)"
                 + (f", renaming {renamed} that absorbed another" if renamed else ""))
            with llm.client() as http:
                for topic in unnamed:
                    title, summary = naming.name_topic(http, topic.keywords, topic.member_texts)
                    titles[topic.key] = (title, summary)

        if dry_run:
            total_topics += len(result.topics)
            continue

        with conn.transaction():
            assignments: list[dict] = []
            for topic in result.topics:
                carried = plan.reuse.get(topic.key)
                if carried is not None:
                    topic_id = carried
                    # titles only holds an entry for a carried topic that
                    # absorbed another; everything else keeps its old name.
                    title, summary = titles.get(topic.key, (None, None))
                    db.update_topic(
                        conn,
                        topic_id,
                        name=title or known[topic_id],
                        summary=summary,
                        keywords=topic.keywords,
                        centroid=topic.centroid,
                        size=topic.counts["total"],
                    )
                else:
                    title, summary = titles.get(topic.key, (None, None))
                    topic_id = db.insert_topic(
                        conn,
                        form_id=form_id,
                        name=title or topic.name,
                        summary=summary,
                        keywords=topic.keywords,
                        centroid=topic.centroid,
                        size=topic.counts["total"],
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
            # An outlier keeps no topic and is marked 'unknown': the pipeline
            # looked at it and it fitted nowhere, which is a different thing
            # from 'pending', where no run has considered it yet. Grouping
            # these for the dashboard is the server's to do — they are found
            # by status, not by the NULL topic a pending point also has.
            db.assign_points(
                conn,
                [
                    {"id": p.id, "topic_id": None, "confidence": None,
                     "processing_status": "unknown"}
                    for p in result.outliers
                ],
            )

            # Once every point has been placed, a topic holding none is over:
            # absorbed by another group, or dissolved. Dropped here rather
            # than from plan.merged / plan.dissolved, so a row left behind by
            # an earlier run goes with it.
            dropped = db.delete_empty_topics(conn, form_id)

            db.record_run(
                conn,
                form_id=form_id,
                parameters=json.dumps({
                    "identity": counts,
                    "carry_min": config.TOPIC_CARRY_MIN,
                    "min_cluster_size": result.min_cluster_size,
                    "outlier_threshold": config.OUTLIER_THRESHOLD,
                    "n_points": len(points),
                    "n_topics": len(result.topics),
                }),
                started_at=started,
                completed_at=datetime.now(timezone.utc),
            )

        _log(
            f"  carried {counts['carried']}, added {counts['new']}, "
            f"merged {counts['merged']}, dropped {dropped}"
        )
        total_topics += len(result.topics)

    if dry_run:
        _log(f"cluster: dry run — {total_topics} topic(s) across {len(forms)} form(s), nothing written")
    return total_topics


# ── CLI ────────────────────────────────────────────────────────────────

def pass_once(conn, args) -> None:
    if args.report:
        text, points = report.build(conn)
        print(text)
        report_path, csv_path = report.save(text, points)
        _log(f"report: {report_path}")
        _log(f"csv:    {csv_path}")
        return

    if args.relabel:
        relabel(conn, dry_run=args.dry_run, form_ids=args.form)

    if args.backfill:
        backfill(conn, dry_run=args.dry_run, skip_sentiment=args.skip_sentiment, form_ids=args.form)

    new_points = 0
    if not args.cluster_only:
        new_points = ingest(
            conn, limit=args.limit, dry_run=args.dry_run, skip_sentiment=args.skip_sentiment
        )

    if args.ingest_only:
        return

    # --cluster-only is someone asking for a re-cluster, usually because a
    # parameter changed; otherwise recluster() decides per form, from what is
    # in the database rather than from what this pass happened to ingest — a
    # run that ingested and then died still has work waiting.
    recluster(
        conn,
        dry_run=args.dry_run,
        name_topics=not args.no_name_topics,
        force=args.cluster_only,
    )


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
    ap.add_argument("--no-name-topics", action="store_true",
                    help="skip LLM topic naming; keep the raw c-TF-IDF keyword names")
    ap.add_argument("--report", action="store_true",
                    help="print what every stage produced, save it to data/reports/ "
                         "and dump data/pipeline_points.csv; writes nothing to the database")
    ap.add_argument("--relabel", action="store_true",
                    help="re-run the sentiment and violation classifiers over existing "
                         "pipeline-owned points (after changing either model)")
    ap.add_argument("--form", type=int, nargs="+", metavar="ID", default=None,
                    help="restrict --backfill and --relabel to these form ids "
                         "(clustering always spans every embedded point)")
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
