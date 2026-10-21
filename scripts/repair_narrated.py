"""Rewrite stored points that are told about the writer instead of by them.

The same repair pipeline/decompose.py applies at ingest, applied to rows that
predate it — but in place, not by re-ingesting.

Re-ingesting is the obvious route and it does not work here. Deleting an
answer's points and clearing decomposed_at puts that answer back on the work
queue, and this database has more than one pipeline reading that queue: a
second instance on another host has been seen taking the freed answers within
twenty seconds and writing its own, un-repaired decomposition. Every pass of
delete-and-reingest therefore raced, and lost about as often as it won.

Editing point_text leaves decomposed_at alone, so the answer never returns to
the queue and there is no window to race over. It also preserves the point ids,
which is what keeps the existing canonical_topic_id assignments intact.

The embedding is recomputed: it was built from the old wording, and leaving it
would cluster the point by text no longer in the row.

    python -m scripts.repair_narrated              # report only
    python -m scripts.repair_narrated --apply
    python -m scripts.repair_narrated --apply --limit 5
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from pipeline import db, embed, llm
from pipeline.decompose import _repair_voice
from scripts.decompose_ollama import narrated
from scripts.reingest_narrated import NARRATED_SQL

ROOT = Path(__file__).resolve().parent.parent
BACKUP_DIR = ROOT / "data"


def load(conn) -> dict[int, tuple[str, list[tuple[int, str]]]]:
    """{answer_id: (answer_text, [(point_id, point_text), ...])} for every
    answer holding at least one narrated point.

    All of an answer's points come along, not only the narrated ones: the
    repair rewrites the set as a whole, and asking it to fix two points out of
    four while hiding the rest would strand them in different voices.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT p.answer_id, a.answer_text, p.id, p.point_text
            FROM points p
            JOIN answers a ON a.id = p.answer_id
            WHERE p.answer_id IN (
                SELECT answer_id FROM points WHERE point_text ~* %s
            )
              AND a.answer_text IS NOT NULL
            ORDER BY p.answer_id, p.id
            """,
            (NARRATED_SQL,),
        )
        rows = cur.fetchall()

    grouped: dict[int, tuple[str, list[tuple[int, str]]]] = {}
    by_answer = defaultdict(list)
    answer_text = {}
    for answer_id, text, point_id, point_text in rows:
        answer_text[answer_id] = text
        by_answer[answer_id].append((point_id, point_text))
    for answer_id, points in by_answer.items():
        grouped[answer_id] = (answer_text[answer_id], points)
    return grouped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--limit", type=int, default=None,
                    help="repair at most N answers this run")
    args = ap.parse_args()

    conn = db.connect()
    work = load(conn)
    if not work:
        print("no narrated points — nothing to do")
        return

    total_points = sum(len(p) for _, p in work.values())
    narrated_now = sum(
        len(narrated([t for _, t in points])) for _, points in work.values()
    )
    print(f"answers with narrated points : {len(work)}")
    print(f"points in those answers      : {total_points}")
    print(f"  of which narrated          : {narrated_now}")

    if not args.apply:
        print("\nreport only — pass --apply to rewrite")
        return

    items = list(work.items())
    if args.limit:
        items = items[: args.limit]

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = BACKUP_DIR / f"backup_narrated_text_{stamp}.json"
    backup.write_text(json.dumps([
        {"answer_id": aid, "answer_text": text,
         "points": [{"id": pid, "point_text": ptext} for pid, ptext in points]}
        for aid, (text, points) in items
    ], indent=1, ensure_ascii=False))
    print(f"\nbacked up to {backup}")

    dims = db.embedding_dims(conn)
    changed = skipped = 0
    statuses: dict[str, int] = {}

    with llm.client() as http:
        for answer_id, (text, points) in items:
            ids = [pid for pid, _ in points]
            before = [ptext for _, ptext in points]

            repaired, status = _repair_voice(http, text, before, "ok")
            statuses[status] = statuses.get(status, 0) + 1

            if repaired == before:
                skipped += 1
                print(f"  answer {answer_id}: unchanged [{status}]")
                continue

            vectors = embed.encode(repaired, dims)
            rows = [
                {"id": pid, "text": new, "embedding": vec}
                for pid, new, vec in zip(ids, repaired, vectors)
            ]
            with conn.transaction():
                with conn.cursor() as cur:
                    # processing_status is left alone deliberately: the point
                    # keeps whatever topic it was assigned, and the next
                    # clustering run picks the new wording up from the embedding.
                    cur.executemany(
                        "UPDATE points SET point_text = %(text)s, embedding = %(embedding)s "
                        "WHERE id = %(id)s",
                        rows,
                    )
            changed += 1
            left = len(narrated(repaired))
            print(f"  answer {answer_id}: rewrote {len(rows)} point(s) "
                  f"[{status}]" + (f" — {left} still narrated" if left else ""))

    print(f"\nrewrote {changed} answer(s), left {skipped} unchanged")
    print(f"status: {statuses}")
    print("\nrun `python -m pipeline.run --cluster-only` to re-cluster on the new wording")


if __name__ == "__main__":
    main()
