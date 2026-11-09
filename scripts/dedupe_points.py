"""Remove points left behind when two ingests ran over the same answers.

An answer's points are written in one transaction, so a single ingest gives
every point of an answer the same created_at. Two timestamps on one answer
therefore mean two ingests decomposed it — which happens because
`pending_answers()` and `mark_decomposed()` are not atomic across processes:
a second ingest started before the first commits reads the same work queue and
writes the same answers again.

Only the most recent write survives. The earlier one is from a run that was
superseded before it could matter, and keeping it would double every topic's
count for those answers.

    python -m scripts.dedupe_points            # report only, writes nothing
    python -m scripts.dedupe_points --apply
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from pipeline import db

ROOT = Path(__file__).resolve().parent.parent
BACKUP_DIR = ROOT / "data"

# Points of an answer that are not from its newest write.
SUPERSEDED = """
    SELECT p.id, p.answer_id, p.point_text, p.created_at, s.form_id
    FROM points p
    JOIN answers a ON a.id = p.answer_id
    JOIN submissions s ON s.id = a.submission_id
    WHERE p.created_at < (
        SELECT MAX(q.created_at) FROM points q WHERE q.answer_id = p.answer_id
    )
    ORDER BY p.answer_id, p.created_at
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="actually delete; without it nothing is written")
    args = ap.parse_args()

    conn = db.connect()
    with conn.cursor() as cur:
        cur.execute(SUPERSEDED)
        rows = cur.fetchall()

    if not rows:
        print("no superseded points — nothing to do")
        return

    print(f"superseded points : {len(rows)}")
    print(f"answers affected  : {len({r[1] for r in rows})}")
    print(f"by form           : {dict(Counter(r[4] for r in rows))}")

    if not args.apply:
        print("\nreport only — pass --apply to delete")
        return

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = BACKUP_DIR / f"backup_superseded_{stamp}.json"
    backup.write_text(json.dumps([
        {"id": r[0], "answer_id": r[1], "point_text": r[2],
         "created_at": r[3].isoformat(), "form_id": r[4]}
        for r in rows
    ], indent=1, ensure_ascii=False))
    print(f"\nbacked up to {backup}")

    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM points WHERE id = ANY(%s)", ([r[0] for r in rows],)
            )
            deleted = cur.rowcount
    print(f"deleted {deleted} superseded point(s)")


if __name__ == "__main__":
    main()
