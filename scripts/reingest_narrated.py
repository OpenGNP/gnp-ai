"""Re-decompose the answers whose points were written as a report about the writer.

BASE_RULES 7-8 tell the model to replace a pronoun with the noun it refers to,
and on feedback written in the first person it read "I" as one of those: "I want
Lockers" was stored as "Request for Lockers", "I've never talked to SIT about
visa stuff" as "User mentions not having talked to SIT about visa-related
matters". pipeline/decompose.py now repairs that voice at ingest time, but the
rows already in the database predate the repair and are not rewritten by it.

This deletes those answers' points and clears decomposed_at so the next ingest
produces them again, with the repair in place.

Every point of an affected answer goes, not only the narrated ones: decomposition
emits the whole set for an answer at once, so keeping the good ones would
duplicate them on the way back in.

    python -m scripts.reingest_narrated              # report only, writes nothing
    python -m scripts.reingest_narrated --apply      # delete and reset

Then, to rebuild what was removed:

    python -m pipeline.run --ingest-only
    python -m pipeline.run --cluster-only
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

# The SQL twin of decompose_ollama.NARRATED_RE, applied to rows the model wrote
# before the repair existed. Postgres spells a word boundary \y, not \b; keep
# the two patterns in step when either changes.
NARRATED_SQL = r"\y(users?|respondents?|the\s+student|the\s+writer)\y|^request for\y"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="actually delete; without it nothing is written")
    args = ap.parse_args()

    conn = db.connect()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT a.id FROM points p
            JOIN answers a ON a.id = p.answer_id
            WHERE p.point_text ~* %s
            """,
            (NARRATED_SQL,),
        )
        answer_ids = [row[0] for row in cur.fetchall()]

    if not answer_ids:
        print("no narrated points left — nothing to do")
        return

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT p.id, p.answer_id, p.point_text, p.sentiment_label, p.is_violation,
                   p.canonical_topic_id, p.processing_status, s.form_id
            FROM points p
            JOIN answers a ON a.id = p.answer_id
            JOIN submissions s ON s.id = a.submission_id
            WHERE p.answer_id = ANY(%s)
            ORDER BY p.id
            """,
            (answer_ids,),
        )
        rows = cur.fetchall()

    print(f"answers to re-decompose : {len(answer_ids)}")
    print(f"points to delete        : {len(rows)}")
    print(f"affected forms          : {dict(Counter(r[7] for r in rows))}")

    if not args.apply:
        print("\nreport only — pass --apply to delete")
        return

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = BACKUP_DIR / f"backup_points_{stamp}.json"
    backup.write_text(json.dumps({
        "answer_ids": answer_ids,
        "points": [
            {"id": r[0], "answer_id": r[1], "point_text": r[2],
             "sentiment_label": r[3], "is_violation": r[4],
             "canonical_topic_id": r[5], "processing_status": r[6], "form_id": r[7]}
            for r in rows
        ],
    }, indent=1, ensure_ascii=False))
    print(f"\nbacked up to {backup}")

    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute("DELETE FROM points WHERE answer_id = ANY(%s)", (answer_ids,))
            deleted = cur.rowcount
            cur.execute(
                "UPDATE answers SET decomposed_at = NULL WHERE id = ANY(%s)",
                (answer_ids,),
            )
            reset = cur.rowcount

    print(f"deleted {deleted} point(s), reset decomposed_at on {reset} answer(s)")
    print("\nnow run:")
    print("    python -m pipeline.run --ingest-only")
    print("    python -m pipeline.run --cluster-only")


if __name__ == "__main__":
    main()
