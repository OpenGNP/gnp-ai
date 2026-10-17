"""Score every distinct point with the zero-shot detector, for sampling.

Not a measurement of the detector — its job here is to spread the annotation
budget. Violations are about 3% of the corpus, so a random thousand would
carry thirty of them and teach a classifier almost nothing. Scoring first lets
the sample take most of its items from where the model is unsure, which is
where a human judgement is worth the most.

The scores are never shown to the annotator: seeing them would turn the task
into agreeing with the model rather than reading the text.

    python scripts/score_pool.py                    # whole corpus
    python scripts/score_pool.py --limit 50         # smoke test
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import config, db, violation  # noqa: E402

OUT = ROOT / "data" / "labels" / "scored_pool.csv"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, help="score only this many (smoke test)")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    conn = db.connect()
    with conn.cursor() as cur:
        # One row per distinct text: the synthetic corpus repeats sentences, and
        # labelling the same string twice buys nothing.
        cur.execute(
            """
            SELECT min(p.id), p.point_text, min(s.form_id), count(*)
            FROM points p
            JOIN answers a ON a.id = p.answer_id
            JOIN submissions s ON s.id = a.submission_id
            WHERE p.point_text IS NOT NULL
            GROUP BY p.point_text
            ORDER BY min(p.id)
            """
        )
        rows = cur.fetchall()
    if args.limit:
        rows = rows[: args.limit]

    print(f"{len(rows)} distinct point(s) to score with {config.VIOLATION_ZEROSHOT_MODEL}")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["point_id", "form_id", "copies", "score", "category", "text"]
        )
        writer.writeheader()
        for n, (point_id, text, form_id, copies) in enumerate(rows, 1):
            _, category, score = violation.classify_zeroshot(text)
            writer.writerow({
                "point_id": point_id, "form_id": form_id, "copies": copies,
                "score": score, "category": category, "text": text,
            })
            if n % 25 == 0 or n == len(rows):
                rate = (time.perf_counter() - started) / n
                left = rate * (len(rows) - n)
                print(f"  [{n}/{len(rows)}] {rate:.2f}s each, ~{left/60:.0f} min left",
                      flush=True)

    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
