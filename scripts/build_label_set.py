"""Choose which points a human should label, and in what order.

Violations are about 3% of the corpus, so a thousand points drawn at random
would carry thirty of them — not enough to train on, and almost none of them
near the boundary where the decision is actually hard. The sample is drawn
from three bands of the zero-shot score instead:

The bands are read from the category *and* the score together, which the
score alone cannot express: the detector reports the confidence of whichever
label won, so 0.90 on "an ordinary complaint" means confidently *not* a
violation. Treating a high number as "likely a violation" would have filled
the sample with the clearest negatives in the corpus.

A number of items are then repeated far apart in the running order. Consistency
is measured from those repeats rather than from a second pass days later: the
gap in the queue does the work the gap in the calendar would have done, it
costs a minute instead of a day, and it measures the annotator under real
conditions, including the third hour when attention is worst.

    python scripts/build_label_set.py --size 200 --repeats 20 --out round1.csv
    python scripts/build_label_set.py --size 840 --exclude round1.csv --out round2.csv
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POOL = ROOT / "data" / "labels" / "scored_pool.csv"

# name, is the winning label a violation, score range, share of the sample.
# Weighted towards where the model is least sure, since an item it already
# gets right teaches a classifier nothing.
BANDS = [
    ("flagged",         True,  0.75, 1.01, 0.18),  # confident violation
    ("borderline",      True,  0.50, 0.75, 0.22),  # leaning violation, unsure
    ("weak-violation",  True,  0.00, 0.50, 0.22),  # barely leaning violation
    ("uncertain-none",  False, 0.00, 0.60, 0.26),  # unsure it is ordinary
    ("confident-none",  False, 0.60, 1.01, 0.12),  # clear negatives, for balance
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", type=Path, default=POOL)
    ap.add_argument("--size", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=20,
                    help="items shown twice, far apart, to measure consistency")
    ap.add_argument("--gap", type=int, default=100,
                    help="minimum items between a repeat and its first showing")
    ap.add_argument("--exclude", type=Path, nargs="*", default=[],
                    help="label files whose items are already spoken for")
    ap.add_argument("--seed", type=int, default=20261017)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = list(csv.DictReader(args.pool.open()))
    taken = set()
    for path in args.exclude:
        for row in csv.DictReader(path.open()):
            taken.add(row["id"].split("#")[0])
    rows = [r for r in rows if r["point_id"] not in taken]

    rng = random.Random(args.seed)
    chosen: list[dict] = []
    for name, violation, lo, hi, share in BANDS:
        band = [r for r in rows
                if (r["category"] != "none") == violation
                and lo <= float(r["score"]) < hi]
        want = round(args.size * share)
        rng.shuffle(band)
        picked = band[:want]
        chosen.extend(picked)
        print(f"  {name:<16} {len(band):>5} available, took {len(picked)}")

    if len(chosen) < args.size:
        print(f"  (only {len(chosen)} available, wanted {args.size})")

    rng.shuffle(chosen)
    queue = [dict(r, id=r["point_id"]) for r in chosen]

    # Repeats go in second, far enough back that the first showing is out of
    # mind. The suffix keeps the two rows distinct in the file while leaving
    # the original point id recoverable.
    if args.repeats:
        gap = min(args.gap, max(1, len(queue) // 2))
        sources = rng.sample(range(min(len(queue), len(queue) - gap)), args.repeats)
        for offset, index in enumerate(sorted(sources)):
            copy = dict(queue[index])
            copy["id"] = f"{copy['id']}#r"
            at = min(len(queue), index + gap + offset)
            queue.insert(at, copy)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "text", "label"])
        writer.writeheader()
        for row in queue:
            writer.writerow({"id": row["id"], "text": row["text"], "label": ""})

    print(f"\nwrote {args.out}: {len(queue)} rows "
          f"({len(chosen)} distinct + {len(queue) - len(chosen)} repeats)")
    print("the score is deliberately not in the file")


if __name__ == "__main__":
    main()
