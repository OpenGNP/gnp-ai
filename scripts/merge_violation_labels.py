"""Fold the reviewed labels into data/violation_testset.csv.

`your_label` wins where the reviewer filled it in; a blank accepts the proposal
in `claude_label`. The merged file keeps the testset's two columns so every
existing caller still reads it, and records where each case came from in a
sibling file rather than adding a column the readers would have to ignore.

    python -m scripts.merge_violation_labels            # report only
    python -m scripts.merge_violation_labels --apply
"""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REVIEW = ROOT / "data" / "violation_labels_review.csv"
TESTSET = ROOT / "data" / "violation_testset.csv"
PROVENANCE = ROOT / "data" / "violation_testset_provenance.csv"

VALID = {"legal", "ethics", "moral", "none"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    if not REVIEW.exists():
        raise SystemExit(f"{REVIEW} not found — run scripts.build_violation_labels first")

    with REVIEW.open() as fh:
        review = list(csv.DictReader(fh))

    merged, overridden, bad = [], 0, []
    for row in review:
        chosen = (row["your_label"] or "").strip().lower() or row["claude_label"].strip().lower()
        if chosen not in VALID:
            bad.append((row["n"], chosen))
            continue
        if (row["your_label"] or "").strip():
            overridden += 1
        merged.append({
            "expected": chosen,
            "text": row["text"],
            "stratum": row["stratum"],
            "source": "reviewed",
        })

    if bad:
        raise SystemExit(f"invalid labels on rows {bad} — must be one of {sorted(VALID)}")

    with TESTSET.open() as fh:
        existing = [
            {"expected": r["expected"], "text": r["text"],
             "stratum": "handwritten", "source": "original"}
            for r in csv.DictReader(fh)
        ]

    # An identical point can reach the set from both halves; the hand-written
    # original wins, since it is the one every past measurement was taken on.
    seen = {r["text"].strip().lower() for r in existing}
    added = [r for r in merged if r["text"].strip().lower() not in seen]
    combined = existing + added

    print(f"reviewed rows      : {len(review)} ({overridden} overridden by hand)")
    print(f"already in testset : {len(merged) - len(added)}")
    print(f"existing cases     : {len(existing)}")
    print(f"after merge        : {len(combined)}")
    print(f"  label mix        : {dict(Counter(r['expected'] for r in combined))}")

    if not args.apply:
        print("\nreport only — pass --apply to write")
        return

    backup = TESTSET.with_suffix(f".{datetime.now():%Y%m%d-%H%M%S}.bak.csv")
    shutil.copy2(TESTSET, backup)
    print(f"\nbacked up {TESTSET.name} to {backup.name}")

    with TESTSET.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["expected", "text"])
        writer.writeheader()
        for row in combined:
            writer.writerow({"expected": row["expected"], "text": row["text"]})

    with PROVENANCE.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["expected", "text", "stratum", "source"])
        writer.writeheader()
        writer.writerows(combined)

    print(f"wrote {len(combined)} cases to {TESTSET.name}")
    print(f"wrote provenance to {PROVENANCE.name}")


if __name__ == "__main__":
    main()
