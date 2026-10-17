"""Two checks on a finished batch of labels, before more are collected.

    consistency   the repeated items — did the same text get the same answer
                  the second time, a hundred items later
    agreement     how the labels line up against the zero-shot model

Consistency is the ceiling: a classifier cannot learn a boundary the labeller
does not hold steady, so a low number here means the guideline is still vague
and no amount of training will fix it.

Agreement is not a score for either side. The labels are the reference. It is
read to find where the model and the guideline disagree, because those items
are what the model would have to learn, and because a disagreement that looks
wrong on reading is a sign the guideline missed a case.

    python scripts/check_labels.py data/labels/round1.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POOL = ROOT / "data" / "labels" / "scored_pool.csv"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", type=Path)
    ap.add_argument("--pool", type=Path, default=POOL)
    ap.add_argument("--threshold", type=float, default=0.75,
                    help="score at which the zero-shot model calls it a violation")
    ap.add_argument("--show", type=int, default=12, help="examples to print per section")
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(args.file.open()) if r["label"]]
    if not rows:
        raise SystemExit(f"{args.file} has no labels yet")

    counts = Counter(r["label"] for r in rows)
    total = len(rows)
    decided = counts["0"] + counts["1"]
    print(f"{args.file.name}: {total} labelled · violation {counts['1']} "
          f"· not {counts['0']} · unsure {counts['unsure']}")
    if decided:
        print(f"  positive rate (of decided): {counts['1'] / decided:.1%}")

    # ── consistency ────────────────────────────────────────────────────
    by_id: dict[str, list[str]] = {}
    for row in rows:
        by_id.setdefault(row["id"].split("#")[0], []).append(row["label"])
    pairs = {k: v for k, v in by_id.items() if len(v) > 1}

    print(f"\n── consistency ── {len(pairs)} repeated item(s)")
    if pairs:
        same = sum(1 for v in pairs.values() if len(set(v)) == 1)
        rate = same / len(pairs)
        print(f"  answered the same way both times: {same}/{len(pairs)} ({rate:.0%})")
        verdict = ("definitions are sharp enough — go on" if rate >= 0.85 else
                   "fix the cases you flipped on, then re-check" if rate >= 0.75 else
                   "the definition is not usable yet")
        print(f"  {verdict}")

        text_of = {r["id"].split("#")[0]: r["text"] for r in rows}
        flipped = [k for k, v in pairs.items() if len(set(v)) > 1]
        if flipped:
            print("\n  flipped:")
            for key in flipped[: args.show]:
                print(f"    {' → '.join(by_id[key])}  {text_of[key][:78]}")
    else:
        print("  none in this batch — build it with --repeats to measure this")

    # ── agreement with the model ───────────────────────────────────────
    scores = {r["point_id"]: float(r["score"]) for r in csv.DictReader(args.pool.open())}
    texts = {r["point_id"]: r["text"] for r in csv.DictReader(args.pool.open())}
    judged = [(r["id"].split("#")[0], r["label"]) for r in rows if r["label"] in ("0", "1")]
    seen: dict[str, str] = dict(judged)

    tp = fp = fn = tn = 0
    false_pos: list[str] = []
    false_neg: list[str] = []
    for pid, label in seen.items():
        if pid not in scores:
            continue
        model = scores[pid] >= args.threshold
        human = label == "1"
        if model and human:
            tp += 1
        elif model and not human:
            fp += 1
            false_pos.append(pid)
        elif human and not model:
            fn += 1
            false_neg.append(pid)
        else:
            tn += 1

    print(f"\n── zero-shot at {args.threshold} against these labels ──")
    print(f"  agreed {tp + tn}/{tp + tn + fp + fn}"
          f"   model said yes, you said no: {fp}"
          f"   you said yes, model said no: {fn}")
    if tp + fp:
        print(f"  precision {tp / (tp + fp):.2f}", end="")
    if tp + fn:
        print(f"   recall {tp / (tp + fn):.2f}")
    else:
        print()

    for title, ids in (("model said yes, you said no", false_pos),
                       ("you said yes, model said no", false_neg)):
        if not ids:
            continue
        print(f"\n  {title}:")
        for pid in sorted(ids, key=lambda p: -scores[p])[: args.show]:
            print(f"    {scores[pid]:.2f}  {texts[pid][:76]}")

    unsure = [r for r in rows if r["label"] == "unsure"]
    if unsure:
        print(f"\n── unsure ── {len(unsure)}")
        for row in unsure[: args.show]:
            print(f"    {row['text'][:80]}")
        print("  a kind that keeps coming back belongs in the guideline as an edge case")


if __name__ == "__main__":
    main()
