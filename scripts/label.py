"""Label points as violation / not, one keypress at a time.

    python scripts/label.py data/labels/round1.csv

Reads a CSV with `id` and `text`, writes a `label` column beside them. Already
answered rows are skipped, so the file is both the task and the progress: stop
with `q` and run the same command again to carry on.

The model's score is never displayed even when the input file carries one.
Seeing it turns the job into agreeing with the model, and the whole point of
labelling is to find out where the model is wrong.

Keys: y = violation · n = not · s = unsure · b = go back one · q = save and quit
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GUIDELINE = ROOT / "docs" / "violation_guideline.md"

ANSWERS = {"y": "1", "n": "0", "s": "unsure"}
SAVE_EVERY = 20


def _read(path: Path) -> list[dict]:
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{path} is empty")
    for row in rows:
        row.setdefault("label", "")
    return rows


def _write(path: Path, rows: list[dict]) -> None:
    # Only the columns the task needs: a score column in the input is dropped
    # here so a finished file cannot leak it back into a later round.
    fields = ["id", "text", "label"]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: row.get(k, "") for k in fields} for row in rows)
    tmp.replace(path)


def _tally(rows: list[dict]) -> str:
    done = [r for r in rows if r["label"]]
    yes = sum(1 for r in done if r["label"] == "1")
    no = sum(1 for r in done if r["label"] == "0")
    unsure = sum(1 for r in done if r["label"] == "unsure")
    return f"{len(done)}/{len(rows)} done · violation {yes} · not {no} · unsure {unsure}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", type=Path)
    ap.add_argument("--relabel", action="store_true",
                    help="start over, ignoring labels already in the file")
    args = ap.parse_args()

    rows = _read(args.file)
    if args.relabel:
        for row in rows:
            row["label"] = ""

    print(f"\n{args.file.name} — {_tally(rows)}")
    print(f"guideline: {GUIDELINE.relative_to(ROOT)}")
    print("y = violation   n = not   s = unsure   b = back   q = save and quit\n")

    i = 0
    since_save = 0
    while True:
        while i < len(rows) and rows[i]["label"]:
            i += 1
        if i >= len(rows):
            break

        row = rows[i]
        print(f"[{i + 1}/{len(rows)}] " + "─" * 56)
        print(f"  {row['text']}\n")
        try:
            key = input("  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if key == "q":
            break
        if key == "b":
            # step back over the previous answered row and clear it
            j = i - 1
            while j >= 0 and not rows[j]["label"]:
                j -= 1
            if j >= 0:
                rows[j]["label"] = ""
                i = j
            else:
                print("  (nothing to go back to)")
            continue
        if key not in ANSWERS:
            print("  (y / n / s / b / q)")
            continue

        row["label"] = ANSWERS[key]
        i += 1
        since_save += 1
        if since_save >= SAVE_EVERY:
            _write(args.file, rows)
            since_save = 0
            print(f"  ── saved · {_tally(rows)}\n")

    _write(args.file, rows)
    print(f"\nsaved {args.file} — {_tally(rows)}")


if __name__ == "__main__":
    main()
