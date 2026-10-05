"""Try a violation detector on real points without touching the database.

Two backends answer different questions:

    toxic-bert  is the wording abusive?
    llm         does this feedback report a legal, ethical or moral breach?

Nothing here writes. Use it to tune the definition, then apply the winner with
`python -m pipeline.run --relabel`.

    python scripts/test_violation.py --backend llm --limit 40
    python scripts/test_violation.py --backend llm --form 16
    python scripts/test_violation.py --backend both --limit 30
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import db, llm, violation  # noqa: E402

REPORTS = ROOT / "data" / "reports"


def load(form_ids: list[int] | None, limit: int | None) -> list[tuple[int, str]]:
    sql = """
        SELECT p.id, p.point_text FROM points p
        WHERE p.point_text IS NOT NULL
    """
    params: list = []
    if form_ids:
        sql += """
            AND p.answer_id IN (
                SELECT a.id FROM answers a
                JOIN submissions s ON s.id = a.submission_id
                WHERE s.form_id = ANY(%s)
            )
        """
        params.append(form_ids)
    sql += " ORDER BY p.id"
    if limit:
        sql += f" LIMIT {int(limit)}"

    conn = db.connect()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return [(row[0], row[1]) for row in cur.fetchall()]
    finally:
        conn.close()


def load_file(path: Path) -> list[tuple[str, str]]:
    """A labelled set: expected,text. Lets the definition be tuned against known
    answers instead of guessing from unlabelled production text."""
    with path.open() as fh:
        return [(row["expected"].strip().lower(), row["text"]) for row in csv.DictReader(fh)]


def score(cases: list[tuple[str, str]], backend: str) -> str:
    """Run one backend over labelled cases and report where it disagrees."""
    rows, correct, flag_wrong, cat_wrong = [], 0, 0, 0
    started = time.perf_counter()

    with llm.client() as http:
        for n, (expected, text) in enumerate(cases, 1):
            if backend == "zeroshot":
                flagged, category, reason = violation.classify_zeroshot(text)
            else:
                flagged, category, reason = violation.classify_llm(http, text)
            should_flag = expected != "none"
            flag_ok = bool(flagged) == should_flag
            cat_ok = flag_ok and (category == expected)

            correct += flag_ok
            flag_wrong += not flag_ok
            cat_wrong += flag_ok and not cat_ok

            rows.append((expected, category, flag_ok, cat_ok, reason, text))
            print(f"  [{n}/{len(cases)}] {'PASS' if flag_ok else 'FAIL'} "
                  f"expected={expected} got={category}", flush=True)

    seconds = time.perf_counter() - started
    lines = [
        "=" * 78,
        f"labelled violation set   backend={backend}   {datetime.now():%Y-%m-%d %H:%M:%S}",
        "=" * 78,
        f"cases              {len(cases)}",
        f"flag correct       {correct}/{len(cases)}  ({correct / len(cases):.0%})",
        f"flag wrong         {flag_wrong}",
        f"category wrong     {cat_wrong}  (flagged correctly, named the wrong kind)",
        f"time               {seconds:.1f}s  ({seconds / len(cases):.2f}s per case)",
        "",
        "-" * 78,
        "DISAGREEMENTS",
        "-" * 78,
    ]
    for expected, got, flag_ok, cat_ok, reason, text in rows:
        if flag_ok and cat_ok:
            continue
        kind = "MISSED" if not flag_ok and expected != "none" else (
            "FALSE POSITIVE" if not flag_ok else "wrong category")
        lines.append(f"\n[{kind}] expected {expected}, got {got}")
        if reason:
            lines.append(f"  reason: {reason}")
        lines.append(f"  {text}")
    if all(f and c for _, _, f, c in [(r[0], r[1], r[2], r[3]) for r in rows]):
        lines.append("\n(none)")
    return "\n".join(lines)


def run_toxic(rows: list[tuple[int, str]]) -> list[tuple[int, str, bool, str, str]]:
    flags = violation.flag([text for _, text in rows])
    return [
        (pid, text, bool(flag), "toxic" if flag else "none", "")
        for (pid, text), flag in zip(rows, flags)
    ]


def run_llm(rows: list[tuple[int, str]]) -> list[tuple[int, str, bool, str, str]]:
    out = []
    with llm.client() as http:
        for n, (pid, text) in enumerate(rows, 1):
            flagged, category, reason = violation.classify_llm(http, text)
            out.append((pid, text, bool(flagged), category, reason))
            print(f"  [{n}/{len(rows)}] point {pid} {'FLAG ' + category if flagged else 'ok'}",
                  flush=True)
    return out


def report(name: str, results: list[tuple[int, str, bool, str, str]], seconds: float) -> str:
    flagged = [r for r in results if r[2]]
    lines = [
        "=" * 78,
        f"violation backend: {name}   {datetime.now():%Y-%m-%d %H:%M:%S}",
        "=" * 78,
        f"points checked   {len(results)}",
        f"flagged          {len(flagged)}  ({len(flagged) / max(len(results), 1):.1%})",
        f"time             {seconds:.1f}s  ({seconds / max(len(results), 1):.2f}s per point)",
    ]

    by_category: dict[str, int] = {}
    for _, _, _, category, _ in flagged:
        by_category[category] = by_category.get(category, 0) + 1
    if by_category:
        lines.append(f"by category      {by_category}")

    lines += ["", "-" * 78, "FLAGGED", "-" * 78]
    for pid, text, _, category, reason in flagged:
        lines.append(f"\n[{category}] point {pid}")
        if reason:
            lines.append(f"  reason: {reason}")
        lines.append(f"  {text}")

    if not flagged:
        lines.append("\n(nothing flagged)")

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["toxic-bert", "llm", "zeroshot", "both"], default="llm")
    ap.add_argument("--form", type=int, nargs="+", metavar="ID")
    ap.add_argument("--limit", type=int, metavar="N")
    ap.add_argument("--file", type=Path, metavar="CSV",
                    help="labelled set with columns expected,text — scores the LLM backend "
                         "against known answers instead of reading the database")
    args = ap.parse_args()

    if args.file:
        cases = load_file(args.file)
        print(f"{len(cases)} labelled case(s)\n")
        text = score(cases, args.backend)
        print("\n" + text)
        REPORTS.mkdir(parents=True, exist_ok=True)
        out = (REPORTS /
               f"violation_labelled_{args.backend}_n{len(cases)}_{datetime.now():%Y%m%d-%H%M%S}.txt")
        out.write_text(text)
        print(f"\nsaved {out}")
        return

    rows = load(args.form, args.limit)
    print(f"{len(rows)} point(s) to check\n")

    backends = ["toxic-bert", "llm"] if args.backend == "both" else [args.backend]
    REPORTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    for name in backends:
        started = time.perf_counter()
        if name == "toxic-bert":
            results = run_toxic(rows)
        elif name == "zeroshot":
            results = [(pid, text, *violation.classify_zeroshot(text)) for pid, text in rows]
        else:
            results = run_llm(rows)
        text = report(name, results, time.perf_counter() - started)
        print("\n" + text)

        path = REPORTS / f"violation_{name}_n{len(rows)}_{stamp}.txt"
        path.write_text(text)
        print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
