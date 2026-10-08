"""Load a feedback CSV into the database as one form's submissions.

Normally this goes through the server's POST /api/feedback, which owns
validation and stamps created_at itself. But submission dates have to be
spread across months for the dashboard's trend to show a curve, and the API
always stamps "now" — so the rows are written directly, scoped to one form.

Loading the whole file at once makes every topic appear in the same run, which
is not how the pipeline will ever be used. `--offset/--limit` with `--form-id`
load the file in waves instead, so the corpus grows between pipeline runs the
way it does in production:

    python scripts/load_corpus.py --limit 212 --from 2026-03-16 --to 2026-05-15
    python -m pipeline.run --once
    python scripts/load_corpus.py --form-id 3 --offset 212 --limit 212 \\
        --from 2026-05-16 --to 2026-07-15
    python -m pipeline.run --once
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import db  # noqa: E402

# The dashboard segments feedback by the answers to a form's `demographic`
# section, so a corpus with only a feedback field leaves half of it empty.
# Weights are deliberately uneven — a uniform split makes every segment look
# identical and hides whether the segmentation works at all.
DEMOGRAPHICS: list[tuple[str, list[tuple[str, int]]]] = [
    ("Program of study", [
        ("Computer Science", 45), ("Information Technology", 30),
        ("Data Science", 15), ("International Program", 10),
    ]),
    ("Year of study", [
        ("Year 1", 20), ("Year 2", 30), ("Year 3", 30), ("Year 4", 20),
    ]),
    ("Gender", [
        ("Male", 55), ("Female", 40), ("Prefer not to say", 5),
    ]),
    ("Age group", [
        ("18–20", 45), ("21–23", 40), ("24–26", 10), ("27 or older", 5),
    ]),
]


def create_demographics(cur, form_id: int) -> list[tuple[int, list[int], list[int]]]:
    """Add the demographic section to a new form.

    Returns (field_id, option_ids, weights) per field, ready for random.choices.
    """
    fields = []
    for order, (label, options) in enumerate(DEMOGRAPHICS, start=1):
        cur.execute(
            """
            INSERT INTO form_fields
                (form_id, field_label, field_type, section, is_required,
                 field_order, analyze_with_ai)
            VALUES (%s, %s, 'radio', 'demographic', true, %s, false)
            RETURNING id
            """,
            (form_id, label, order),
        )
        field_id = cur.fetchone()[0]

        option_ids = []
        for option_order, (text, _) in enumerate(options, start=1):
            cur.execute(
                """
                INSERT INTO field_options
                    (field_id, option_label, option_value, option_order)
                VALUES (%s, %s, %s, %s) RETURNING id
                """,
                (field_id, text, text, option_order),
            )
            option_ids.append(cur.fetchone()[0])

        fields.append((field_id, option_ids, [w for _, w in options]))
    return fields


def read_demographics(cur, form_id: int) -> list[tuple[int, list[int], list[int]]]:
    """The same shape for a form the earlier waves already set up."""
    fields = []
    for label, options in DEMOGRAPHICS:
        cur.execute(
            """
            SELECT id FROM form_fields
            WHERE form_id = %s AND field_label = %s AND deleted_at IS NULL
            """,
            (form_id, label),
        )
        row = cur.fetchone()
        if not row:
            continue
        field_id = row[0]
        cur.execute(
            """
            SELECT id FROM field_options
            WHERE field_id = %s AND deleted_at IS NULL ORDER BY option_order
            """,
            (field_id,),
        )
        option_ids = [r[0] for r in cur.fetchall()]
        fields.append((field_id, option_ids, [w for _, w in options][: len(option_ids)]))
    return fields


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=ROOT / "data" / "GNP_Data_full.csv")
    ap.add_argument("--title", default="CS Focus group 2025 [demo3]")
    ap.add_argument("--description", default="")
    ap.add_argument("--column", default="raw_text", help="CSV column holding the feedback")
    ap.add_argument("--from", dest="start", default="2026-03-16")
    ap.add_argument("--to", dest="end", default="2026-09-13")
    ap.add_argument("--seed", type=int, default=20260913)
    ap.add_argument("--form-id", type=int,
                    help="append to this form instead of creating one (for later waves)")
    ap.add_argument("--offset", type=int, default=0, help="skip this many CSV rows")
    ap.add_argument("--limit", type=int, help="load at most this many rows")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = [r[args.column].strip() for r in csv.DictReader(args.csv.open())]
    rows = [t for t in rows if t][args.offset:]
    if args.limit:
        rows = rows[: args.limit]
    start, end = datetime.fromisoformat(args.start), datetime.fromisoformat(args.end)
    span = (end - start).days
    rng = random.Random(args.seed + args.offset)

    print(f"{len(rows)} response(s) from {args.csv.name} "
          f"(rows {args.offset}..{args.offset + len(rows) - 1}), dated {args.start}..{args.end}")
    if args.dry_run:
        print("dry run, nothing written")
        return

    conn = db.connect()
    with conn.transaction():
        with conn.cursor() as cur:
            if args.form_id:
                cur.execute(
                    """
                    SELECT id FROM form_fields
                    WHERE form_id = %s AND analyze_with_ai IS TRUE AND deleted_at IS NULL
                    ORDER BY field_order LIMIT 1
                    """,
                    (args.form_id,),
                )
                row = cur.fetchone()
                if not row:
                    raise SystemExit(f"form {args.form_id} has no analysed field to append to")
                form_id, field_id = args.form_id, row[0]
                demographics = read_demographics(cur, form_id)
            else:
                # inherit owner and organization from an existing form so the rows
                # stay visible to the same admin the dashboard logs in as
                cur.execute("SELECT admin_id, organization_id FROM forms ORDER BY id LIMIT 1")
                admin_id, org_id = cur.fetchone()

                cur.execute(
                    """
                    INSERT INTO forms (admin_id, organization_id, form_title, form_description,
                                       status, access_type, record_name, one_response_per_person,
                                       start_date, end_date)
                    VALUES (%s, %s, %s, %s, 'active', 'organization', false, false, %s, %s)
                    RETURNING id
                    """,
                    (admin_id, org_id, args.title, args.description, start, end),
                )
                form_id = cur.fetchone()[0]

                demographics = create_demographics(cur, form_id)

                # analyze_with_ai defaults to false, and the pipeline reads only
                # the fields it is true for — so an unflagged field is invisible.
                cur.execute(
                    """
                    INSERT INTO form_fields
                        (form_id, field_label, field_type, section, is_required,
                         field_order, analyze_with_ai)
                    VALUES (%s, 'Feedback', 'textarea', 'feedback', true, %s, true)
                    RETURNING id
                    """,
                    (form_id, len(DEMOGRAPHICS) + 1),
                )
                field_id = cur.fetchone()[0]

            for text in rows:
                # weighted toward recent weeks so the trend line has a shape
                when = start + timedelta(
                    days=span * (rng.random() ** 1.6),
                    hours=rng.randrange(8, 20), minutes=rng.randrange(60),
                )
                cur.execute(
                    """
                    INSERT INTO submissions (form_id, anonymous_code, submission_status, created_at)
                    VALUES (%s, %s, 'completed', %s) RETURNING id
                    """,
                    (form_id, f"resp_{uuid.uuid4()}", when),
                )
                submission_id = cur.fetchone()[0]
                cur.execute(
                    """
                    INSERT INTO answers (submission_id, field_id, answer_text, created_at)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (submission_id, field_id, text[:5000], when),
                )
                for demo_field, option_ids, weights in demographics:
                    cur.execute(
                        """
                        INSERT INTO answers
                            (submission_id, field_id, answer_option_id, created_at)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (submission_id, demo_field,
                         rng.choices(option_ids, weights=weights)[0], when),
                    )

    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*), min(created_at), max(created_at) FROM submissions WHERE form_id = %s",
            (form_id,),
        )
        count, first, last = cur.fetchone()
    conn.close()
    print(f"form {form_id}: {count} submissions, {first:%Y-%m-%d} .. {last:%Y-%m-%d}")


if __name__ == "__main__":
    main()
