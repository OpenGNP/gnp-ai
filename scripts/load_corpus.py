"""Load a feedback CSV into the database as one form's submissions.

Normally this goes through the server's POST /api/feedback, which owns
validation and stamps created_at itself. That route is unavailable while the
server points at a schema that no longer exists, and submission dates have to
be spread across months anyway for the dashboard's trend to show a curve —
something the API cannot express. So the rows are written directly, scoped to
one new form.

    python scripts/load_corpus.py --csv data/GNP_Data_full.csv --title "CS Focus group 2025 [demo3]"
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
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = [r[args.column].strip() for r in csv.DictReader(args.csv.open())]
    rows = [t for t in rows if t]
    start, end = datetime.fromisoformat(args.start), datetime.fromisoformat(args.end)
    span = (end - start).days
    rng = random.Random(args.seed)

    print(f"{len(rows)} response(s) from {args.csv.name}, dated {args.start}..{args.end}")
    if args.dry_run:
        print("dry run, nothing written")
        return

    conn = db.connect()
    with conn.transaction():
        with conn.cursor() as cur:
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

            cur.execute(
                """
                INSERT INTO form_fields (form_id, field_label, field_type, is_required, field_order)
                VALUES (%s, 'Feedback', 'textarea', true, 1) RETURNING id
                """,
                (form_id,),
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
