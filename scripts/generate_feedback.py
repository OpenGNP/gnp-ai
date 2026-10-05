"""Generate synthetic student feedback to grow the corpus for pipeline testing.

Built compositionally rather than from whole-sentence templates: each row picks a
subject, a problem, an optional consequence and a sentence shape independently,
so two rows about the same topic rarely share wording. That matters here — a
fixed pool of phrasings makes rows cluster on sentence shape instead of meaning,
which would flatter the clustering numbers.

Still synthetic. Keep the `source` column and report real and synthetic results
separately in any write-up.

    python scripts/generate_feedback.py --count 500
"""

from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BUILDINGS = ["CB1", "CB2", "the LX building", "the SIT building", "S11"]
ROOMS = ["CB2306", "CB2312", "LX10", "the co-working space", "the common room",
         "Lab 1", "the second floor study area"]
COURSES = ["CSC102", "CSC122", "CSC210", "CSC231", "DSI201", "INT202", "LNG103"]
TOOLS = ["LINE", "Microsoft Teams", "email", "the LEB2 system", "Google Classroom"]
TIMES = ["during lab sessions", "in the afternoon", "before exam week", "on enrollment day",
         "during evening classes", "at lunchtime", "at the start of term"]

# topic -> (complaint clauses, praise clauses, consequences)
# Clauses are whole subject+predicate units. Composing subjects and predicates
# separately produced agreement errors ("it are translated") and nonsense pairs
# ("the prayer room has better options"), so variety comes from the clause pool,
# the sentence shape and the slot fills instead.
TOPICS: dict[str, tuple[list[str], list[str], list[str]]] = {
    "wifi": (
        ["the wifi in {r} drops every few minutes",
         "SIT-Secure never appears in the network list in {b}",
         "the connection in {r} dies as soon as a full class joins",
         "the signal does not reach the back rows of {r}",
         "I get logged out of the campus wifi every time I move between rooms",
         "the wifi asks me to re-authenticate three or four times a day",
         "uploads fail halfway through on the {b} network"],
        ["the wifi in {b} has been stable all semester",
         "network coverage finally reaches every corner of {r}"],
        ["so I tether from my phone instead", "which makes online quizzes a gamble",
         "and I lose whatever I was submitting"],
    ),
    "facilities": (
        ["the chairs in {r} are painful to sit on for a full session",
         "the desks in {r} cannot hold a laptop and a notebook at once",
         "half the power outlets in {r} are dead",
         "the air conditioning in {b} is either off or freezing",
         "the lighting in {r} is too dim to read printed handouts",
         "{r} has not been cleaned properly in weeks",
         "the door to {r} has been broken since the start of term"],
        ["{r} was refurbished and it is far more usable now",
         "{b} is kept clean and the rooms are comfortable"],
        ["so people give up and study off campus", "which makes three-hour labs exhausting",
         "and there is nowhere else free to work"],
    ),
    "equipment": (
        ["the monitors in {r} are too dim to use from more than a metre away",
         "the lab machines in {r} cannot run Docker for {c}",
         "there are never enough monitors in {r} for pair programming",
         "the printers near {b} are out of ink more often than not",
         "the projector in {r} loses signal every ten minutes",
         "the lab machines take fifteen minutes just to log in"],
        ["the machines in {r} were upgraded and everything runs properly now",
         "there are finally enough printers for the whole floor"],
        ["so three of us share one screen", "which eats half the lab session"],
    ),
    "curriculum": (
        ["{c} jumps to frameworks before covering the fundamentals",
         "{c} still teaches tools the industry dropped years ago",
         "{c} repeats most of what we already did in {c}",
         "nothing in the programme covers deployment or infrastructure",
         "{c} assumes prior experience that most of us do not have",
         "the elective list has not changed in three years",
         "there is no course that covers testing in any depth",
         "project work only starts in year three, which is far too late"],
        ["{c} builds up at a pace that actually works",
         "the project in {c} connected theory to something real"],
        ["so the first assignment is a wall", "which shows badly in interviews",
         "and we end up teaching ourselves anyway"],
    ),
    "teaching": (
        ["the lecturer for {c} reads the slides without explaining any reasoning",
         "sessions for {c} get cancelled with no replacement class",
         "the answer to the same question changes depending on who asks",
         "the {c} lecturer moves on before anyone has followed",
         "tutorials for {c} are run by someone who has not seen the assignment",
         "we are told to work it out ourselves whenever we ask for help"],
        ["the instructor for {c} explains the reasoning, not just the answer",
         "the {c} demonstrator stays behind until everyone understands"],
        ["so I learn the material from YouTube", "which is why attendance keeps dropping"],
    ),
    "assessment": (
        ["the {c} midterm covered topics that were never taught",
         "grading in {c} is noticeably different between sections",
         "the assignment brief for {c} changed during submission week",
         "the marking scheme for {c} was never published",
         "we get a numeric grade with no comments at all",
         "the {c} final was twice as long as the time allowed"],
        ["marking in {c} was fair and matched what we studied",
         "assignments in {c} come back with comments I can act on"],
        ["so nobody knew what to revise", "and appeals go nowhere"],
    ),
    "communication": (
        ["course announcements only go out on {t} and disappear in the scroll",
         "deadline changes reach us after the deadline",
         "material for {c} is split between {t} and {t}",
         "schedule changes are announced verbally in class and nowhere else",
         "there is no way to search past announcements on {t}",
         "urgent messages to the office sit unread for days"],
        ["everything for {c} now lands in one place on {t}",
         "changes are announced early enough to plan around"],
        ["so missing one class means missing everything",
         "which is how I submitted late twice"],
    ),
    "admin": (
        ["document approval takes weeks with no status update",
         "the registration system goes down {ti}",
         "every counter asks for a different set of forms",
         "a single request needs three signatures from three offices",
         "nobody can tell me which office my form is sitting in",
         "office hours overlap exactly with our class timetable"],
        ["my request was processed within a day this term",
         "the new online form removed most of the paperwork"],
        ["so I missed the scholarship deadline", "and I had to take a day off to chase it"],
    ),
    "international": (
        ["announcements for international students are Thai-only",
         "activity sign-ups assume everyone reads Thai",
         "translated material arrives after the event has happened",
         "whether I get help in English depends on who is at the counter",
         "orientation covered nothing about visas or residence registration",
         "international students hear about events from friends rather than the faculty"],
        ["the international office replies quickly and in English",
         "activity announcements are bilingual this year"],
        ["so we miss most of what is organised", "which makes it hard to feel part of the faculty"],
    ),
    "career": (
        ["the internship is two months, which is too short to contribute anything",
         "internship dates clash with the exam period",
         "the partner company list is much shorter than other faculties offer",
         "the placements on offer only cover a narrow set of roles",
         "career events are announced a week in advance",
         "nobody explains how the internship grade is decided"],
        ["the CV workshop was genuinely useful",
         "the career fair connected me to the company I now work for"],
        ["so most of us find placements ourselves", "and the paperwork falls on the student"],
    ),
    "welfare": (
        ["the cafeteria closes before evening classes finish",
         "the lunch queue takes twenty minutes {ti}",
         "scholarship conditions exclude anyone who works part time",
         "there are no lockers for students who commute",
         "the prayer room is used as a storage space",
         "there is nowhere to eat on campus after six"],
        ["the cafeteria has far better options than last year",
         "food on campus is cheap and quick"],
        ["so people skip meals before evening labs", "which is hardest on commuting students"],
    ),
    "library": (
        ["the library closes too early {ti}",
         "group study rooms cannot be booked online",
         "there are three copies of a set text for two hundred students",
         "every seat is taken by nine in the morning",
         "the printing service in the library is cash only"],
        ["the library is quiet and well run",
         "library staff find what I need straight away"],
        ["so we work in the corridor instead", "which pushes everyone to cafes off campus"],
    ),
}

# Sentence shapes wrap a whole clause, so any clause fits any shape.
# {cl}=clause {c_}=consequence
NEGATIVE_SHAPES = [
    "{cl_u}.",
    "{cl_u}, {c_}.",
    "{cl_u} — {c_}.",
    "My main issue this term is that {cl}.",
    "It is frustrating that {cl}.",
    "Every term the same thing: {cl}.",
    "I raised this last semester and nothing has changed: {cl}.",
    "Small thing, but {cl}.",
    "Other faculties manage this better — here {cl}.",
    "Worth saying that {cl}.",
    "Please can someone look at this: {cl}.",
    "{cl_u}. This has been the case all term.",
]

POSITIVE_SHAPES = [
    "{cl_u}.",
    "Credit where it is due: {cl}.",
    "{cl_u}, and it makes a real difference.",
    "One thing that works well is that {cl}.",
    "No complaints here: {cl}.",
]

NEUTRAL_SHAPES = [
    "Not a complaint exactly, but {cl}.",
    "It would help a lot if {cl} were not the case.",
    "{cl_u}, though I understand the constraints.",
    "Minor point: {cl}.",
]

VIOLATIONS = [
    "A lecturer in {c} asked students for money in exchange for a passing grade.",
    "Someone was selling the {c} midterm paper in a {t} group before the exam.",
    "My laptop was taken from {r} and the office refused to check the CCTV.",
    "A senior student threatened me after I reported him to the department.",
    "A staff member used department budget to buy equipment for personal use.",
    "The lab supervisor made international students work unpaid overtime and implied our visas depend on it.",
    "A lecturer posted photos of students taken in class on a public page without asking.",
    "The lecturer for {c} gives higher marks to students who pay for his private tutoring.",
    "My advisor put his own name as first author on work I did entirely myself.",
    "A TA shared the {c} answer key with his own friends before the exam.",
    "A lecturer discussed a student's medical record in front of the whole class.",
    "The scholarship was decided before the interviews were held.",
    "After I complained about {c}, the same staff member started marking me down.",
    "A professor pressured students to work at his company for free in exchange for credit.",
    "An advisor humiliated a student in front of the class for not affording the field trip.",
    "A lecturer mocked a student's accent every session until they stopped attending.",
    "Staff refer to international students as 'the foreigners' rather than by name.",
    "A professor told a female student that engineering is not a field for women.",
    "Senior students make juniors do their assignments and call it tradition.",
    "A lecturer read the failing students' grades out loud to shame them.",
]

JOINERS = ["{a} {b_}", "{a} Separately, {b_}", "{a} The other thing is {b_}"]


def fill(text: str, rng: random.Random) -> str:
    for token, pool in (("{b}", BUILDINGS), ("{r}", ROOMS), ("{c}", COURSES),
                        ("{t}", TOOLS), ("{ti}", TIMES)):
        while token in text:
            text = text.replace(token, rng.choice(pool), 1)
    return text


def sentence(rng: random.Random) -> str:
    complaints, praises, consequences = TOPICS[rng.choice(list(TOPICS))]

    roll = rng.random()
    if roll < 0.55:
        shape, clause = rng.choice(NEGATIVE_SHAPES), rng.choice(complaints)
    elif roll < 0.78:
        shape, clause = rng.choice(NEUTRAL_SHAPES), rng.choice(complaints)
    else:
        shape, clause = rng.choice(POSITIVE_SHAPES), rng.choice(praises)

    text = shape.format(cl=clause, cl_u=clause[0].upper() + clause[1:],
                        c_=rng.choice(consequences))
    return fill(text, rng)


def normalise(text: str) -> str:
    """Compare rows on content words only, so two rows that differ solely by
    sentence shape still count as duplicates."""
    return " ".join(sorted(re.findall(r"[a-z0-9]+", text.lower())))


def generate(count: int, seed: int, violation_rate: float, multi_rate: float,
             existing: set[str] | None = None) -> list[str]:
    rng = random.Random(seed)
    seen = {normalise(t) for t in (existing or ())}
    out: list[str] = []
    attempts = 0

    while len(out) < count and attempts < count * 200:
        attempts += 1
        if rng.random() < violation_rate:
            text = fill(rng.choice(VIOLATIONS), rng)
        else:
            text = sentence(rng)
            if rng.random() < multi_rate:
                second = sentence(rng)
                text = rng.choice(JOINERS).format(a=text, b_=second)

        key = normalise(text)
        if key not in seen:
            seen.add(key)
            out.append(text)

    if len(out) < count:
        raise SystemExit(f"only {len(out)} unique rows possible from the current pools")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260913)
    ap.add_argument("--violation-rate", type=float, default=0.06)
    ap.add_argument("--multi-rate", type=float, default=0.25)
    ap.add_argument("--base", type=Path, default=ROOT / "data" / "GNP_Data_clean.csv")
    ap.add_argument("--exclude", nargs="*", default=["P'DJ", "P’DJ"])
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "GNP_Data_full.csv")
    args = ap.parse_args()

    base: list[tuple[str, str]] = []
    dropped = 0
    with args.base.open() as fh:
        for row in csv.DictReader(fh):
            text = row["raw_text"]
            if any(bad in text for bad in args.exclude):
                dropped += 1
                continue
            base.append((text, "violation" if int(row["feedback_id"]) >= 1000 else "real"))

    synthetic = [(t, "synthetic") for t in
                 generate(args.count, args.seed, args.violation_rate, args.multi_rate,
                          existing={t for t, _ in base})]

    rows = base + synthetic
    with args.out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["feedback_id", "raw_text", "source"])
        for i, (text, source) in enumerate(rows, 1):
            w.writerow([i, text, source])

    from collections import Counter
    print(f"{args.out}  {len(rows)} rows  (dropped {dropped})")
    print(" ", dict(Counter(s for _, s in rows)))


if __name__ == "__main__":
    main()
