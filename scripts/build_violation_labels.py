"""Write the review sheet for the extended violation test set.

The existing data/violation_testset.csv holds 42 hand-written cases, which is
too few to separate two detectors: a 2.4-point difference there is one case.
These are real points drawn from the database, pre-labelled so a human only has
to agree or correct rather than read 1128 points cold.

Sampling is stratified, because a uniform draw is about 97% ordinary complaint
and would measure precision well and recall not at all:

    flagged   every point the pipeline currently marks is_violation — these are
              where its false positives live
    keyword   points it did not flag whose wording suggests misconduct
    ordinary  a random 90 of the rest, which is where false negatives hide and
              where the false-positive rate on normal feedback is measured

`claude_label` is a proposal, not an answer. `confidence` says where to look
first: every `low` and `medium` row is one where a reasonable person could
disagree, and those are the rows that decide a threshold.

    python -m scripts.build_violation_labels

Fill in `your_label` (leave blank to accept `claude_label`), then:

    python -m scripts.merge_violation_labels
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "violation_labels_review.csv"

# (text, claude_label, confidence, note)
# Labels follow the definitions in pipeline/violation.py: legal = a crime,
# ethics = a breach of professional or academic conduct, moral = serious
# dishonesty or treatment that strips a person of dignity, none = everything
# else however angrily written.
FLAGGED: list[tuple[str, str, str, str]] = [
    ("After complaining about LNG103, the same staff member started marking me down.",
     "ethics", "high", "retaliation for a complaint"),
    ("A lecturer asked for 5,000 baht in exchange for a passing grade in CSC102",
     "legal", "high", "bribery"),
    ("A lecturer disclosed a student's mental health record to the entire class.",
     "ethics", "high", "breach of confidentiality"),
    ("A lecturer discussed a student's medical record in front of the whole class.",
     "ethics", "high", "breach of confidentiality"),
    ("A lecturer in LNG103 asked students for money in exchange for a passing grade.",
     "legal", "high", "bribery"),
    ("A lecturer took photos of students in class without consent",
     "ethics", "medium", "privacy; the original answer added posting them, which would make it legal"),
    ("An advisor humiliated a student in front of the class for not affording the field trip.",
     "moral", "high", "humiliation over economic status"),
    ("Announcements are only written/made in Thai, making international students feel behind and excluded.",
     "none", "high", "communication complaint; model false positive"),
    ("Announcements for international students are in Thai-only",
     "none", "high", "model false positive"),
    ("Announcements for international students are in Thai-only.",
     "none", "high", "model false positive"),
    ("Announcements for international students are Thai-only",
     "none", "high", "model false positive"),
    ("Announcements for international students are Thai-only and have been the case all term.",
     "none", "high", "model false positive"),
    ("Announcements for international students are Thai-only (issue with communication)",
     "none", "high", "model false positive"),
    ("Announcements for international students are Thai-only, though I understand the constraints.",
     "none", "high", "model false positive"),
    ("A senior student threatened me after I reported him to the department.",
     "legal", "high", "threat, and retaliation for reporting"),
    ("International students get to participate in 3rd year",
     "none", "high", "not a complaint at all; model false positive"),
    ("I raised an issue last semester about inconsistent answers to the same question depending on who asks.",
     "none", "high", "ordinary complaint; model false positive"),
    ("It is frustrating that announcements for international students are Thai-only.",
     "none", "high", "model false positive"),
    ("My laptop was stolen from the CB2 common room", "legal", "high", "theft"),
    ("My laptop was taken from CB2306", "legal", "medium", "theft, though 'taken' is less explicit"),
    ("One lecturer inflates scores for the same course, resulting in different grades by section.",
     "ethics", "high", "grade manipulation"),
    ("Professor made sexist comment about engineering not being a field for women (regarding CSC102)",
     "moral", "high", "gender discrimination"),
    ("Professor offered credit in exchange for unpaid work",
     "ethics", "high", "abuse of authority; academic credit traded for labour"),
    ("Professor pressured students to work at his startup as unpaid interns",
     "ethics", "high", "abuse of authority"),
    ("Senior students force juniors to do their assignments (tradition)",
     "moral", "high", "hazing and coercion"),
    ("Students in my section copied the whole assignment", "ethics", "high", "cheating"),
    ("TA shared LNG103 answer key with his own friends before the exam",
     "ethics", "high", "leaking an answer key, and favouritism"),
    ("TA shared the answer key with his own friends the night before the exam",
     "ethics", "high", "leaking an answer key, and favouritism"),
    ("The instructor for INT202 does not just provide answers for the course",
     "none", "high", "reads as praise; model false positive"),
    ("The issue I raised last semester regarding inconsistency in answers to the same question has not been addressed.",
     "none", "high", "ordinary complaint; model false positive"),
    ("The lab supervisor made international students work unpaid overtime",
     "legal", "medium", "labour violation; ethics is defensible"),
    ("The lecturer did this to shame the students", "moral", "medium",
     "fragment, but the intent to humiliate is explicit"),
    ("The lecturer for CSC102 gives higher marks to students who pay for his private tutoring.",
     "ethics", "high", "grades traded for money"),
    ("The lecturer for CSC210 gives higher marks to students who pay for his private tutoring.",
     "ethics", "high", "grades traded for money"),
    ("The lecturer in CSC102 is asking students for money in exchange for a passing grade",
     "legal", "high", "bribery"),
    ("This issue has been ongoing throughout the term in the INT202 course",
     "none", "high", "contentless fragment; model false positive"),
    ("Two friends handed in similar work and received different marks",
     "none", "medium", "reads as inconsistent marking, not an accusation"),
]

# The random draw. Three of these are violations the pipeline did not flag —
# they are the false negatives worth knowing about.
ORDINARY_VIOLATIONS = {
    "Professor offered course credit for working at his startup":
        ("ethics", "high", "abuse of authority; MODEL MISSED THIS"),
    "The lab supervisor hinted that our visas depend on working overtime":
        ("legal", "high", "coercion using immigration status; MODEL MISSED THIS"),
    "The lecturer read out the failing students' grades out loud":
        ("moral", "high", "public humiliation; MODEL MISSED THIS"),
}

# Ordinary rows a reasonable reviewer might move. Called out so they are read
# rather than skimmed past.
ORDINARY_BORDERLINE = {
    "Internship Support: The faculty needs to provide much more support for internships. "
    "Jobs for international students are very rare and often pay less than positions for "
    "Thai students. I would like to see partnerships similar to those in the Engineering "
    "faculty to make finding internships easier.":
        "describes market conditions rather than an act by the faculty",
    "Students from other international majors had already been preparing since the previous "
    "semester for opportunities that were not well communicated to Burmese students.":
        "unequal communication; could be read as discrimination",
    "Important academic and opportunity-related information does not always reach Burmese "
    "students (or other international students) in a timely or clear manner, specifically "
    "regarding the Taiwan internship or related programs.":
        "unequal communication; could be read as discrimination",
}

ORDINARY: list[str] = [
    "Someone needs to look at the issue of the cafeteria closing before evening classes finish",
    "CB2306 was refurbished and is now more usable",
    "The project in CSC210 makes a real difference",
    "Marking in INT202 was fair and matched what was studied.",
    "I liked seeing the variety of projects and talking to the presenters.",
    "The tour route worked well",
    "We work in the corridor instead",
    "writing tables inside classroom are small in most rooms except for CB2312",
    "We estimated the number of visitors lower than the actual count",
    "The CSC231 midterm included topics that were not taught.",
    "Large student population may cause delays in responsiveness",
    "The SIT staffs are good",
    "Urgent messages to the office are not being handled promptly",
    "The lunch queue taking more than twenty minutes on enrollment day",
    "Everything was alright overall",
    "Loss of voice during afternoon session",
    "Material for DSI201 is split between the LEB2 system and LINE — which is how I submitted late twice.",
    "It would be helpful if the program did cover deployment",
    "Professor offered course credit for working at his startup",
    "The chairs in the second floor study area are painful to sit on for a full session.",
    "There are not enough project-based courses early on",
    "Internship Support: The faculty needs to provide much more support for internships. "
    "Jobs for international students are very rare and often pay less than positions for "
    "Thai students. I would like to see partnerships similar to those in the Engineering "
    "faculty to make finding internships easier.",
    "A few lab computers in the CB building won't run Docker",
    "The CSC210 lecturer moves on before anyone has followed.",
    "LNG103 jumps to frameworks before covering the fundamentals. This has been the case all term.",
    "A queue built up in front of me while I was guessing how the check-in sheet works",
    "Everything is fine, all the class/lab is good",
    "The INT202 lecturer moves on before anyone has followed",
    "The CSC102 final should be twice as long",
    "The lecturer for CSC231 reads the slides without explaining any reasoning. This has been the case all term.",
    "Understand the constraints of activity sign-ups",
    "The door to the co-working space has not been broken since the start of term",
    "They respond a bit slow",
    "The lab supervisor hinted that our visas depend on working overtime",
    "It would be helpful if this issue was less frequent",
    "Other faculties manage this better — here uploads fail halfway through on the SIT building network.",
    "The cafeteria closes before evening classes finish and has been doing so throughout the term.",
    "The co-working space was refurbished and is far more usable now.",
    "The faculty should increase the number of support staff similar to P'Nicky and P'Vicky.",
    "The machines in the co-working space have been upgraded and they run properly now.",
    "The course is considered a complete waste of time",
    "The printing service in the library is cash only and we work in the corridor instead",
    "The assistance provided in English depends on who is at the counter.",
    "The internship is two months, which is too short to contribute anything",
    "Every term, CSC102 assumes prior experience that most of us do not have",
    "In year 2, the pace is considered too fast and many students fall behind.",
    "Lab machines in CB2312 cannot run Docker for CSC231",
    "The Faculty of Engineering and Science has its own International Affairs office, which is "
    "very active in regularly posting and announcing overseas opportunities. In comparison, SIT "
    "IR has been relatively quiet, and some students are not even aware that the office exists.",
    "The new Visa system recently adopted by IA KMUTT",
    "The lab machines take fifteen minutes to log in and no changes have been made since last semester.",
    "Course announcements disappear in the scroll on Microsoft Teams",
    "Students from other international majors had already been preparing since the previous "
    "semester for opportunities that were not well communicated to Burmese students.",
    "The afternoon crowd was smaller",
    "The brochures ran out by two o'clock",
    "It might be possible for SIT to open the study room until night, not just until 7 pm",
    "The balance between theory and hands-on labs is appreciated.",
    "The printing service in the library is cash only",
    "The response is slow",
    "Everything for CSC231 now lands in one place on Google Classroom.",
    "It is frustrating that nothing in the programme covers deployment or infrastructure.",
    "The other thing is Small thing, but translated material arrives after the event has happened",
    "Allow pair-programming in the common rooms",
    "Study rooms are great but there are never enough of them before exams",
    "The wifi in S11 has been stable all semester.",
    "The first assignment is a wall",
    "Inclusion of real-life stuff",
    "Facilities are okay overall",
    "Important academic and opportunity-related information does not always reach Burmese "
    "students (or other international students) in a timely or clear manner, specifically "
    "regarding the Taiwan internship or related programs.",
    "The lecturer read out the failing students' grades out loud",
    "Language and General Education Courses (LNG and GEN) are not useful and waste a lot of "
    "time for students with very high IELTS scores who still have to take basic language "
    "classes. The grading standards are unclear and difficult to verify.",
    "The door to CB2312 has been broken since the start of term",
    "Gets logged out of the campus WiFi every time moves between rooms",
    "The wifi asks me to re-authenticate three or four times a day",
    "It is frustrating that a single request requires three signatures from three offices.",
    "Discover the registration queue by programme halfway through",
    "My main issue this term is that the elective list has not changed in three years.",
    "This issue has been ongoing all term.",
    "It is frustrating that schedule changes are announced",
    "CSC102 assumes prior experience that most of us do not have",
    "The lab machines take fifteen minutes just to log in and this has been the case all term.",
    "Every term, orientation covered nothing about visas",
    "Every term, the lecturer for LNG103 reads the slides without explaining any reasoning.",
    "The registration system goes down before exam week (issue persists from last semester)",
    "Understood constraints",
    "Every term, the printing service in the library is cash only",
    "I enjoyed talking to the people who created them",
    "No ultimate solution is provided, but something needs to be done.",
    "The air conditioning in S11 should be either off or freezing, as it currently is not.",
    "Three of us share one screen at the printers near CB2",
    "CB2 classroom tables are small and cramped, making it hard to take notes and focus on lectures.",
]


def main() -> None:
    rows = []
    for text, label, confidence, note in FLAGGED:
        rows.append({
            "text": text, "claude_label": label, "confidence": confidence,
            "note": note, "stratum": "flagged", "model_flagged": "yes",
            "your_label": "",
        })

    for text in ORDINARY:
        if text in ORDINARY_VIOLATIONS:
            label, confidence, note = ORDINARY_VIOLATIONS[text]
        elif text in ORDINARY_BORDERLINE:
            label, confidence, note = "none", "medium", ORDINARY_BORDERLINE[text]
        else:
            label, confidence, note = "none", "high", ""
        rows.append({
            "text": text, "claude_label": label, "confidence": confidence,
            "note": note, "stratum": "ordinary", "model_flagged": "no",
            "your_label": "",
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "n", "text", "claude_label", "your_label", "confidence", "note",
            "stratum", "model_flagged",
        ])
        writer.writeheader()
        for i, row in enumerate(rows, 1):
            writer.writerow({"n": i, **row})

    from collections import Counter
    labels = Counter(r["claude_label"] for r in rows)
    disagree = sum(
        1 for r in rows
        if (r["claude_label"] != "none") != (r["model_flagged"] == "yes")
    )
    print(f"wrote {len(rows)} rows to {OUT}")
    print(f"  proposed labels : {dict(labels)}")
    print(f"  rows where the pipeline and this proposal disagree : {disagree}")
    print(f"  rows to read first (confidence != high) : "
          f"{sum(1 for r in rows if r['confidence'] != 'high')}")


if __name__ == "__main__":
    main()
