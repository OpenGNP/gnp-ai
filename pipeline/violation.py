"""Violation / toxicity flag for one atomic point.

unitary/toxic-bert is multi-label (toxic, severe_toxic, obscene, threat, insult,
identity_hate). A point is flagged when any label clears the threshold, which
maps onto the single boolean the schema offers: points.is_severe, counted by
/api/analytics/summary as severeIssues.

Note what this does and does not measure: it detects abusive or offensive
*language*, not how serious the *problem being reported* is. "The lab has no
power outlets" is a serious complaint and scores near zero here.
"""

from __future__ import annotations

import httpx

from . import config, llm

_classifier = None


def classifier():
    global _classifier
    if _classifier is None:
        from transformers import pipeline as hf_pipeline

        _classifier = hf_pipeline(
            "text-classification",
            model=config.VIOLATION_MODEL,
            truncation=True,
            top_k=None,
            device=-1,
        )
    return _classifier


def flag(texts: list[str]) -> list[bool]:
    if not texts:
        return []

    flags: list[bool] = []
    for scores in classifier()(texts, batch_size=config.CLASSIFIER_BATCH_SIZE):
        worst = max((entry["score"] for entry in scores), default=0.0)
        flags.append(worst >= config.VIOLATION_THRESHOLD)
    return flags


# ── LLM backend ────────────────────────────────────────────────────────
# toxic-bert answers "is the wording abusive". The question the dashboard
# actually needs is "does this report misconduct" — a legal, ethical or moral
# breach, which is a different axis: a calmly worded report of a lecturer taking
# bribes is a violation, and an angry rant about the cafeteria is not.

LLM_SYSTEM_PROMPT = (
    "You review student feedback for reports of misconduct. You answer with JSON only."
)

LLM_TEMPLATE = """Decide whether this student feedback reports a VIOLATION.

A violation is feedback that reports one of:
- legal: a crime or breach of law — theft, fraud, drugs, weapons, physical violence,
  sexual offence, illegal discrimination.
- ethics: a breach of professional or academic conduct — cheating, plagiarism, bribery,
  favouritism, conflict of interest, abuse of authority, harassment, retaliation,
  leaking confidential information, an instructor neglecting a duty of care.
- moral: serious dishonesty, exploitation, or treatment that strips a person of dignity.

NOT a violation, no matter how angrily it is written:
- ordinary complaints about quality, speed, facilities, workload, cost or scheduling
- slow or unresponsive staff, unclear instructions, outdated material, hard exams
- rude or emotional wording in the feedback itself

Feedback:
{text}

Answer with exactly this JSON object and nothing else:
{{"category": "...", "reason": "..."}}

category must be exactly one of: legal, ethics, moral, none.
Use "none" for ordinary complaints. reason: at most 12 words."""

CATEGORIES = {"legal", "ethics", "moral", "none"}


def classify_llm(http: httpx.Client, text: str) -> tuple[bool | None, str, str]:
    """Returns (is_violation, category, reason), or (None, ...) when the answer
    cannot be parsed.

    The category is the only field asked for: an earlier version also asked for a
    boolean and the model copied the `false` shown in the example template even
    while naming a real category ("ethics" / "Favoritism"). One field cannot
    contradict itself.
    """
    raw, _ = llm.chat(http, LLM_SYSTEM_PROMPT, LLM_TEMPLATE.format(text=text), num_predict=96)
    obj = llm.parse_object(raw)
    if not obj:
        return None, "none", "unparseable"

    category = str(obj.get("category", "")).strip().lower()
    if category not in CATEGORIES:
        return None, "none", "unparseable"

    return category != "none", category, str(obj.get("reason", "")).strip()[:120]


def flag_llm(http: httpx.Client, texts: list[str]) -> list[bool | None]:
    return [classify_llm(http, text)[0] for text in texts]


# ── zero-shot NLI backend ──────────────────────────────────────────────
# An NLI model answers "does text A imply hypothesis B", so the classification
# task is expressed as hypotheses we write ourselves. Unlike toxic-bert the
# labels are not fixed by the training set, and unlike the LLM there is no
# generation step — one forward pass per label, on CPU.

# Wording matters more than the model here. An earlier version said "abuse of
# authority" under ethics, which pulled in every complaint about a lecturer
# ("reads the slides without explaining") — the biggest source of false
# positives in the demo3 run. Each label now names concrete acts, and the "none"
# label explicitly claims teaching quality and facilities so those complaints
# have somewhere correct to land.
ZEROSHOT_LABELS = {
    "a crime such as bribery, theft, fraud, threats of violence or sexual misconduct": "legal",
    "academic or professional dishonesty such as exam cheating, plagiarism, leaking "
    "answer keys, trading grades for money, or punishing a student for complaining": "ethics",
    "discrimination or harassment aimed at a person's gender, nationality, race or "
    "economic status, or publicly humiliating a person": "moral",
    "an ordinary complaint about teaching quality, facilities, workload, scheduling "
    "or administration": "none",
}

ZEROSHOT_HYPOTHESIS = "This feedback reports {}."

_zeroshot = None


def zeroshot():
    global _zeroshot
    if _zeroshot is None:
        from transformers import pipeline as hf_pipeline

        _zeroshot = hf_pipeline(
            "zero-shot-classification",
            model=config.VIOLATION_ZEROSHOT_MODEL,
            device=-1,
        )
    return _zeroshot


def classify_zeroshot(text: str) -> tuple[bool, str, str]:
    """Returns (is_violation, category, reason). The reason carries the winning
    score so a threshold can be tuned from the report."""
    labels = list(ZEROSHOT_LABELS)
    result = zeroshot()(text, labels, hypothesis_template=ZEROSHOT_HYPOTHESIS, multi_label=False)

    top_label = result["labels"][0]
    top_score = result["scores"][0]
    category = ZEROSHOT_LABELS[top_label]

    flagged = category != "none" and top_score >= config.VIOLATION_ZEROSHOT_THRESHOLD
    return flagged, category, f"{top_score:.2f}"


def flag_zeroshot(texts: list[str]) -> list[bool]:
    return [classify_zeroshot(text)[0] for text in texts]
