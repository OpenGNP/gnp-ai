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

from . import config

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
