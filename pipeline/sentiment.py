"""Sentiment for one atomic point.

cardiffnlp/twitter-roberta-base-sentiment-latest emits exactly the three labels
points.sentiment_label allows (chk_sentiment: positive | neutral | negative), so
no mapping table is needed. It runs on CPU in milliseconds, which is why the
classifier stages are batched here rather than called per point like the LLM.
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
            model=config.SENTIMENT_MODEL,
            truncation=True,
            device=-1,
        )
    return _classifier


VALID = {"positive", "neutral", "negative"}


def classify(texts: list[str]) -> list[str | None]:
    """One label per input, None when the model returns something unexpected —
    a NULL column is honest, a guessed label is not."""
    if not texts:
        return []

    labels: list[str | None] = []
    for result in classifier()(texts, batch_size=config.CLASSIFIER_BATCH_SIZE):
        label = str(result["label"]).strip().lower()
        labels.append(label if label in VALID else None)
    return labels
