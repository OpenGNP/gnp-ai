"""Thai answers are translated to English before decomposition, so Mistral and
the English classifiers downstream read them like any other answer."""

from __future__ import annotations

import re

from . import config

_THAI = re.compile(r"[฀-๿]")
_model = None


def is_thai(text: str) -> bool:
    return bool(_THAI.search(text))


def _load():
    global _model
    if _model is None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        _model = (
            AutoTokenizer.from_pretrained(config.TRANSLATE_MODEL),
            AutoModelForSeq2SeqLM.from_pretrained(config.TRANSLATE_MODEL),
        )
    return _model


def to_english(text: str) -> str:
    """Unchanged unless it contains Thai."""
    if not is_thai(text):
        return text
    tok, model = _load()
    out = model.generate(**tok([text], return_tensors="pt", truncation=True))
    return tok.decode(out[0], skip_special_tokens=True).strip() or text
