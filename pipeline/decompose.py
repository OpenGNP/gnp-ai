"""Split one free-text answer into atomic points.

The prompt and the JSON recovery logic are the tuned artifacts of the
experiments in scripts/decompose_ollama.py; they are imported, not copied, so
there is exactly one version of each.
"""

from __future__ import annotations

import re

import httpx

from scripts.decompose_ollama import SYSTEM_PROMPT, build_prompt, parse_output

from . import config, llm

# What a second clause looks like: a list separator, a joining word with space
# on both sides, or a sentence that ends and starts again.
_CLAUSE_BREAK = re.compile(
    r"[,;:]|\s(?:and|but|also|however|although|though|while|plus)\s|[.!?]\s*\S",
    re.IGNORECASE,
)


def needs_splitting(text: str) -> bool:
    """Is this answer long enough, or structured enough, to hold two points?

    A short answer with no clause break cannot: "Need more tree", "I want
    Lockers", "The AC is sometimes too loud" are each exactly one point, and
    across the corpus no answer of two to six words ever split into more than
    one. Sending them to the model is not merely wasted — asked to find
    points in something that holds one, it invents them. Real examples from
    the ingested data:

        "5655655665566655"   -> five complaints about CSC102, its lecturer,
                                its assignments, its textbook and its labs
        "นักขับมืออาชีพ"       -> four points on course selection and credit
                                transfer
        "เท่"                 -> "The user encountered an issue with CSC102"

    None of it was written by anyone. The guard is a word count *and* a
    clause-break test, because length alone would wave through "good food,
    bad wifi" — four words, two points.
    """
    if len(text.split()) > config.SHORT_ANSWER_WORDS:
        return True
    return bool(_CLAUSE_BREAK.search(text))


def decompose(http: httpx.Client, text: str) -> tuple[list[str], str]:
    """Returns (points, status). On a parse failure the whole answer is kept as
    a single point — losing a respondent's feedback is worse than a coarse split."""
    text = text.strip()
    if not needs_splitting(text):
        # Kept verbatim rather than dropped: "good" is still a positive
        # response, and the classifiers downstream read it like any other
        # point. Only the model is skipped.
        return [text], "short"

    prompt = build_prompt(config.DECOMPOSE_PROMPT, text)
    raw, _ = llm.chat(http, SYSTEM_PROMPT, prompt)
    points, status = parse_output(config.DECOMPOSE_PROMPT, raw)

    if not points:
        return [text], "decompose_failed"
    return points, status
