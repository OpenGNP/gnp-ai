"""Split one free-text answer into atomic points.

The prompt and the JSON recovery logic are the tuned artifacts of the
experiments in scripts/decompose_ollama.py; they are imported, not copied, so
there is exactly one version of each.
"""

from __future__ import annotations

import re

import httpx

from scripts.decompose_ollama import (
    SYSTEM_PROMPT,
    build_prompt,
    build_repair_prompt,
    narrated,
    parse_output,
    parse_points,
)

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
    if narrated(points):
        points, status = _repair_voice(http, text, points, status)
    return points, status


def _repair_voice(
    http: httpx.Client, text: str, points: list[str], status: str
) -> tuple[list[str], str]:
    """Ask again for the same points in the writer's own voice.

    A plain retry cannot work: decoding is temperature 0 (llm.py), so the same
    prompt returns the same bytes. This sends a different, narrower one — the
    points are already decided, and only the person they are told in is at
    stake, which is why it succeeds where forbidding the voice up front did
    not.

    The repair is accepted only if it returns the same number of points and
    reports fewer of them from the outside. Anything else keeps the original:
    a point in the wrong voice still carries the respondent's content, and
    content is the thing that must not be lost here.

    The same reasoning covers the call failing outright. This is a second
    request per affected answer, so it doubles the chances of meeting the 180s
    client timeout, and a voice that stayed in the third person is a blemish
    while a raised ReadTimeout would end the whole ingest — taking the
    decompositions already done in this chunk with it.
    """
    best, best_status = points, f"{status}+repair_failed"

    # Two attempts. One pass reliably removes the narration it opens a sentence
    # with and often leaves a "the user" buried mid-clause; feeding its own
    # output back clears those. A third pass has never been observed to add
    # anything, and every pass is a whole model call.
    for _ in range(2):
        try:
            raw, _ = llm.chat(http, SYSTEM_PROMPT, build_repair_prompt(text, best))
        except httpx.HTTPError:
            return best, f"{status}+repair_error"

        repaired, _ = parse_points(raw)
        if len(repaired) != len(points) or not _grounded(repaired, text):
            return best, best_status if best is not points else f"{status}+repair_shape"
        if len(narrated(repaired)) >= len(narrated(best)):
            return best, best_status
        best, best_status = repaired, f"{status}+repaired"
        if not narrated(best):
            break

    return best, best_status


_WORD = re.compile(r"[a-z0-9']+")
# Words a rewrite is free to introduce: they carry no content, so requiring them
# to appear in the original would reject honest rephrasing.
_FUNCTION_WORDS = frozenset(
    "a an the i we my our me us they them their it its is are was were be been am "
    "do does did have has had will would can could should of to in on at for with "
    "and or but so that this these those there here as by from not no".split()
)


def _grounded(points: list[str], text: str) -> bool:
    """Is every point still about what the respondent actually wrote?

    The repair prompt carries worked examples, and the model has been seen
    copying them into its answer: an answer about the curriculum came back with
    "I want Lockers." among its points, which is the prompt's own example and
    not something that respondent said. The count check does not catch it —
    the example arrives *instead of* a point, so the length matches.

    Content words are the test. A rewrite in the writer's voice keeps what the
    sentence is about and changes who it is told by, so most of what it names
    must appear in the original answer; a sentence lifted from the prompt names
    things the answer never mentions.
    """
    source = set(_WORD.findall(text.lower()))
    for point in points:
        content = [w for w in _WORD.findall(point.lower()) if w not in _FUNCTION_WORDS]
        if not content:
            continue  # nothing to check against; the count test still applies
        hits = sum(1 for w in content if w in source)
        if hits / len(content) < 0.5:
            return False
    return True
