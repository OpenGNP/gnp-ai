"""Human-readable title and summary for a cluster.

c-TF-IDF gives the terms that separate a topic from its neighbours, which makes
good keywords ("sit, students, activities") and a bad heading. This asks the
local Mistral to read the cluster's own points and name what they share.

The result goes into canonical_topics.canonical_name / canonical_summary.
"""

from __future__ import annotations

import httpx

from . import llm

SYSTEM_PROMPT = "You label clusters of student feedback. You answer with JSON only, no explanation."

TEMPLATE = """These feedback points were grouped together because they are about the same issue.

Distinguishing keywords: {keywords}

Points:
{points}

Write a title and a summary for this group.

title: 2-5 words naming the shared issue as a topic heading, like "Wi-Fi Reliability" or
"Course Pacing". Not a sentence. Do not use the words "feedback", "students" or "issues".
summary: one sentence describing what the group is complaining about or praising, specific
to these points.

Answer with exactly this JSON object and nothing else:
{{"title": "...", "summary": "..."}}"""

MAX_EXAMPLES = 12
TITLE_LIMIT = 255


def _examples(points: list[str], k: int = MAX_EXAMPLES) -> list[str]:
    """Pick k points spread across the whole group, not the k most typical.

    The caller orders members by confidence, so the head of the list is
    whatever sits closest to the centroid — one neighbourhood of what may be
    a large topic. Naming a 123-point group from its eight most central
    members produced "English Communication at International Counter" for a
    topic about staff communication in general. Sampling at even intervals
    shows the model the group's edges as well as its middle.

    Duplicates are dropped first: feedback repeats, and eight copies of one
    sentence spend the whole budget saying one thing.
    """
    seen: set[str] = set()
    unique = [t for t in points if not (t in seen or seen.add(t))]
    if len(unique) <= k:
        return unique
    step = (len(unique) - 1) / (k - 1)
    return [unique[round(i * step)] for i in range(k)]


def name_topic(
    http: httpx.Client, keywords: str, points: list[str]
) -> tuple[str | None, str | None]:
    """Returns (title, summary), or (None, None) when the model's answer is
    unusable or Ollama is down — the caller keeps the keyword name, so
    clustering does not stop with Ollama."""
    listed = "\n".join(f"- {text}" for text in _examples(points))
    prompt = TEMPLATE.format(keywords=keywords, points=listed)

    try:
        raw, _ = llm.chat(http, SYSTEM_PROMPT, prompt, num_predict=160)
    except httpx.HTTPError:
        return None, None
    obj = llm.parse_object(raw)
    if not obj:
        return None, None

    title = str(obj.get("title", "")).strip().strip('"')
    summary = str(obj.get("summary", "")).strip().strip('"')
    return (title[:TITLE_LIMIT] or None), (summary or None)
