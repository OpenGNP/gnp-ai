"""Human-readable title and summary for a cluster.

c-TF-IDF gives the terms that separate a topic from its neighbours, which makes
good keywords ("sit, students, activities") and a bad heading. This asks the
local Mistral to read the cluster's own points and name what they share.

The result goes into canonical_topics.canonical_name / canonical_summary and is
also recorded in topic_versions (generated_title, generated_summary,
model_version) — the table the schema already provides for exactly this.
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

MAX_EXAMPLES = 8
TITLE_LIMIT = 255


def name_topic(
    http: httpx.Client, keywords: str, points: list[str]
) -> tuple[str | None, str | None]:
    """Returns (title, summary), or (None, None) when the model's answer is
    unusable — the caller keeps the keyword-derived name in that case."""
    listed = "\n".join(f"- {text}" for text in points[:MAX_EXAMPLES])
    prompt = TEMPLATE.format(keywords=keywords, points=listed)

    raw, _ = llm.chat(http, SYSTEM_PROMPT, prompt, num_predict=160)
    obj = llm.parse_object(raw)
    if not obj:
        return None, None

    title = str(obj.get("title", "")).strip().strip('"')
    summary = str(obj.get("summary", "")).strip().strip('"')
    return (title[:TITLE_LIMIT] or None), (summary or None)
