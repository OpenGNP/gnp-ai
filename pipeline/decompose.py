"""Split one free-text answer into atomic points.

The prompt and the JSON recovery logic are the tuned artifacts of the
experiments in scripts/decompose_ollama.py; they are imported, not copied, so
there is exactly one version of each.
"""

from __future__ import annotations

import httpx

from scripts.decompose_ollama import SYSTEM_PROMPT, build_prompt, parse_points

from . import config, llm


def decompose(http: httpx.Client, text: str) -> tuple[list[str], str]:
    """Returns (points, status). On a parse failure the whole answer is kept as
    a single point — losing a respondent's feedback is worse than a coarse split."""
    prompt = build_prompt(config.DECOMPOSE_PROMPT, text)
    raw, _ = llm.chat(http, SYSTEM_PROMPT, prompt)
    points, status = parse_points(raw)

    if not points:
        return [text.strip()], "decompose_failed"
    return points, status
