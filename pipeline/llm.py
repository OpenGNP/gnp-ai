"""Single Ollama chat entry point, shared by decomposition and sentiment.

Both steps hit the same local model with the same decoding options; only the
system prompt differs. Keeping one caller means timeouts and retries are
defined once.
"""

from __future__ import annotations

import json
import re
import time

import httpx

from . import config

_OBJECT_RE = re.compile(r"\{.*?\}", re.DOTALL)


def client() -> httpx.Client:
    return httpx.Client(timeout=180.0)


def chat(http: httpx.Client, system: str, prompt: str, num_predict: int = 512) -> tuple[str, float]:
    started = time.perf_counter()
    resp = http.post(
        f"{config.OLLAMA_URL}/api/chat",
        json={
            "model": config.DECOMPOSE_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.0, "repeat_penalty": 1.05, "num_predict": num_predict},
        },
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"], time.perf_counter() - started


def parse_object(raw: str) -> dict | None:
    """Same two-stage recovery as parse_points() in scripts/decompose_ollama.py,
    for a JSON object instead of an array."""
    text = raw.strip()
    candidates = [text]
    match = _OBJECT_RE.search(text)
    if match:
        candidates.append(match.group(0))

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj
    return None
