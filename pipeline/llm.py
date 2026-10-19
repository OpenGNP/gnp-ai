"""Single Ollama chat entry point, shared by decomposition and topic naming."""

from __future__ import annotations

import json
import re
import time

import httpx

from . import config

_OBJECT_RE = re.compile(r"\{.*?\}", re.DOTALL)


def client() -> httpx.Client:
    # With streaming, `read` is the gap between tokens: a long answer is never
    # cut off, a hung server is noticed within OLLAMA_READ_TIMEOUT.
    return httpx.Client(
        timeout=httpx.Timeout(
            connect=config.OLLAMA_CONNECT_TIMEOUT,
            read=config.OLLAMA_READ_TIMEOUT,
            write=10.0,
            pool=10.0,
        )
    )


def chat(http: httpx.Client, system: str, prompt: str, num_predict: int = 512) -> tuple[str, float]:
    # Ollama's default num_ctx silently truncates long prompts.
    started = time.perf_counter()
    parts: list[str] = []
    with http.stream(
        "POST",
        f"{config.OLLAMA_URL}/api/chat",
        json={
            "model": config.DECOMPOSE_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": True,
            "options": {
                "temperature": 0.0,
                "repeat_penalty": 1.05,
                "num_predict": num_predict,
                "num_ctx": config.OLLAMA_NUM_CTX,
            },
        },
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            if "error" in chunk:
                # Mid-stream failures arrive as a line, not a status code.
                raise httpx.HTTPError(f"ollama: {chunk['error']}")
            parts.append(chunk.get("message", {}).get("content", ""))
            if chunk.get("done"):
                break
    return "".join(parts), time.perf_counter() - started


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
