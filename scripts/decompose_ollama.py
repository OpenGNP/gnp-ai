"""ทดสอบ decomposition ด้วย Mistral 7B จริงผ่าน Ollama (รันในเครื่อง)

ตอบคำถามที่ค้างมาตั้งแต่ต้น: prompt ที่ตีพิมพ์ (`"3-5 points"`) กับ prompt ที่
บอกตรง ๆ ว่า "ส่วนใหญ่ไม่ต้องซอย" ต่างกันแค่ไหนจริง ๆ เมื่อรันกับ Mistral ตัวจริง
บนข้อมูลครบทั้ง 111 responses — ไม่ใช่แค่ 26 ชุดที่เป็น multi-issue ที่ทุกการ
ทดลองก่อนหน้าใช้ ทั้งที่ของจริง 88 ใน 111 ชุดควรได้แค่ 1 point

ไม่ใช้ has_multiple_points หรือ is_multi เป็น input เด็ดขาด — โมเดลต้องตัดสินใจ
เองว่าจะซอยกี่จุด เหมือนตอนใช้งานจริงที่ไม่มีใคร annotate ให้ คอลัมน์พวกนี้ใช้
วัดผลอย่างเดียว

ก่อนรัน:
    brew install ollama && brew services start ollama
    ollama pull mistral:7b-instruct

ใช้:
    python scripts/decompose_ollama.py --limit 5       # ลองก่อน
    python scripts/decompose_ollama.py                 # รันครบ 111 x 2 prompt
    python scripts/decompose_ollama.py --variant v1     # รันแค่ prompt เดียว
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REPORTS = DATA / "reports"
OLLAMA_URL = "http://localhost:11434"
MODEL = "mistral:7b-instruct"

SYSTEM_PROMPT = (
    "You analyze user feedback and split it into several clear points "
    "while preserving the user's original wording and emotion."
)

BASE_RULES = """1. Extract ALL distinct ideas mentioned in the feedback. Do NOT remove ideas.
2. Keep the wording as close as possible to the original feedback.
3. Prefer copying phrases directly from the original text instead of rewriting them.
4. Preserve the user's casual tone and wording.
5. Do NOT summarize, shorten, or formalize the feedback.
6. Do NOT improve grammar.
7. Only modify text when necessary to replace vague pronouns with clear nouns.
8. Replace pronouns like "she", "he", "it", "this", "that" with the correct noun
mentioned earlier.
9. GROUP RELATED IDEAS: if multiple sentences describe the same issue, keep them
in the SAME point.
10. Only create a new point when the idea is clearly different.
11. Every major idea in the feedback must appear in the output.
12. If a sentence depends on context, include the subject (for example the course
name like CSC102)"""

# v0 คือ prompt ที่ตีพิมพ์ในเปเปอร์ทุกคำ — ใช้เป็น control เทียบผล
TASKS = {
    "v0": "Task: Split the following user feedback into 3-5 points.",
    "v1": (
        "Task: Split the following user feedback into separate points.\n\n"
        "IMPORTANT: Most feedback describes only ONE issue. Only split into "
        "multiple points when the feedback clearly raises two or more distinct, "
        "unrelated concerns. If in doubt, keep it as a single point."
    ),
}

OUTPUT_BLOCK = """

Output format:
[
"decomposed feedback point",
"another decomposed feedback point"
]
Generate the JSON array once and STOP after the closing ].

User feedback:
{raw_text}

Return ONLY the JSON array."""


def build_prompt(variant: str, raw_text: str) -> str:
    return f"{TASKS[variant]}\n\nRules:\n{BASE_RULES}{OUTPUT_BLOCK.format(raw_text=raw_text)}"


_ARRAY_RE = re.compile(r"\[.*?\]", re.DOTALL)


def parse_points(raw: str) -> tuple[list[str], str]:
    """สองขั้น: parse ตรง ๆ ก่อน ไม่ได้ค่อยดึง JSON array แรกที่เจอ

    คืนสถานะกำกับไว้เสมอ ไม่กลบว่า parse ไม่ผ่าน เพื่อวัด reliability ได้จริง
    """
    text = raw.strip()
    candidates = [(text, "ok")]
    match = _ARRAY_RE.search(text)
    if match:
        candidates.append((match.group(0), "ok_fallback"))

    for candidate, status in candidates:
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if isinstance(obj, list) and all(isinstance(x, str) for x in obj):
            points = [p.strip() for p in obj if p.strip()]
            if points:
                return points, status
    return [], "parse_error"


def call_mistral(client: httpx.Client, prompt: str) -> tuple[str, float]:
    t0 = time.perf_counter()
    resp = client.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.0, "repeat_penalty": 1.05, "num_predict": 512},
        },
        timeout=180.0,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"], time.perf_counter() - t0


# ── metrics (นิยามเดียวกับที่ใช้ตอน fine-tune บน Colab เพื่อให้เทียบกันได้) ──
STOP = set(
    "a an the and or but if then than that this these those is are was were be been being am "
    "do does did doing have has had having i me my we our you your he she it they them his her "
    "its their to of in on at for with as by from about into over after before between out up "
    "down off not no so very can will just should now more most some such only own same too "
    "s t don".split()
)
_WORD = re.compile(r"[a-z0-9']+")


def _words(t: str) -> list[str]:
    return _WORD.findall(t.lower())


def _content(t: str) -> set[str]:
    return {w for w in _words(t) if w not in STOP and len(w) > 2}


def _bigrams(t: str) -> set[tuple[str, str]]:
    w = _words(t)
    return set(zip(w, w[1:]))


def score(raw_text: str, gold: list[str], pred: list[str]) -> dict:
    if not pred:
        return {"coverage": np.nan, "fidelity": np.nan, "pronoun_rate": np.nan,
                "count_exact": False, "over_split": np.nan, "under_split": np.nan}

    joined = " ".join(pred)
    src_c = _content(raw_text)
    out_bg = _bigrams(joined)

    pronouns = {"it", "this", "that", "they", "them", "she", "he", "her", "his", "these", "those"}
    return {
        "coverage": len(src_c & _content(joined)) / len(src_c) if src_c else np.nan,
        "fidelity": len(out_bg & _bigrams(raw_text)) / len(out_bg) if out_bg else np.nan,
        "pronoun_rate": sum(1 for w in _words(joined) if w in pronouns) / max(len(_words(joined)), 1),
        "count_exact": len(pred) == len(gold),
        "over_split": len(pred) > len(gold),
        "under_split": len(pred) < len(gold),
    }


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):
        return self.streams[0].isatty()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["v0", "v1", "both"], default="both")
    ap.add_argument("--limit", type=int, default=None, help="ทดสอบแค่ N ชุดแรก")
    args = ap.parse_args()

    variants = ["v0", "v1"] if args.variant == "both" else [args.variant]

    df = pd.read_csv(DATA / "dataset.csv")
    df["gold"] = df.points_json.map(json.loads)
    if args.limit:
        df = df.head(args.limit)

    REPORTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_path = REPORTS / f"decompose_ollama_{'-'.join(variants)}_n{len(df)}_{stamp}.txt"
    real_stdout = sys.stdout

    with open(report_path, "w") as fh, httpx.Client() as client:
        sys.stdout = Tee(real_stdout, fh)
        try:
            _run(client, df, variants)
        finally:
            sys.stdout = real_stdout
    print(f"บันทึกรายงานที่ {report_path}")


def _run(client: httpx.Client, df: pd.DataFrame, variants: list[str]) -> None:
    print(f"responses {len(df)} ชุด × {len(variants)} prompt = {len(df) * len(variants)} calls")
    print(f"เฉลย: {df.n_points.mean():.2f} points เฉลี่ย · single-issue {(~df.is_multi).sum()} "
          f"· multi-issue {df.is_multi.sum()}\n")

    rows = []
    for variant in variants:
        print(f"=== prompt {variant} ===")
        for i, r in enumerate(df.itertuples(), 1):
            prompt = build_prompt(variant, r.raw_text)
            raw, latency = call_mistral(client, prompt)
            points, status = parse_points(raw)
            metrics = score(r.raw_text, r.gold, points)

            rows.append({
                "variant": variant, "feedback_id": r.feedback_id,
                "gold_n": len(r.gold), "pred_n": len(points), "is_multi": r.is_multi,
                "status": status, "latency_s": latency, "pred": json.dumps(points, ensure_ascii=False),
                **metrics,
            })
            print(f"  [{i}/{len(df)}] id={r.feedback_id:3d} gold={len(r.gold)} "
                  f"pred={len(points)} {status:14s} {latency:5.1f}s")

    results = pd.DataFrame(rows)
    out_csv = DATA / "decompose_ollama_results.csv"
    results.to_csv(out_csv, index=False)

    # แยกไฟล์ผลจริงต่อ variant ให้เป็น input ของ pipeline ขั้นถัดไปได้เลย (schema
    # เดียวกับ dataset.csv: feedback_id + points_json) โดยไม่ต้องพึ่งเฉลย —
    # นี่คือ atomic points ที่ Mistral ตัดสินใจเองจริง ๆ ไม่ใช่ของที่คนแก้ไว้แล้ว
    raw_lookup = df.set_index("feedback_id").raw_text
    for variant in results.variant.unique():
        sub = results[results.variant == variant]
        points_out = pd.DataFrame({
            "feedback_id": sub.feedback_id.values,
            "raw_text": raw_lookup.loc[sub.feedback_id].values,
            "points_json": sub.pred.values,
            "n_points": sub.pred_n.values,
            "status": sub.status.values,
        })
        points_path = DATA / f"decomposed_{variant}.csv"
        points_out.to_csv(points_path, index=False)
        print(f"บันทึก atomic points จริงของ prompt {variant} ที่ {points_path}")

    print(f"\n{'='*74}\nสรุปผล\n{'='*74}")
    summary = results.groupby("variant").agg(
        n=("feedback_id", "count"), pred_points=("pred_n", "mean"),
        count_exact=("count_exact", "mean"), over_split=("over_split", "mean"),
        under_split=("under_split", "mean"), coverage=("coverage", "mean"),
        fidelity=("fidelity", "mean"), parse_error=("status", lambda s: (s == "parse_error").mean()),
        latency_s=("latency_s", "mean"),
    ).round(3)
    print(f"เฉลย: {results.gold_n.mean():.2f} points เฉลี่ย (n=111 ที่แท้จริงคือ 1.31)\n")
    print(summary.to_string())

    print("\nแยกตาม single-issue vs multi-issue")
    for is_multi, label in [(False, "single-issue (เฉลย 1 point)"), (True, "multi-issue (เฉลย >1)")]:
        sub = results[results.is_multi == is_multi]
        if sub.empty:
            continue
        print(f"\n{label}  n={sub.feedback_id.nunique()} responses")
        print(sub.groupby("variant").agg(
            pred_points=("pred_n", "mean"), count_exact=("count_exact", "mean"),
            over_split=("over_split", "mean"), under_split=("under_split", "mean"),
        ).round(3).to_string())

    print(f"\nบันทึกตารางที่ {out_csv}")


if __name__ == "__main__":
    main()
