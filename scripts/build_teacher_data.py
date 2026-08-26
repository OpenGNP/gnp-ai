"""สร้างข้อมูล teacher ด้วย Gemini สำหรับ feedback ทั้ง 111 ชุด (ขั้นที่ 1)

prompt คัดลอกมาจาก gemini2.5flash_feedback_decomposer.ipynb แบบคำต่อคำ เพราะกฎ
GROUP LOGICALLY และ PRIORITIZE CONCISENESS ในนั้นคือสิ่งที่ทำให้ได้พฤติกรรม
2.35 points ต่อ response ซึ่งเป็นพฤติกรรมที่ต้องการสอน student

`has_multiple_points` ไม่ถูกอ่านในไฟล์นี้เลย โมเดลต้องตัดสินใจเองว่าจะซอยกี่จุด
เหมือนตอนใช้งานจริงที่ไม่มีใคร annotate ให้ คอลัมน์นั้นเก็บไว้ใช้ตอนวัดผลอย่างเดียว

เขียนผลลง JSONL ทีละแถวเพื่อให้รันซ้ำได้ ถ้า API ล้มกลางทางหรือโดน rate limit
รันใหม่แล้วมันจะข้ามชุดที่ทำไปแล้ว

ใช้:
    export GEMINI_API_KEY=...
    python scripts/build_teacher_data.py
    python scripts/build_teacher_data.py --limit 3    # ลองก่อน
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW_JSONL = DATA / "teacher_raw.jsonl"
OUT_CSV = DATA / "teacher_decomposed.csv"

MODEL = "gemini-2.5-flash"

PROMPT = """
        Task: Split the following student feedback into separate, autonomous topics.

        Instructions:
        1. GROUP LOGICALLY: Combine related points into single topics. If sentences discuss the same issue (problem + reason + example), keep them together as ONE topic.
        2. PRIORITIZE CONCISENESS: Only create separate topics when ideas are genuinely distinct and unrelated. Avoid over-fragmenting the feedback.
        3. MAKE EACH TOPIC STANDALONE: Each point must be fully understandable in isolation with complete context.
        4. NO PRONOUNS: Replace vague references (it, this, that) with specific nouns (e.g., "CS102", "the course", "AI tools").
        5. ADD CONTEXT WHERE NEEDED: If a point references something mentioned earlier, include that context explicitly.
        6. PRESERVE ORIGINAL TONE: Keep language the same as the original and first-person perspective. Don't over-formalize.
        7. KEEP EXAMPLES AS EXAMPLES: When students mention specific examples (like "100+ LeetCode" or "P.philix"), treat them as illustrations, not the main point.

        Example Input:
        "The curriculum needs change because of AI. Fundamental courses like CSC102 are good, no change needed. But we don't know much about infrastructure like Docker. So we should learn on our own."

        Example Output:
        [
        {{
        "topic": "AI-Driven Curriculum Update Need",
        "content": "The CS curriculum needs updating because the job market is changing rapidly due to AI, though fundamental courses like CSC102 should remain as they are still essential."
        }},
        {{
        "topic": "Infrastructure Knowledge Gap & Self-Learning",
        "content": "There's a gap in learning about infrastructure tools (like Docker as an example), so students need to take responsibility for learning these technologies on their own."
        }}
        ]

        Feedback to analyze:
        {text}

        Return ONLY the JSON array.
        - If the feedback contains only 1 clearly unified idea, return a single-item array.
        - Only split when topics are genuinely distinct. Do NOT force splitting.
        """


def get_model():
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        sys.exit(
            "ไม่พบ GEMINI_API_KEY\n"
            "  ขอ key ที่ https://aistudio.google.com/apikey\n"
            "  แล้วรัน: export GEMINI_API_KEY=...\n"
            "  (อย่าเขียน key ลงไฟล์ที่ commit)"
        )

    import google.generativeai as genai

    genai.configure(api_key=key)
    return genai, genai.GenerativeModel(MODEL)


def strip_fence(text: str) -> str:
    """ตัด markdown code fence ที่ Gemini ชอบใส่มา"""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def decompose(genai, model, text: str, retries: int = 3) -> tuple[list[str], str]:
    """คืน (points, status) — status บอกว่าได้ผลมายังไง ไม่กลบข้อผิดพลาด"""
    for attempt in range(retries):
        try:
            response = model.generate_content(
                PROMPT.format(text=text),
                generation_config=genai.types.GenerationConfig(
                    temperature=0, max_output_tokens=8192
                ),
            )
            parsed = json.loads(strip_fence(response.text))
            if not isinstance(parsed, list):
                parsed = [parsed]

            points = [
                str(item["content"]).strip() if isinstance(item, dict) else str(item).strip()
                for item in parsed
            ]
            points = [p for p in points if p]
            if points:
                return points, "ok"
            return [], "empty"

        except json.JSONDecodeError:
            if attempt == retries - 1:
                return [], "parse_error"
            time.sleep(2)
        except Exception as exc:
            if attempt == retries - 1:
                return [], f"api_error: {type(exc).__name__}"
            time.sleep(2 * (attempt + 1))

    return [], "failed"


def load_done() -> dict[int, dict]:
    """อ่านผลที่มีอยู่ แถวหลังทับแถวก่อนถ้า feedback_id ซ้ำ (กรณี retry)"""
    if not RAW_JSONL.exists():
        return {}
    done = {}
    for line in RAW_JSONL.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            done[row["feedback_id"]] = row
    return done


def succeeded_ids(done: dict[int, dict]) -> set[int]:
    """เฉพาะ id ที่สำเร็จจริงเท่านั้นที่นับว่าเสร็จ

    แถวที่ล้มต้องถูกลองใหม่ตอนรันซ้ำ ไม่ใช่ถูกข้าม ไม่งั้นข้อผิดพลาดชั่วคราว
    อย่าง rate limit จะกลายเป็นข้อมูลหายถาวรโดยไม่มีอะไรเตือน
    """
    return {fid for fid, row in done.items() if row["status"] == "ok"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="ทำแค่ N ชุดแรกที่ยังไม่ได้ทำ")
    ap.add_argument("--sleep", type=float, default=1.0, help="หน่วงระหว่างเรียก API (วินาที)")
    args = ap.parse_args()

    df = pd.read_csv(DATA / "GNP_Data.csv")
    # อ่านแค่ 2 คอลัมน์นี้ — has_multiple_points ห้ามหลุดเข้ามาเป็น input
    df = df[["feedback_id", "raw_text"]].dropna()
    print(f"feedback ทั้งหมด {len(df)} ชุด")

    done = load_done()
    ok_ids = succeeded_ids(done)
    todo = df[~df.feedback_id.isin(ok_ids)]
    if args.limit:
        todo = todo.head(args.limit)

    retrying = len(done) - len(ok_ids)
    print(f"สำเร็จแล้ว {len(ok_ids)} · เหลือ {len(todo)}" + (f" (ลองใหม่ {retrying})" if retrying else ""))

    if len(todo):
        genai, model = get_model()
        DATA.mkdir(exist_ok=True)

        with RAW_JSONL.open("a") as fh:
            for i, row in enumerate(todo.itertuples(), 1):
                points, status = decompose(genai, model, str(row.raw_text))
                record = {
                    "feedback_id": int(row.feedback_id),
                    "raw_text": str(row.raw_text),
                    "points": points,
                    "status": status,
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                fh.flush()

                mark = "ok" if status == "ok" else f"!! {status}"
                print(f"  [{i}/{len(todo)}] id={row.feedback_id} points={len(points)} {mark}")
                time.sleep(args.sleep)

        done = load_done()

    # ---- รวมเป็น CSV ----
    rows = [
        {
            "feedback_id": r["feedback_id"],
            "raw_text": r["raw_text"],
            "points_json": json.dumps(r["points"], ensure_ascii=False),
            "n_points": len(r["points"]),
            "status": r["status"],
        }
        for r in done.values()
    ]
    out = pd.DataFrame(rows).sort_values("feedback_id")
    out.to_csv(OUT_CSV, index=False)

    ok = out[out.status == "ok"]
    print(f"\nเขียน {OUT_CSV} ({len(out)} แถว)")
    print(f"  สำเร็จ {len(ok)} · ล้มเหลว {len(out) - len(ok)}")
    if len(ok):
        print(f"  points เฉลี่ย {ok.n_points.mean():.2f} (อ้างอิง Gemini เดิม = 2.35)")
        print(f"  การกระจาย: {ok.n_points.value_counts().sort_index().to_dict()}")

    failed = out[out.status != "ok"]
    if len(failed):
        print(f"\n  ล้มเหลว: {failed.status.value_counts().to_dict()}")
        print("  รันสคริปต์ซ้ำได้เลย จะลองเฉพาะแถวที่ล้มให้อัตโนมัติ")


if __name__ == "__main__":
    main()
