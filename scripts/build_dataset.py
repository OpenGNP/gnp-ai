"""สร้าง dataset สำหรับ decomposition จากเฉลยที่มีอยู่แล้ว (ขั้นที่ 1)

ไม่เรียก API ใด ๆ เพราะเฉลยครบทั้ง 111 ชุดมีอยู่ในเครื่องแล้ว
`feedback_expanded_with_gemini.xlsx` มี 2 sheet:

    Gemini  ผลจาก Gemini API  148 แถว  ครบ 111 responses
    Me      ฉบับที่คนแก้แล้ว   145 แถว  ครบ 111 responses

ใช้ sheet `Me` เป็น target เพราะเป็นเฉลยที่คนตรวจแล้ว ซึ่งเหนือกว่า output ของ
teacher โดยตรง ส่วน sheet `Gemini` เก็บไว้เทียบเพื่อดูว่า teacher ผิดตรงไหน
(ต่างกันแค่ 3 จาก 111 และทั้งสามเป็นกรณีที่ Gemini ซอย response ที่มีประเด็นเดียว)

`has_multiple_points` ติดมาด้วยแต่ใช้ตอนวัดผลอย่างเดียว ห้ามเข้าไปเป็น input
ของโมเดล — ของจริงไม่มีใคร annotate ให้

ใช้:
    python scripts/build_dataset.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
XLSX = DATA / "feedback_expanded_with_gemini.xlsx"
SOURCE = DATA / "GNP_Data.csv"
OUT = DATA / "dataset.csv"

N_FOLDS = 5
SEED = 42


def points_by_response(sheet: pd.DataFrame) -> pd.Series:
    """รวม sub_point_text ของแต่ละ response เป็น list เรียงตาม sub_point_number"""
    ordered = sheet.sort_values(["original_feedback_id", "sub_point_number"])
    return (
        ordered.groupby("original_feedback_id")["sub_point_text"]
        .apply(lambda s: [str(x).strip() for x in s if str(x).strip()])
    )


def assign_folds(df: pd.DataFrame) -> pd.Series:
    """แบ่งเป็น k folds แบบ stratified ตาม `is_multi`

    ใช้ k-fold แทนการแบ่ง train/test ครั้งเดียว เพราะ multi-issue มีแค่ 23 ชุด
    จาก 111 การกันไว้เป็น test ครั้งเดียวจะเหลือ multi-issue ราว 4 ชุด
    ซึ่งน้อยเกินกว่าจะสรุปอะไรเกี่ยวกับความสามารถในการซอยได้

    ด้วย k-fold โมเดลที่ต้องเทรนจะถูกประเมินครบทั้ง 111 ชุดโดยไม่มี leakage
    (เทรน 4 folds ทดสอบ 1 fold วนจนครบ) ส่วนวิธีที่ใช้ prompt อย่างเดียว
    ไม่มีการเทรน จึงประเมินบนทั้ง 111 ชุดได้ตรง ๆ

    stratify ตาม `is_multi` ที่คำนวณจากเฉลย ไม่ใช่ตาม `has_multiple_points`
    เพราะเฉลยคือคำตอบที่ถูกจริง ส่วน label ไม่ตรงกับเฉลย 7 ชุดและว่างอีก 2 ชุด
    """
    fold = pd.Series(-1, index=df.index, name="fold")

    for _, group in df.groupby("is_multi"):
        shuffled = group.sample(frac=1.0, random_state=SEED).index
        for position, idx in enumerate(shuffled):
            fold.loc[idx] = position % N_FOLDS

    return fold


def main() -> None:
    xl = pd.ExcelFile(XLSX)
    gold = points_by_response(xl.parse("Me")).rename("points")
    teacher = points_by_response(xl.parse("Gemini")).rename("gemini_points")

    source = pd.read_csv(SOURCE).set_index("feedback_id")

    df = pd.concat([gold, teacher], axis=1).join(
        source[["raw_text", "has_multiple_points"]]
    )
    df.index.name = "feedback_id"

    # ข้อความใน xlsx ต้องตรงกับต้นฉบับ ไม่งั้นแปลว่าจับคู่ผิดแถว
    assert df.raw_text.notna().all(), "มี response ที่หา raw_text ไม่เจอ"
    assert df.points.map(len).gt(0).all(), "มี response ที่ไม่มี point เลย"

    df["n_points"] = df.points.map(len)
    df["is_multi"] = df.n_points > 1
    df["fold"] = assign_folds(df)

    out = pd.DataFrame(
        {
            "feedback_id": df.index,
            "raw_text": df.raw_text.values,
            "points_json": [json.dumps(p, ensure_ascii=False) for p in df.points],
            "n_points": df.n_points.values,
            "is_multi": df.is_multi.values,
            "gemini_points_json": [
                json.dumps(p, ensure_ascii=False) for p in df.gemini_points
            ],
            "gemini_n_points": df.gemini_points.map(len).values,
            "has_multiple_points": df.has_multiple_points.values,
            "fold": df.fold.values,
        }
    )
    out.to_csv(OUT, index=False)

    # ---- รายงาน ----
    print(f"เขียน {OUT}  ({len(out)} responses)\n")

    print("จำนวน points ของเฉลย (human gold)")
    print(f"  เฉลี่ยรวม {out.n_points.mean():.2f}   "
          f"การกระจาย {out.n_points.value_counts().sort_index().to_dict()}")
    print(f"  single-issue {(~out.is_multi).sum()} ชุด · multi-issue {out.is_multi.sum()} ชุด")

    print("\nเฉลย เทียบกับ label has_multiple_points")
    cross = pd.crosstab(
        out.has_multiple_points.fillna("(ว่าง)"),
        out.is_multi.map({False: "gold=1 point", True: "gold>1 point"}),
    )
    print(cross.to_string())
    mismatch = (
        ((out.has_multiple_points == "No") & out.is_multi)
        | ((out.has_multiple_points == "Yes") & ~out.is_multi)
    ).sum()
    print(f"  ไม่ตรงกัน {mismatch} ชุด + ว่าง {out.has_multiple_points.isna().sum()} ชุด")
    print("  ใช้เฉลยเป็นหลัก เพราะเป็นคำตอบที่คาดหวังจริง ส่วน label เป็นคำอธิบายคร่าว ๆ")

    disagree = out[out.n_points != out.gemini_n_points]
    print(f"\nteacher (Gemini) ไม่ตรงกับคน {len(disagree)}/{len(out)} responses")
    for _, r in disagree.iterrows():
        print(f"  id={r.feedback_id} gemini={r.gemini_n_points} human={r.n_points}")

    print(f"\nการแบ่ง {N_FOLDS} folds (คอลัมน์ is_multi)")
    print(out.groupby(["fold", "is_multi"]).size().unstack(fill_value=0).to_string())

    assert (out.fold >= 0).all(), "มี response ที่ไม่ได้ fold"
    print("\nวิธีใช้")
    print("  วิธีที่ใช้ prompt อย่างเดียว  ประเมินบนทั้ง 111 ชุด (ไม่มีการเทรน ไม่มี leakage)")
    print(f"  วิธีที่ต้องเทรน            เทรน {N_FOLDS-1} folds ทดสอบ 1 fold วนให้ครบ {N_FOLDS} รอบ")


if __name__ == "__main__":
    main()
