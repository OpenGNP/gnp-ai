"""จับกลุ่ม atomic points เป็น topic ด้วย sentence transformer + HDBSCAN

ใช้ atomic points จากเฉลย ไม่ใช่ output จากโมเดล เพื่อแยกความเสี่ยงสองอย่าง
ออกจากกัน ถ้าใช้ output ของโมเดลที่ยังรวบจุดอยู่ แล้ว clustering ออกมาไม่ดี
จะแยกไม่ออกว่าปัญหาอยู่ที่ clustering หรือที่ input เฉลยคือ input ที่ดีที่สุด
เท่าที่มี จึงตอบได้ตรง ๆ ว่าตัว clustering ใช้งานได้หรือไม่

ใช้:
    python scripts/cluster_points.py
    python scripts/cluster_points.py --examples 2
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import warnings
from collections import Counter
from datetime import datetime
from itertools import combinations
from pathlib import Path

# anaconda บน macOS มี libomp คนละตัวกับที่ numba/hdbscan โหลด ต้องตั้งก่อน import
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REPORTS = DATA / "reports"
SEED = 42


class Tee:
    """เขียนออกทั้งหน้าจอและไฟล์พร้อมกัน ใช้แทน stdout ตลอดการรัน

    ทำแบบนี้แทนการรัน `python ... > out.txt` เอง เพื่อให้ได้ผลทั้งสองทาง
    ในคำสั่งเดียว ไม่ต้องจำ redirect ทุกครั้ง
    """

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text: str) -> None:
        for s in self.streams:
            s.write(text)

    def flush(self) -> None:
        for s in self.streams:
            s.flush()

    def isatty(self) -> bool:
        # library ที่เช็ค isatty() ก่อน print สี/progress bar (เช่น transformers)
        # ต้องได้คำตอบจาก terminal จริง ไม่ใช่จากไฟล์ที่แนบมาด้วย
        return self.streams[0].isatty()


def report_filename(args, n_docs: int) -> str:
    embed = args.embedding_model.split("/")[-1]
    mcs_part = "-".join(map(str, args.mcs)) if args.mcs else "sweep"

    flags = []
    if args.reduce_outliers:
        flags.append(f"reduce{args.outlier_threshold}")
    if args.split_large:
        flags.append(f"split{args.split_large}")
    flag_part = "_" + "_".join(flags) if flags else ""

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"cluster_{embed}_n{n_docs}_mcs{mcs_part}{flag_part}_{stamp}.txt"


# ── metrics ────────────────────────────────────────────────────────────
_TOKEN = re.compile(r"[a-z][a-z'-]{1,}")


def npmi_coherence(topic_words: list[list[str]], docs: list[str], top_k: int = 10) -> float:
    """NPMI เฉลี่ยของคู่คำเด่นในแต่ละ topic

    ช่วง [-1, 1] ยิ่งสูงยิ่งดี เกิน 0.1 ถือว่าคนอ่านแล้วรู้สึกว่าเป็นกลุ่มเดียวกัน
    ต่ำกว่า 0.05 แปลว่าคำที่เจอแทบไม่ต่างจากการสุ่ม
    """
    doc_tokens = [set(_TOKEN.findall(d.lower())) for d in docs]
    n_docs = len(doc_tokens)
    if not n_docs:
        return float("nan")

    df: Counter[str] = Counter()
    for toks in doc_tokens:
        df.update(toks)

    scores = []
    for words in topic_words:
        words = [w for w in words[:top_k] if df[w] > 0]
        if len(words) < 2:
            continue
        pair_scores = []
        for a, b in combinations(words, 2):
            co = sum(1 for toks in doc_tokens if a in toks and b in toks)
            if co == 0:
                pair_scores.append(-1.0)
                continue
            p_a, p_b, p_ab = df[a] / n_docs, df[b] / n_docs, co / n_docs
            pair_scores.append(math.log(p_ab / (p_a * p_b)) / -math.log(p_ab))
        if pair_scores:
            scores.append(float(np.mean(pair_scores)))

    return float(np.mean(scores)) if scores else float("nan")


def topic_diversity(topic_words: list[list[str]], top_k: int = 10) -> float:
    """สัดส่วนคำเด่นที่ไม่ซ้ำกันข้าม topic

    ค่าต่ำแปลว่า topic ต่างกันแค่ลำดับคำ ซึ่งบน dashboard จะออกมาเป็นการ์ด
    หลายใบที่พูดเรื่องเดียวกัน
    """
    all_words = [w for words in topic_words for w in words[:top_k]]
    if not all_words:
        return float("nan")
    return len(set(all_words)) / len(all_words)


# ── clustering ─────────────────────────────────────────────────────────
def build_configs(n_docs: int) -> list[int]:
    """ค่า min_cluster_size ที่จะลอง — ต้องน้อยกว่าครึ่งของจำนวนเอกสาร"""
    return [m for m in (2, 3, 4, 5, 6, 7, 8, 10) if m < n_docs // 2]


def _topic_representation():
    """ตัวสกัดคำเด่นของ topic

    ตัด stopword สำคัญมากกับ corpus เล็ก ไม่งั้น c-TF-IDF จะดึง function word
    ขึ้นมาเป็นคำเด่น แล้วทุก topic จะได้ชื่อคล้ายกันหมด

    แยกออกมาเป็นฟังก์ชันเพราะต้องใช้ทั้งตอน fit และตอน update_topics —
    ถ้าไม่ส่งให้ update_topics มันจะกลับไปใช้ค่า default ที่ไม่ตัด stopword
    """
    from bertopic.vectorizers import ClassTfidfTransformer
    from sklearn.feature_extraction.text import CountVectorizer

    return (
        CountVectorizer(stop_words="english", min_df=1, ngram_range=(1, 2)),
        ClassTfidfTransformer(reduce_frequent_words=True),
    )


def fit(min_cluster_size: int, docs: list[str], embeddings: np.ndarray):
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP

    n_docs = len(docs)
    # default ของ UMAP คือ 15 ซึ่งเกินจำนวนเอกสารบน corpus เล็กแล้ว library
    # จะ error แทนที่จะทำงานแย่ลงเฉย ๆ
    n_neighbors = max(2, min(15, n_docs - 1))

    cluster_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    vectorizer, ctfidf = _topic_representation()
    model = BERTopic(
        umap_model=UMAP(
            n_neighbors=n_neighbors, n_components=5, min_dist=0.0,
            metric="cosine", random_state=SEED,
        ),
        hdbscan_model=cluster_model,
        vectorizer_model=vectorizer,
        ctfidf_model=ctfidf,
        calculate_probabilities=False,
        verbose=False,
    )
    labels, _ = model.fit_transform(docs, embeddings)
    return model, [int(x) for x in labels]


def reduce_outliers(model, docs, labels, embeddings, threshold=0.30):
    """ย้ายจุดที่ถูกทิ้งไปยัง topic ที่ embedding ใกล้ที่สุด เฉพาะที่ใกล้พอ

    HDBSCAN ทิ้งจุดที่อยู่ขอบความหนาแน่นเป็น noise แม้ว่าโดยความหมายแล้วจุดนั้น
    ชัดเจนว่าอยู่กลุ่มไหน เช่น "SIT staffs respond a bit slow" ถูกทิ้งทั้งที่มี
    กลุ่ม staff อยู่แล้ว การทิ้งแบบนี้ทำให้ topic frequency ต่ำกว่าความจริง
    ซึ่งกระทบ priority โดยตรง

    แต่บางจุดก็ไม่มีกลุ่มให้อยู่จริง ๆ เขียนเอง (ไม่ใช้ model.reduce_outliers)
    เพื่อกำหนด threshold เอง — วัดจากข้อมูลจริงแล้วว่า cosine similarity ต่อ
    centroid ที่ใกล้สุด ~0.30 คือจุดแบ่งที่กันเคสย้ายผิดกลุ่มได้ (เช่น
    "The AC is sometimes too loud" ที่คะแนน 0.298 เคยถูกย้ายเข้ากลุ่ม wifi ผิด ๆ)
    โดยยังคงเคสที่ควรย้ายจริงไว้ได้เกือบหมด ต่ำกว่านี้ปล่อยเป็น noise ต่อไป
    """
    labels = np.asarray(labels)
    topic_ids = sorted(t for t in set(labels) if t != -1)
    if not topic_ids:
        return labels.tolist()

    centroids = np.stack([embeddings[labels == t].mean(axis=0) for t in topic_ids])
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)

    new_labels = labels.copy()
    for i in np.where(labels == -1)[0]:
        sims = centroids @ embeddings[i]
        best = int(np.argmax(sims))
        if sims[best] >= threshold:
            new_labels[i] = topic_ids[best]

    new_labels = [int(x) for x in new_labels]
    # ต้องคำนวณ c-TF-IDF ใหม่ ไม่งั้นชื่อกลุ่มยังเป็นของสมาชิกชุดเดิมก่อนย้าย
    # และต้องส่ง vectorizer ตัวเดิมไปด้วย ไม่งั้นจะกลับไปใช้ค่า default ที่ไม่ตัด stopword
    vectorizer, ctfidf = _topic_representation()
    model.update_topics(docs, topics=new_labels, vectorizer_model=vectorizer, ctfidf_model=ctfidf)
    return new_labels


def top_words(docs: list[str], k: int = 5) -> list[str]:
    """คำ/วลีเด่นของ docs กลุ่มหนึ่ง ใช้ตอนตั้งชื่อ sub-topic ที่ไม่ได้อยู่ในโมเดลหลัก"""
    from sklearn.feature_extraction.text import CountVectorizer

    vec = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    try:
        counts = vec.fit_transform(docs)
    except ValueError:
        return []
    freq = np.asarray(counts.sum(axis=0)).ravel()
    vocab = vec.get_feature_names_out()
    order = np.argsort(-freq)[:k]
    return [vocab[i] for i in order]


def split_large_topics(model, points: pd.DataFrame, embeddings: np.ndarray, min_size: int):
    """จับกลุ่มซ้ำเฉพาะข้างในกลุ่มที่ใหญ่เกินไป เพื่อดึงรายละเอียดย่อยออกมา

    กลุ่มอย่าง "courses, curriculum" ใช้คำคล้ายกันจนรวมประเด็นย่อยที่ต่างกันจริง
    ไว้ด้วยกัน (ความเร็วสอน, เนื้อหาตกยุค, อยากได้ lab จริง ฯลฯ) การจับกลุ่มซ้ำ
    เฉพาะสมาชิกกลุ่มนั้นด้วย min_cluster_size เล็กลง จะดึงรายละเอียดที่ถูกกลืน
    ไว้ออกมาโดยไม่กระทบกลุ่มอื่นที่แยกดีอยู่แล้ว

    ใช้ UMAP embedding ที่คำนวณไว้แล้วจากรอบแรก (model.umap_model.embedding_)
    ไม่ต้องคำนวณใหม่ และเรียงตามลำดับเดียวกับ points เพราะเป็นอินพุตชุดเดียวกัน
    """
    from hdbscan import HDBSCAN

    reduced = model.umap_model.embedding_
    sub_topic = points["topic"].astype(str).copy()
    sub_words: dict[str, list[str]] = {}

    assigned = points[points.topic != -1]
    for topic_id, group in assigned.groupby("topic"):
        if len(group) < min_size:
            continue
        idx = group.index.to_numpy()
        sub_mcs = max(2, len(idx) // 6)
        sub_labels = HDBSCAN(min_cluster_size=sub_mcs, metric="euclidean").fit_predict(reduced[idx])

        for pos, lab in zip(idx, sub_labels):
            if lab != -1:
                sub_topic.loc[pos] = f"{topic_id}.{lab}"

        for lab in set(sub_labels):
            if lab == -1:
                continue
            key = f"{topic_id}.{lab}"
            sub_words[key] = top_words(points.loc[idx[sub_labels == lab], "text"].tolist())

    return sub_topic, sub_words


def evaluate(model, labels, docs) -> dict:
    topic_ids = sorted(t for t in set(labels) if t != -1)
    words = [[w for w, _ in model.get_topic(t)] for t in topic_ids]
    return {
        "n_topics": len(topic_ids),
        "outlier_rate": sum(1 for t in labels if t == -1) / len(labels),
        "npmi": npmi_coherence(words, docs),
        "diversity": topic_diversity(words),
    }


# ── main ───────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcs", type=int, nargs="+", default=None, metavar="N",
                    help="ค่า min_cluster_size ที่จะลอง เช่น --mcs 5 หรือ --mcs 3 5 8 "
                         "(ไม่ใส่ = ลองทุกค่า)")
    ap.add_argument("--examples", type=int, default=0, metavar="N",
                    help="แสดงข้อความจริงในแต่ละ topic กี่ข้อความ (ใส่ 99 = แสดงหมด)")
    ap.add_argument("--full-text", action="store_true",
                    help="แสดงข้อความตัวอย่างเต็ม ไม่ตัดที่ 88 ตัวอักษร")
    ap.add_argument("--reduce-outliers", action="store_true",
                    help="ย้ายจุดที่ HDBSCAN ทิ้งไปยัง topic ที่ใกล้ที่สุด แทนที่จะปล่อยทิ้ง "
                         "(เฉพาะที่ cosine similarity >= --outlier-threshold)")
    ap.add_argument("--outlier-threshold", type=float, default=0.30, metavar="X",
                    help="ค่า cosine similarity ขั้นต่ำที่จะย้าย outlier เข้ากลุ่ม "
                         "(ค่าเริ่มต้น 0.30 วัดจากข้อมูลจริงว่าแยกเคสย้ายถูก/ผิดได้ดี)")
    ap.add_argument("--split-large", type=int, default=None, metavar="N",
                    help="จับกลุ่มซ้ำเฉพาะกลุ่มที่มีสมาชิก >= N เพื่อดึงรายละเอียดย่อยออกมา "
                         "เช่น --split-large 15")
    ap.add_argument("--embedding-model", default="all-MiniLM-L6-v2",
                    help="โมเดล sentence-transformers ที่ใช้ทำ embedding")
    args = ap.parse_args()

    df = pd.read_csv(DATA / "dataset.csv")
    rows = [
        {"feedback_id": r.feedback_id, "text": p}
        for r in df.itertuples()
        for p in json.loads(r.points_json)
    ]
    points = pd.DataFrame(rows)
    docs = points.text.tolist()

    REPORTS.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS / report_filename(args, len(points))
    real_stdout = sys.stdout

    with open(report_path, "w") as fh:
        sys.stdout = Tee(real_stdout, fh)
        try:
            _run(args, points, docs)
        finally:
            sys.stdout = real_stdout

    print(f"บันทึกรายงานฉบับเต็มที่ {report_path}")


def _run(args, points, docs) -> None:

    print(f"atomic points {len(points)} จุด จาก {points.feedback_id.nunique()} responses")
    print(f"ความยาวเฉลี่ย {points.text.str.split().str.len().mean():.0f} คำ\n")

    # 2. embedding
    from sentence_transformers import SentenceTransformer

    print(f"embedding ด้วย {args.embedding_model} ...")
    encoder = SentenceTransformer(args.embedding_model)
    embeddings = encoder.encode(
        docs, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    ).astype(np.float32)

    # 3. จับกลุ่มด้วย min_cluster_size หลายค่า
    configs = args.mcs if args.mcs else build_configs(len(docs))
    too_big = [m for m in configs if m >= len(docs) // 2]
    if too_big:
        print(f"ข้าม mcs {too_big} เพราะมากกว่าครึ่งของจำนวนเอกสาร ({len(docs)} จุด)")
        configs = [m for m in configs if m not in too_big]
    if not configs:
        print("ไม่เหลือค่าให้ลอง")
        return

    mode = (f"ย้าย outlier เข้ากลุ่มใกล้สุด (threshold {args.outlier_threshold})"
            if args.reduce_outliers else "ปล่อย outlier ทิ้ง")
    if args.split_large:
        mode += f"  ·  แตกกลุ่มที่มี >= {args.split_large} จุด"
    print(f"ทดลอง min_cluster_size {configs}  ·  {mode}\n")

    results = []
    for mcs in configs:
        try:
            model, labels = fit(mcs, docs, embeddings)
            row = {"mcs": mcs, "outlier_before": sum(1 for t in labels if t == -1)}
            if args.reduce_outliers:
                labels = reduce_outliers(model, docs, labels, embeddings, args.outlier_threshold)
            row |= evaluate(model, labels, docs)
            results.append(row | {"_model": model, "_labels": labels})
        except Exception as exc:
            results.append({"mcs": mcs, "error": f"{type(exc).__name__}: {exc}"})

    table = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in results])
    out_csv = DATA / "cluster_results.csv"
    table.to_csv(out_csv, index=False)

    print("=" * 74)
    print("สรุปภาพรวม")
    print("=" * 74)
    print(f"{'mcs':>4}  {'topics':>6}  {'outlier':>8}  {'ไม่เข้ากลุ่ม':>10}  {'npmi':>7}  {'diversity':>9}")
    for r in results:
        if "error" in r:
            print(f"{r['mcs']:>4}  ล้มเหลว {r['error'][:45]}")
            continue
        n_out = sum(1 for t in r["_labels"] if t == -1)
        print(f"{r['mcs']:>4}  {r['n_topics']:>6}  {r['outlier_rate']:>7.1%}  "
              f"{n_out:>8} จุด  {r['npmi']:>+7.3f}  {r['diversity']:>9.2f}")

    # 4. รายละเอียดของทุกค่า
    for r in results:
        if "error" in r:
            continue
        model, labels = r["_model"], r["_labels"]
        points["topic"] = labels

        sub_words: dict[str, list[str]] = {}
        if args.split_large:
            sub_topic, sub_words = split_large_topics(model, points, embeddings, args.split_large)
            points["topic"] = sub_topic
            group_col = points["topic"]
            is_outlier = group_col == "-1"
        else:
            group_col = points["topic"]
            is_outlier = group_col == -1

        assigned = points[~is_outlier]

        summary = (
            assigned.groupby("topic")
            .agg(mentions=("text", "size"), reach=("feedback_id", "nunique"))
            .sort_values("reach", ascending=False)
        )

        print("\n" + "=" * 74)
        print(f"min_cluster_size = {r['mcs']}   ({r['n_topics']} topics · "
              f"outlier {r['outlier_rate']:.1%} · npmi {r['npmi']:+.3f})")
        if args.split_large:
            print(f"(แสดงแบบแตกกลุ่มใหญ่ — จำนวน topic ที่เห็นด้านล่างมากกว่า {r['n_topics']} ข้างต้น)")
        print("=" * 74)

        for topic_id, stat in summary.iterrows():
            if topic_id in sub_words:
                words = sub_words[topic_id]
            else:
                base_id = int(str(topic_id).split(".")[0])
                words = [w for w, _ in model.get_topic(base_id)][:5]
            inflation = stat.mentions / stat.reach
            flag = f"  เฟ้อ {inflation:.2f}x" if inflation > 1.01 else ""
            print(f"  {stat.mentions:>3} จุด /{stat.reach:>3} คน{flag:>12}  {', '.join(words)}")
            for text in assigned[assigned.topic == topic_id].text.head(args.examples):
                print(f"        - {text if args.full_text else text[:88]}")

        total_m, total_r = summary.mentions.sum(), assigned.feedback_id.nunique()
        outliers = points[is_outlier]
        median_reach = summary.reach.median()
        print(f"  {'-' * 70}")
        print(f"  รวม {total_m} จุด / {total_r} คน · ไม่เข้ากลุ่ม {len(outliers)} จุด · "
              f"reach กลางของกลุ่ม {median_reach:.0f} คน")

        if len(outliers):
            print(f"\n  [ไม่เข้ากลุ่มไหนเลย]")
            for text in outliers.text.head(args.examples):
                print(f"        - {text if args.full_text else text[:88]}")

    print(f"\nบันทึกตารางที่ {out_csv}")
    print("\nวิธีเลือก: npmi สูงอย่างเดียวไม่พอ เพราะกลุ่มยิ่งเล็กยิ่งได้ npmi สูงโดยธรรมชาติ")
    print("ให้ดูด้วยว่าชื่อกลุ่มอ่านแล้วแยกออกจากกันจริงไหม และ reach ต่อกลุ่มมากพอจะตัดสินใจได้ไหม")



if __name__ == "__main__":
    main()
