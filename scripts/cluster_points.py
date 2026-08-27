"""จับกลุ่ม atomic points เป็น topic ด้วย sentence transformer

ใช้ atomic points จากเฉลย ไม่ใช่ output จากโมเดล เพื่อแยกความเสี่ยงสองอย่าง
ออกจากกัน ถ้าใช้ output ของโมเดลที่ยังรวบจุดอยู่ แล้ว clustering ออกมาไม่ดี
จะแยกไม่ออกว่าปัญหาอยู่ที่ clustering หรือที่ input เฉลยคือ input ที่ดีที่สุด
เท่าที่มี จึงตอบได้ตรง ๆ ว่าตัว clustering ใช้งานได้หรือไม่

ใช้:
    python scripts/cluster_points.py
    python scripts/cluster_points.py --top-n 2
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import warnings
from collections import Counter
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
SEED = 42


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
def build_configs(n_docs: int) -> list[dict]:
    configs = []
    for min_cluster_size in (2, 3, 4, 5, 8, 10):
        if min_cluster_size < n_docs // 2:
            configs.append({"algo": "hdbscan", "min_cluster_size": min_cluster_size})
    for k in (5, 8, 10, 12, 15, 20):
        if k < n_docs // 2:
            configs.append({"algo": "kmeans", "n_clusters": k})
    return configs


def fit(cfg: dict, docs: list[str], embeddings: np.ndarray):
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer
    from hdbscan import HDBSCAN
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    n_docs = len(docs)
    # default ของ UMAP คือ 15 ซึ่งเกินจำนวนเอกสารบน corpus เล็กแล้ว library
    # จะ error แทนที่จะทำงานแย่ลงเฉย ๆ
    n_neighbors = max(2, min(15, n_docs - 1))

    cluster_model = (
        HDBSCAN(
            min_cluster_size=cfg["min_cluster_size"],
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )
        if cfg["algo"] == "hdbscan"
        else KMeans(n_clusters=cfg["n_clusters"], random_state=SEED, n_init=10)
    )

    model = BERTopic(
        umap_model=UMAP(
            n_neighbors=n_neighbors, n_components=5, min_dist=0.0,
            metric="cosine", random_state=SEED,
        ),
        hdbscan_model=cluster_model,
        # ตัด stopword สำคัญมากกับ corpus เล็ก ไม่งั้น c-TF-IDF จะดึง function word
        # ขึ้นมาเป็นคำเด่น แล้วทุก topic จะได้ชื่อคล้ายกันหมด
        vectorizer_model=CountVectorizer(stop_words="english", min_df=1, ngram_range=(1, 2)),
        ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
        calculate_probabilities=False,
        verbose=False,
    )
    labels, _ = model.fit_transform(docs, embeddings)
    return model, [int(x) for x in labels]


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
    ap.add_argument("--top-n", type=int, default=1, help="จำนวน config ที่จะแสดงรายละเอียด")
    ap.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    args = ap.parse_args()

    # 1. แตกเฉลยออกเป็น atomic points รายจุด
    df = pd.read_csv(DATA / "dataset.csv")
    rows = [
        {"feedback_id": r.feedback_id, "text": p}
        for r in df.itertuples()
        for p in json.loads(r.points_json)
    ]
    points = pd.DataFrame(rows)
    docs = points.text.tolist()

    print(f"atomic points {len(points)} จุด จาก {points.feedback_id.nunique()} responses")
    print(f"ความยาวเฉลี่ย {points.text.str.split().str.len().mean():.0f} คำ\n")

    # 2. embedding
    from sentence_transformers import SentenceTransformer

    print(f"embedding ด้วย {args.embedding_model} ...")
    encoder = SentenceTransformer(args.embedding_model)
    embeddings = encoder.encode(
        docs, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    ).astype(np.float32)

    # 3. จับกลุ่มหลาย config
    configs = build_configs(len(docs))
    print(f"ทดลอง {len(configs)} configuration\n")

    results = []
    for i, cfg in enumerate(configs, 1):
        name = (
            f"hdbscan/mcs{cfg['min_cluster_size']}"
            if cfg["algo"] == "hdbscan"
            else f"kmeans/k{cfg['n_clusters']}"
        )
        try:
            model, labels = fit(cfg, docs, embeddings)
            row = {"config": name, **cfg, **evaluate(model, labels, docs),
                   "_model": model, "_labels": labels}
            print(f"  [{i}/{len(configs)}] {name:18s} topics={row['n_topics']:3d}  "
                  f"outlier={row['outlier_rate']:5.1%}  npmi={row['npmi']:+.3f}  "
                  f"diversity={row['diversity']:.2f}")
        except Exception as exc:
            row = {"config": name, **cfg, "error": f"{type(exc).__name__}: {exc}"}
            print(f"  [{i}/{len(configs)}] {name:18s} ล้มเหลว {row['error'][:50]}")
        results.append(row)

    table = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in results])
    out_csv = DATA / "cluster_results.csv"
    table.to_csv(out_csv, index=False)

    ok = table[table.get("error").isna()] if "error" in table else table
    if ok.empty:
        print("\nทุก configuration ล้มเหลว")
        return

    print("\n" + "=" * 74)
    print("HDBSCAN เทียบ KMeans")
    print("=" * 74)
    for algo in ("hdbscan", "kmeans"):
        sub = ok[ok.algo == algo]
        if not sub.empty:
            print(f"  {algo:8s} outlier ต่ำสุด {sub.outlier_rate.min():5.1%} "
                  f"กลาง {sub.outlier_rate.median():5.1%}  ·  npmi สูงสุด {sub.npmi.max():+.3f}")

    # ใช้ได้จริงต้องทั้งเก็บเอกสารไว้ได้และ topic มีความหมาย
    usable = ok[(ok.outlier_rate <= 0.30) & (ok.n_topics >= 3)]
    if usable.empty:
        print("\nไม่มี configuration ไหนที่เก็บเอกสารไว้ได้เกิน 70% พร้อมกับมี topic อย่างน้อย 3 กลุ่ม")
        print("แปลว่า HDBSCAN ไม่เหมาะกับ corpus ขนาดนี้ ควรใช้ KMeans")
        usable = ok[ok.n_topics >= 3]
        if usable.empty:
            return

    best = usable.sort_values("npmi", ascending=False).head(args.top_n)
    by_name = {r["config"]: r for r in results if "_model" in r}

    for cfg_name in best.config:
        row = by_name[cfg_name]
        model, labels = row["_model"], row["_labels"]
        points["topic"] = labels

        print("\n" + "=" * 74)
        print(f"รายละเอียดของ {cfg_name}")
        print(f"  topic {row['n_topics']} กลุ่ม · outlier {row['outlier_rate']:.1%} · "
              f"npmi {row['npmi']:+.3f} · diversity {row['diversity']:.2f}")
        print("=" * 74)

        assigned = points[points.topic != -1]
        summary = (
            assigned.groupby("topic")
            .agg(mentions=("text", "size"), reach=("feedback_id", "nunique"))
            .sort_values("reach", ascending=False)
        )

        for topic_id, stat in summary.iterrows():
            words = [w for w, _ in model.get_topic(topic_id)][:5]
            inflation = stat.mentions / stat.reach
            print(f"\n  [{', '.join(words)}]")
            print(f"     mentions {stat.mentions:2d} จุด · reach {stat.reach:2d} คน"
                  + (f"  (เฟ้อ {inflation:.2f}x)" if inflation > 1.01 else ""))
            for text in assigned[assigned.topic == topic_id].text.head(2):
                print(f"     - {text[:95]}")

        if row["outlier_rate"] > 0:
            n_out = (points.topic == -1).sum()
            print(f"\n  ไม่เข้ากลุ่มไหนเลย {n_out} จุด")

        total_m, total_r = summary.mentions.sum(), assigned.feedback_id.nunique()
        print(f"\n  รวม mentions {total_m} จุด แต่ reach {total_r} คน")
        print("  ตัวเลขสองอันนี้ต่างกันเพราะคนเดียวเขียนได้หลายจุด")
        print("  priority ต้องนับ reach ไม่ใช่ mentions ไม่งั้นคนเขียนยาวคนเดียวจะมีน้ำหนักเกินจริง")

    print(f"\nบันทึกตารางที่ {out_csv}")


if __name__ == "__main__":
    main()
