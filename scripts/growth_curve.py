"""Does clustering quality fall as the corpus grows?

The pipeline re-clusters a whole form from scratch on every run, and
min_cluster_size scales with the corpus (config.MIN_CLUSTER_SIZE_PCT). Both
mean "the same survey, twice the responses" is not the same clustering problem
twice — it is a different one. This measures that directly: take one form,
subsample it at increasing sizes, cluster each subsample with the production
parameters, and report the same quality numbers at each size.

kNN purity is the headline. It asks the question the dashboard's reader asks
visually — "do a point's nearest neighbours share its topic?" — and unlike
silhouette it makes no assumption that a topic is a round blob.

    python -m scripts.growth_curve --form-id 3
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

from pipeline import cluster as cluster_mod
from pipeline import db

SEED = 42
FRACTIONS = (0.15, 0.3, 0.5, 0.75, 1.0)


def knn_purity(embeddings: np.ndarray, labels: np.ndarray, k: int = 5) -> float:
    """Share of each point's k nearest neighbours that carry its own topic.

    Outliers are excluded: they have no topic to agree with.
    """
    from sklearn.neighbors import NearestNeighbors

    mask = labels != -1
    X, y = embeddings[mask], labels[mask]
    if len(X) <= k + 1:
        return float("nan")

    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(X)
    _, idx = nn.kneighbors(X)
    # column 0 is the point itself
    return float(np.mean([np.mean(y[idx[i][1:]] == y[i]) for i in range(len(X))]))


def measure(points: list) -> dict:
    result = cluster_mod.cluster(points)

    topic_of = {}
    for i, topic in enumerate(result.topics):
        for point_id, _ in topic.members:
            topic_of[point_id] = i

    embeddings = np.stack([db.as_array(p.embedding) for p in points]).astype(np.float64)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    labels = np.array([topic_of.get(p.id, -1) for p in points])

    sizes = [len(t.members) for t in result.topics]
    return {
        "n": len(points),
        "mcs": result.min_cluster_size,
        "topics": len(result.topics),
        "outlier_rate": len(result.outliers) / len(points),
        "knn_purity": knn_purity(embeddings, labels),
        "largest_share": max(sizes) / len(points) if sizes else 0.0,
        "median_size": float(np.median(sizes)) if sizes else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--form-id", type=int, required=True)
    args = ap.parse_args()

    conn = db.connect()
    points = db.clusterable_points(conn, args.form_id)
    print(f"form {args.form_id}: {len(points)} embedded points\n")

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(points))

    print(f"{'n':>6}  {'mcs':>4}  {'topics':>6}  {'outlier':>8}  "
          f"{'kNN purity':>10}  {'largest':>8}  {'med size':>8}")
    for fraction in FRACTIONS:
        take = max(cluster_mod.MIN_POINTS, int(len(points) * fraction))
        if take > len(points):
            continue
        sample = [points[i] for i in order[:take]]
        row = measure(sample)
        print(f"{row['n']:>6}  {row['mcs']:>4}  {row['topics']:>6}  "
              f"{row['outlier_rate']:>7.1%}  {row['knn_purity']:>9.1%}  "
              f"{row['largest_share']:>7.1%}  {row['median_size']:>8.0f}")


if __name__ == "__main__":
    main()
