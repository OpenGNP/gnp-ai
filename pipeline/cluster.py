"""Group atomic points into canonical topics.

Every modelling decision here was measured in scripts/cluster_points.py and is
imported from it: the BERTopic + HDBSCAN + UMAP configuration (`fit`), the
threshold-based noise reassignment (`reduce_outliers`), the re-clustering of
oversized topics (`split_large_topics`), and the fallback topic labelling
(`top_words`). This module only maps the result onto database rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from scripts.cluster_points import fit, reduce_outliers, split_large_topics, top_words

from . import config
from .db import ClusterablePoint, as_array

# UMAP reduces to 5 components; below roughly this many points the projection is
# meaningless and HDBSCAN has nothing to find.
MIN_POINTS = 10


@dataclass
class TopicDraft:
    key: str
    name: str
    keywords: str
    centroid: np.ndarray
    members: list[tuple[int, float]] = field(default_factory=list)
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    counts: dict[str, int] = field(default_factory=dict)


@dataclass
class ClusterResult:
    topics: list[TopicDraft]
    outliers: list[ClusterablePoint]
    min_cluster_size: int
    split_large: int


def _counts(members: list[ClusterablePoint]) -> dict[str, int]:
    tally = {"total": len(members), "positive": 0, "neutral": 0, "negative": 0, "severe": 0}
    for point in members:
        if point.sentiment_label in ("positive", "neutral", "negative"):
            tally[point.sentiment_label] += 1
        if point.is_severe:
            tally["severe"] += 1
    return tally


def cluster(points: list[ClusterablePoint]) -> ClusterResult:
    if len(points) < MIN_POINTS:
        raise SystemExit(
            f"only {len(points)} embedded points — need at least {MIN_POINTS} to cluster. "
            "Ingest more submissions, or run with --backfill to bring the existing points in."
        )

    docs = [p.text for p in points]
    embeddings = np.stack([as_array(p.embedding) for p in points])
    n_docs = len(docs)

    min_cluster_size = max(2, round(n_docs * config.MIN_CLUSTER_SIZE_PCT / 100))
    split_large = max(3, round(n_docs * config.SPLIT_LARGE_PCT / 100))

    model, labels = fit(min_cluster_size, docs, embeddings)
    labels = reduce_outliers(model, docs, labels, embeddings, config.OUTLIER_THRESHOLD)

    frame = pd.DataFrame({"text": docs, "topic": labels})
    sub_topic, sub_words = split_large_topics(model, frame, embeddings, split_large)
    keys = sub_topic.tolist()

    topics: list[TopicDraft] = []
    outliers: list[ClusterablePoint] = []

    for key in sorted(set(keys)):
        idx = [i for i, k in enumerate(keys) if k == key]
        members = [points[i] for i in idx]

        if key == "-1":
            outliers.extend(members)
            continue

        words = sub_words.get(key)
        if not words:
            base = int(str(key).split(".")[0])
            words = [word for word, _ in model.get_topic(base)][:10]
        if not words:
            words = top_words([points[i].text for i in idx], k=10)

        member_vectors = embeddings[idx]
        centroid = member_vectors.mean(axis=0)
        norm = np.linalg.norm(centroid)
        unit = centroid / norm if norm else centroid

        confidences = member_vectors @ unit
        member_norms = np.linalg.norm(member_vectors, axis=1)
        confidences = np.divide(
            confidences, member_norms, out=np.zeros_like(confidences), where=member_norms > 0
        )

        created = [m.created_at for m in members]
        topics.append(
            TopicDraft(
                key=str(key),
                name=", ".join(words[:3])[:255],
                keywords=", ".join(words[:10]),
                centroid=centroid.astype(np.float32),
                members=[(points[i].id, float(c)) for i, c in zip(idx, confidences)],
                first_seen=min(created),
                last_seen=max(created),
                counts=_counts(members),
            )
        )

    return ClusterResult(
        topics=topics,
        outliers=outliers,
        min_cluster_size=min_cluster_size,
        split_large=split_large,
    )
