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
    member_texts: list[str] = field(default_factory=list)
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    counts: dict[str, int] = field(default_factory=dict)


@dataclass
class ClusterResult:
    topics: list[TopicDraft]
    outliers: list[ClusterablePoint]
    min_cluster_size: int


def _counts(members: list[ClusterablePoint]) -> dict[str, int]:
    tally = {"total": len(members), "positive": 0, "neutral": 0, "negative": 0, "violation": 0}
    for point in members:
        if point.sentiment_label in ("positive", "neutral", "negative"):
            tally[point.sentiment_label] += 1
        if point.is_violation:
            tally["violation"] += 1
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

    min_cluster_size = max(
        config.MIN_CLUSTER_POINTS, round(n_docs * config.MIN_CLUSTER_SIZE_PCT / 100)
    )

    model, labels = fit(min_cluster_size, docs, embeddings)
    labels = reduce_outliers(model, docs, labels, embeddings, config.OUTLIER_THRESHOLD)

    frame = pd.DataFrame({"text": docs, "topic": labels})
    sub_topic, sub_words = split_large_topics(model, frame, embeddings, min_cluster_size)
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
                # ordered by confidence so the namer reads the most typical points first
                member_texts=[points[i].text for i in 
                              [idx[j] for j in confidences.argsort()[::-1]]],
                first_seen=min(created),
                last_seen=max(created),
                counts=_counts(members),
            )
        )

    _merge_near_duplicates(topics, outliers, points, embeddings)

    return ClusterResult(
        topics=topics,
        outliers=outliers,
        min_cluster_size=min_cluster_size,
    )


def _components(embeddings: np.ndarray, threshold: float) -> list[list[int]]:
    """Index groups whose members are transitively within `threshold` cosine.

    A radius query rather than the full n x n similarity matrix: that matrix is
    fine at a thousand points and 800MB at ten thousand, and this stage runs on
    whatever a form has accumulated.
    """
    from sklearn.neighbors import NearestNeighbors

    unit = embeddings / np.clip(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12, None)
    # On unit vectors, cosine >= t is euclidean distance <= sqrt(2 - 2t).
    radius = float(np.sqrt(max(0.0, 2.0 - 2.0 * threshold)))
    neighbours = NearestNeighbors(radius=radius, metric="euclidean").fit(unit)

    parent = list(range(len(unit)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, js in enumerate(neighbours.radius_neighbors(unit, return_distance=False)):
        for j in js:
            if i != j:
                parent[find(i)] = find(int(j))

    groups: dict[int, list[int]] = {}
    for i in range(len(unit)):
        groups.setdefault(find(i), []).append(i)
    return [g for g in groups.values() if len(g) > 1]


def _merge_near_duplicates(
    topics: list[TopicDraft],
    outliers: list[ClusterablePoint],
    points: list[ClusterablePoint],
    embeddings: np.ndarray,
) -> None:
    """Put points that say the same thing in the same topic.

    Two causes, one remedy. Identical text splits because UMAP lays points out
    by a stochastic optimisation over a neighbour graph, so two nodes carrying
    the same vector still take separate paths and can land either side of a
    boundary HDBSCAN treats as hard. Near-identical text splits for a realer
    reason: the embedding carries the varying detail as well as the shared
    complaint, so "the lunch queue takes twenty minutes at the start of term"
    and "...in the afternoon" can be pulled apart by the time of day rather
    than grouped by the queue. Measured on form 3, that put three phrasings of
    one lunch-queue complaint into three different topics, and three of one
    signatures-per-request complaint into three more.

    Both are the same defect — a group the data says is one thing, reported as
    several — so one pass fixes them: take the transitive groups within
    config.NEAR_DUPLICATE_SIM and move each onto the topic already holding its
    highest-confidence member. Identical text is simply the cosine 1.0 case.
    """
    point_by_id = {p.id: p for p in points}
    embedding_by_id = {p.id: v for p, v in zip(points, embeddings)}

    home_of: dict[int, int] = {}      # point id -> index into `topics`
    confidence_of: dict[int, float] = {}
    for ti, topic in enumerate(topics):
        for point_id, confidence in topic.members:
            home_of[point_id] = ti
            confidence_of[point_id] = confidence
    if not home_of:
        return

    # Each group goes where its most confident assigned member already sits. An
    # outlier carries no confidence, so a group of only outliers has no home and
    # is left alone: it fitted nowhere, which is a finding, not an error.
    target: dict[int, int] = {}
    for group in _components(embeddings, config.NEAR_DUPLICATE_SIM):
        ids = [points[i].id for i in group]
        placed = [pid for pid in ids if pid in home_of]
        if not placed:
            continue
        best = max(placed, key=lambda pid: confidence_of[pid])
        for pid in ids:
            target[pid] = home_of[best]

    touched: set[int] = set()

    for ti, topic in enumerate(topics):
        keep = []
        for point_id, confidence in topic.members:
            home = target.get(point_id, ti)
            if home != ti:
                topics[home].members.append((point_id, confidence))
                touched.update((ti, home))
            else:
                keep.append((point_id, confidence))
        topic.members = keep

    still_out = []
    for point in outliers:
        home = target.get(point.id)
        if home is None:
            still_out.append(point)
        else:
            # It has no assignment confidence of its own; inherit the group's,
            # so the dashboard does not read a rescued outlier as a certain one.
            topics[home].members.append((point.id, 0.0))
            touched.add(home)
    outliers[:] = still_out

    for ti in touched:
        topic = topics[ti]
        member_ids = [pid for pid, _ in topic.members]
        if not member_ids:
            continue  # left with nothing; the caller drops empty topics
        members = [point_by_id[pid] for pid in member_ids]
        topic.counts = _counts(members)
        created = [m.created_at for m in members]
        topic.first_seen = min(created)
        topic.last_seen = max(created)
        vectors = np.stack([embedding_by_id[pid] for pid in member_ids])
        topic.centroid = vectors.mean(axis=0).astype(np.float32)
        topic.member_texts = [
            point_by_id[pid].text
            for pid, _ in sorted(topic.members, key=lambda m: -m[1])
        ]
