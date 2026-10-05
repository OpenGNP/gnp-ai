"""Decide which freshly clustered group is which existing topic.

Clustering has no memory: every run produces anonymous groups. Writing them as
new rows each time is why topic ids churned and `topic_trends` never spanned
more than one run.

Identity is resolved from **membership**, not from the embedding geometry that
produced the clusters in the first place — asking "where did topic 44's points
go?" is direct evidence, while comparing centroids is a proxy for it. Centroid
similarity is kept only as a tiebreak for groups that contain no old points at
all, where membership cannot answer.

The measure is containment, not Jaccard:

    containment(old, new) = |old ∩ new| / |old still in the corpus|

A topic that keeps all its old points and gains twenty new ones is the same
topic; Jaccard would punish it for growing.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from . import config
from .cluster import ClusterResult
from .db import ClusterablePoint


@dataclass
class IdentityPlan:
    """What each new group should be written as."""

    # draft key -> existing topic id to carry forward
    reuse: dict[str, int] = field(default_factory=dict)
    # containment scores kept for the run log: draft key -> {old topic id: share}
    containment: dict[str, dict[int, float]] = field(default_factory=dict)
    # absorbed topic id -> surviving topic id
    merged: dict[int, int] = field(default_factory=dict)
    # old topics nothing claimed and nothing absorbed — safe to remove
    gone: list[int] = field(default_factory=list)
    events: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        kinds = Counter(event.split(":", 1)[0] for event in self.events)
        return {
            "carried": len(self.reuse),
            "new": kinds.get("new", 0),
            "split": kinds.get("split", 0),
            "merged": len(self.merged),
            "gone": len(self.gone),
        }


def resolve(
    points: list[ClusterablePoint],
    result: ClusterResult,
    known_topics: dict[int, str],
    *,
    carry_min: float | None = None,
    split_min: float | None = None,
) -> IdentityPlan:
    carry_min = config.TOPIC_CARRY_MIN if carry_min is None else carry_min
    split_min = config.TOPIC_SPLIT_MIN if split_min is None else split_min

    previous = {
        p.id: p.canonical_topic_id
        for p in points
        if p.canonical_topic_id in known_topics
    }
    old_size = Counter(previous.values())

    plan = IdentityPlan()
    if not old_size:
        plan.events.append(f"new:{len(result.topics)} (no previous pipeline topics)")
        return plan

    # ── how much of each old topic landed in each new group ──
    for draft in result.topics:
        hits = Counter(
            previous[point_id] for point_id, _ in draft.members if point_id in previous
        )
        plan.containment[draft.key] = {
            old_id: count / old_size[old_id] for old_id, count in hits.items()
        }

    # ── claim: each group takes the old topic it holds the most of ──
    # Ties are broken by absolute count then by id so two runs over the same
    # data cannot disagree.
    claims: dict[int, list[tuple[float, int, str]]] = {}
    for draft in result.topics:
        scores = plan.containment[draft.key]
        if not scores:
            continue
        best_id = max(scores, key=lambda old_id: (scores[old_id], -old_id))
        if scores[best_id] >= carry_min:
            claims.setdefault(best_id, []).append(
                (scores[best_id], len(draft.members), draft.key)
            )

    for old_id, contenders in claims.items():
        contenders.sort(key=lambda c: (-c[0], -c[1], c[2]))
        winner = contenders[0][2]
        plan.reuse[winner] = old_id
        for _, _, loser in contenders[1:]:
            plan.events.append(
                f"new:{loser} also matched {old_id} but lost the claim"
            )

    # ── merges: a group that holds most of two old topics absorbs the loser ──
    for draft in result.topics:
        survivor = plan.reuse.get(draft.key)
        if survivor is None:
            continue
        for old_id, share in plan.containment[draft.key].items():
            if old_id != survivor and share >= carry_min:
                plan.merged[old_id] = survivor
                plan.events.append(f"merged:{old_id} into {survivor}")

    # ── events: what happened to each old topic ──
    for old_id in old_size:
        children = [
            draft.key
            for draft in result.topics
            if plan.containment[draft.key].get(old_id, 0.0) >= split_min
        ]
        if len(children) > 1:
            plan.events.append(f"split:{old_id} -> {', '.join(children)}")
        if old_id not in plan.reuse.values() and old_id not in plan.merged:
            plan.gone.append(old_id)
            plan.events.append(f"gone:{old_id} ({known_topics.get(old_id, '?')})")

    for draft in result.topics:
        if draft.key not in plan.reuse:
            plan.events.append(f"new:{draft.key}")

    return plan
