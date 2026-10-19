"""2D scatter of one form's points, colored by their canonical topic.

UMAP projects the 384-dim point embeddings down to 2 dimensions purely for
display — this is a separate reduction from the 5-component one `pipeline.cluster`
uses to actually cluster, and never feeds back into the database. Color comes
from `points.canonical_topic_id`, i.e. whatever the last clustering run wrote;
this script only reads.

Usage:
    python -m scripts.visualize_clusters --form-id 1
    python -m scripts.visualize_clusters --form-id 1 --output data/reports/clusters.html
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

from pipeline import config, db

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "data" / "reports"


def _hover_text(text: str, limit: int = 160) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--form-id", type=int, required=True)
    ap.add_argument("--output", default=None, metavar="PATH",
                     help="HTML file to write (default: data/reports/clusters_form<ID>.html)")
    args = ap.parse_args()

    conn = db.connect()
    points = db.clusterable_points(conn, args.form_id)
    if not points:
        raise SystemExit(f"no embedded points for form_id={args.form_id}")

    topic_names = db.existing_topics(conn, args.form_id)

    embeddings = np.stack([db.as_array(p.embedding) for p in points])
    n = len(points)

    from umap import UMAP

    n_neighbors = max(2, min(15, n - 1))
    coords = UMAP(
        n_neighbors=n_neighbors, n_components=2, min_dist=0.1,
        metric="cosine", random_state=42,
    ).fit_transform(embeddings)

    labels = [
        topic_names.get(p.canonical_topic_id, f"topic {p.canonical_topic_id}")
        if p.canonical_topic_id is not None else "unassigned"
        for p in points
    ]

    import pandas as pd
    import plotly.express as px

    frame = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "topic": labels,
        "text": [_hover_text(p.text) for p in points],
        "sentiment": [p.sentiment_label or "n/a" for p in points],
        "violation": [bool(p.is_violation) for p in points],
    })

    # draw "unassigned" first (bottom layer, behind real topics) so outliers
    # don't visually bury the clusters they were left out of
    category_order = ["unassigned"] + sorted(t for t in frame["topic"].unique() if t != "unassigned")

    fig = px.scatter(
        frame, x="x", y="y", color="topic",
        category_orders={"topic": category_order},
        color_discrete_map={"unassigned": "#cccccc"},
        hover_data={"text": True, "sentiment": True, "violation": True, "x": False, "y": False},
        title=f"form_id={args.form_id} · {n} points · {len(topic_names)} topics",
        opacity=0.75,
    )
    fig.update_traces(marker=dict(size=7))
    fig.update_layout(legend_title_text="topic")

    REPORTS.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else REPORTS / f"clusters_form{args.form_id}.html"
    fig.write_html(out_path)
    print(f"{n} points, {len(topic_names)} topics -> {out_path}")


if __name__ == "__main__":
    main()
