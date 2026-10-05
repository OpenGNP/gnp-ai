"""384-dimensional embeddings from all-MiniLM-L6-v2.

Model choice is measured, not assumed — see docs/embedding-comparison.md.
The target columns (points.embedding, canonical_topics.centroid_vector) are
declared as unconstrained pgvector `vector`, so 384 dims insert as-is.
"""

from __future__ import annotations

import numpy as np

from . import config

_encoder = None


def encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer

        _encoder = SentenceTransformer(config.EMBEDDING_MODEL)
    return _encoder


def encode(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    return encoder().encode(texts, show_progress_bar=False).astype(np.float32)
