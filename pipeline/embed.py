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


def encode(texts: list[str], target_dims: int | None = None) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    vectors = encoder().encode(texts, show_progress_bar=False).astype(np.float32)
    return fit_dims(vectors, target_dims)


def fit_dims(vectors: np.ndarray, target_dims: int | None) -> np.ndarray:
    """Right-pad with zeros when the column demands more dimensions than the
    model produces.

    The live schema declares vector(1536) (an OpenAI-era width) while
    all-MiniLM-L6-v2 emits 384. Zero-padding is safe for the only operation that
    matters here: appended zeros change neither the dot product nor either
    vector's norm, so cosine similarity between two padded vectors is identical
    to the similarity between the originals. It wastes 4.6KB per row and is a
    workaround for the column width, not a modelling choice — the clean fix is
    ALTER TABLE ... vector(384) plus an index rebuild.
    """
    if not target_dims or vectors.size == 0 or vectors.shape[1] == target_dims:
        return vectors
    if vectors.shape[1] > target_dims:
        raise SystemExit(
            f"embedding model emits {vectors.shape[1]} dims but the column holds {target_dims}"
        )

    padded = np.zeros((vectors.shape[0], target_dims), dtype=np.float32)
    padded[:, : vectors.shape[1]] = vectors
    return padded
