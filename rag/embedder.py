from __future__ import annotations

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


def get_embedder(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def build_embeddings(texts: List[str], embedder: SentenceTransformer) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    try:
        vectors = embedder.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    except TypeError:
        try:
            vectors = embedder.encode(
                texts,
                convert_to_numpy=True,
            )
        except TypeError:
            vectors = embedder.encode(texts)

    return np.asarray(vectors, dtype=np.float32)
