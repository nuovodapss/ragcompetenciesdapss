from __future__ import annotations

from typing import Dict, List

import numpy as np

from rag.embedder import get_embedder


def _to_numpy(x) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return x / norms


def cosine_similarity_matrix(query_vec: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
    query_vec = _normalize_rows(_to_numpy(query_vec))
    doc_matrix = _normalize_rows(_to_numpy(doc_matrix))
    sims = np.dot(doc_matrix, query_vec.T).reshape(-1)
    return sims


def _safe_encode(embedder, texts: List[str]) -> np.ndarray:
    try:
        vecs = embedder.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    except TypeError:
        try:
            vecs = embedder.encode(
                texts,
                convert_to_numpy=True,
            )
        except TypeError:
            vecs = embedder.encode(texts)

    vecs = np.asarray(vecs, dtype=np.float32)
    if vecs.ndim == 1:
        vecs = vecs.reshape(1, -1)
    return vecs


def search_chunks(
    question: str,
    chunks: List[Dict],
    embeddings,
    model_name: str,
    top_k: int = 4,
) -> List[Dict]:
    if not question or not question.strip():
        return []

    if not chunks or embeddings is None:
        return []

    embedder = get_embedder(model_name)
    query_embedding = _safe_encode(embedder, [question])

    doc_embeddings = _to_numpy(embeddings)
    similarities = cosine_similarity_matrix(query_embedding, doc_embeddings)

    results: List[Dict] = []
    max_idx = min(len(chunks), len(similarities))

    for idx in range(max_idx):
        chunk = dict(chunks[idx])
        chunk["score"] = float(similarities[idx])
        results.append(chunk)

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def filter_chunks(results: List[Dict], min_score: float = 0.20) -> List[Dict]:
    if not results:
        return []

    filtered = [r for r in results if float(r.get("score", 0.0)) >= float(min_score)]
    return filtered if filtered else results[:1]
