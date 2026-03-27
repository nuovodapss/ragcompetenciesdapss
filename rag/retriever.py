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


def search_chunks(
    question: str,
    chunks: List[Dict],
    embeddings,
    model_name: str,
    top_k: int = 4,
) -> List[Dict]:
    if not question or not question.strip():
        return []

    if not chunks:
        return []

    if embeddings is None:
        return []

    embedder = get_embedder(model_name)
    query_embedding = embedder.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    doc_embeddings = _to_numpy(embeddings)
    similarities = cosine_similarity_matrix(query_embedding, doc_embeddings)

    results: List[Dict] = []
    for idx, score in enumerate(similarities):
        chunk = chunks[idx].copy()
        chunk["score"] = float(score)
        results.append(chunk)

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def filter_chunks(results: List[Dict], min_score: float = 0.20) -> List[Dict]:
    if not results:
        return []

    filtered = [r for r in results if float(r.get("score", 0.0)) >= float(min_score)]

    if filtered:
        return filtered

    # fallback: se nessun chunk supera la soglia, restituisci almeno il migliore
    return results[:1]
