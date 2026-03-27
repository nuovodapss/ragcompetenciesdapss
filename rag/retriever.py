from __future__ import annotations

import re
from typing import Dict, List, Optional

import numpy as np


def filter_chunks(
    chunks: List[Dict],
    areas: Optional[List[str]] = None,
    dimensions: Optional[List[str]] = None,
    code_or_keyword: Optional[str] = None,
) -> List[Dict]:
    areas = areas or []
    dimensions = dimensions or []
    needle = (code_or_keyword or "").strip().lower()

    filtered = []
    for chunk in chunks:
        if areas and chunk.get("area") not in areas:
            continue
        if dimensions and chunk.get("dimension") not in dimensions:
            continue
        if needle:
            haystack = " ".join(
                [
                    chunk.get("title") or "",
                    chunk.get("code") or "",
                    chunk.get("area") or "",
                    chunk.get("dimension") or "",
                    chunk.get("text") or "",
                ]
            ).lower()
            if needle not in haystack:
                continue
        filtered.append(chunk)
    return filtered


def search_chunks(
    question: str,
    chunks: List[Dict],
    candidate_indices: List[int],
    embeddings_matrix: np.ndarray,
    embedder,
    top_k: int,
    min_score: float,
) -> List[Dict]:
    if not candidate_indices:
        return []

    question_vec = embedder.encode([question])[0]
    candidate_matrix = embeddings_matrix[candidate_indices]
    scores = candidate_matrix @ question_vec

    ranked_positions = np.argsort(scores)[::-1]
    results: List[Dict] = []

    for pos in ranked_positions[: max(top_k * 2, top_k)]:
        score = float(scores[pos])
        if score < min_score:
            continue
        chunk = chunks[candidate_indices[pos]]
        display_text = chunk["text"]
        if len(display_text) > 2400:
            display_text = display_text[:2400].rstrip() + "…"

        results.append(
            {
                **chunk,
                "score": score,
                "display_text": display_text,
            }
        )
        if len(results) >= top_k:
            break

    return results
