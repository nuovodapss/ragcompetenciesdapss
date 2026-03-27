from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class AppSettings:
    embedding_model_name: str = os.getenv(
        "EMBEDDING_MODEL_NAME",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    top_k: int = int(os.getenv("TOP_K", "4"))
    min_score: float = float(os.getenv("MIN_SCORE", "0.20"))

    generation_mode: str = os.getenv("GENERATION_MODE", "sintesi_locale_light")

    llm_repo_id: str = os.getenv(
        "LLM_REPO_ID",
        "HuggingFaceTB/SmolLM2-360M-Instruct",
    )

    llm_max_new_tokens: int = int(os.getenv("LLM_MAX_NEW_TOKENS", "220"))
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.10"))
    llm_do_sample: bool = os.getenv("LLM_DO_SAMPLE", "false").lower() == "true"

    def to_tuple(self) -> Tuple:
        return (
            self.embedding_model_name,
            self.top_k,
            self.min_score,
            self.generation_mode,
            self.llm_repo_id,
            self.llm_max_new_tokens,
            self.llm_temperature,
            self.llm_do_sample,
        )

    @classmethod
    def from_tuple(cls, values: Tuple) -> "AppSettings":
        return cls(
            embedding_model_name=values[0],
            top_k=values[1],
            min_score=values[2],
            generation_mode=values[3],
            llm_repo_id=values[4],
            llm_max_new_tokens=values[5],
            llm_temperature=values[6],
            llm_do_sample=values[7],
        )
