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

    generation_mode: str = os.getenv("GENERATION_MODE", "llm_locale")

    # Modifica questi due valori se vuoi un altro GGUF pubblico.
    llm_repo_id: str = os.getenv("LLM_REPO_ID", "bartowski/Qwen2.5-0.5B-Instruct-GGUF")
    llm_filename: str = os.getenv("LLM_FILENAME", "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf")

    llm_n_ctx: int = int(os.getenv("LLM_N_CTX", "4096"))
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "512"))
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.10"))

    def to_tuple(self) -> Tuple:
        return (
            self.embedding_model_name,
            self.top_k,
            self.min_score,
            self.generation_mode,
            self.llm_repo_id,
            self.llm_filename,
            self.llm_n_ctx,
            self.llm_max_tokens,
            self.llm_temperature,
        )

    @classmethod
    def from_tuple(cls, values: Tuple) -> "AppSettings":
        return cls(
            embedding_model_name=values[0],
            top_k=values[1],
            min_score=values[2],
            generation_mode=values[3],
            llm_repo_id=values[4],
            llm_filename=values[5],
            llm_n_ctx=values[6],
            llm_max_tokens=values[7],
            llm_temperature=values[8],
        )
