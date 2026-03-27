from __future__ import annotations

import re
from typing import Dict, List, Optional

from huggingface_hub import hf_hub_download

from rag.prompts import SYSTEM_PROMPT_IT, build_user_prompt


STOPWORDS_IT = {
    "come", "della", "delle", "degli", "degli", "nella", "nelle", "quali", "quali", "sono",
    "dalla", "dallo", "dallo", "degli", "dell", "dell'", "dello", "della", "nelle", "nella",
    "questa", "questo", "quello", "quella", "sugli", "sulla", "sulle", "dopo", "prima", "perché",
    "quale", "quali", "dove", "quando", "quindi", "oppure", "della", "delle", "degli", "dell",
}


class LocalLlamaCpp:
    def __init__(self, settings):
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python non è disponibile. Controlla requirements.txt o passa alla modalità estrattiva."
            ) from exc

        if not settings.llm_repo_id or not settings.llm_filename:
            raise RuntimeError("Repo GGUF o filename mancanti.")

        model_path = hf_hub_download(
            repo_id=settings.llm_repo_id,
            filename=settings.llm_filename,
        )
        self.llm = Llama(
            model_path=model_path,
            n_ctx=settings.llm_n_ctx,
            verbose=False,
        )

    def generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        output = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_IT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return output["choices"][0]["message"]["content"].strip()


def get_local_llm(settings):
    return LocalLlamaCpp(settings)


def keyword_tokens(question: str) -> List[str]:
    tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9']+", question.lower())
    return [t for t in tokens if len(t) >= 4 and t not in STOPWORDS_IT]


def best_sentences(text: str, question: str, limit: int = 2) -> List[str]:
    tokens = keyword_tokens(question)
    sentences = re.split(r"(?<=[\.!?])\s+|\n+", text)
    scored = []
    for sentence in sentences:
        sent = sentence.strip()
        if len(sent) < 40:
            continue
        score = sum(1 for tok in tokens if tok in sent.lower())
        scored.append((score, len(sent), sent))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    selected = []
    for score, _, sent in scored:
        if score <= 0 and selected:
            continue
        if sent not in selected:
            selected.append(sent)
        if len(selected) >= limit:
            break
    if not selected:
        text = re.sub(r"\s+", " ", text).strip()
        return [text[:380] + ("…" if len(text) > 380 else "")]
    return selected


def build_extractive_answer(question: str, results: List[Dict]) -> str:
    if not results:
        return "Non ho trovato passaggi sufficientemente pertinenti nel documento."

    lines = ["Ho trovato questi elementi rilevanti nel documento:"]
    for hit in results:
        label_bits = []
        if hit.get("code"):
            label_bits.append(hit["code"])
        if hit.get("title"):
            label_bits.append(hit["title"])
        label = " — ".join(label_bits) if label_bits else hit.get("title", "Passaggio")
        lines.append(f"\n**{label}**")
        for sentence in best_sentences(hit["text"], question, limit=2):
            lines.append(f"- {sentence}")
    lines.append(
        "\nQuesta è una sintesi estrattiva dai chunk recuperati. Per una risposta più naturale puoi attivare l'LLM locale."
    )
    return "\n".join(lines)


def answer_question(
    question: str,
    results: List[Dict],
    mode: str,
    llm,
    settings,
    llm_error: Optional[str] = None,
) -> Dict:
    warning = None
    if not results:
        return {
            "answer_markdown": "Non ho trovato contenuti abbastanza pertinenti nel documento.",
            "results": [],
            "warning": None,
        }

    if mode == "llm_locale" and llm is not None:
        prompt = build_user_prompt(question, results)
        try:
            answer = llm.generate(
                prompt=prompt,
                max_tokens=settings.llm_max_tokens,
                temperature=settings.llm_temperature,
            )
        except Exception as exc:
            answer = build_extractive_answer(question, results)
            warning = f"LLM locale non disponibile in runtime: {exc}. Ho usato la modalità estrattiva."
    else:
        answer = build_extractive_answer(question, results)
        if mode == "llm_locale" and llm is None:
            warning = (
                f"LLM locale non caricato. Motivo: {llm_error}. Ho usato la modalità estrattiva."
                if llm_error
                else "LLM locale non caricato. Ho usato la modalità estrattiva."
            )

    return {
        "answer_markdown": answer,
        "results": results,
        "warning": warning,
    }
