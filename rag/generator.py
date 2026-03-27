from __future__ import annotations

import re
from typing import Dict, List, Optional

from rag.prompts import SYSTEM_PROMPT_IT, build_user_prompt


STOPWORDS_IT = {
    "come", "della", "delle", "degli", "nella", "nelle", "quali", "sono",
    "dalla", "dallo", "dell", "dell'", "dello", "questa", "questo", "quello",
    "quella", "sugli", "sulla", "sulle", "dopo", "prima", "perché",
    "quale", "dove", "quando", "quindi", "oppure",
}


class LocalTransformersGenerator:
    def __init__(self, settings):
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "transformers non è disponibile. Controlla requirements.txt."
            ) from exc

        self.repo_id = settings.llm_repo_id
        self.pipe = pipeline(
            task="text-generation",
            model=self.repo_id,
            tokenizer=self.repo_id,
            device=-1,
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        do_sample: bool,
    ) -> str:
        full_prompt = f"{SYSTEM_PROMPT_IT}\n\n{prompt}\n"

        outputs = self.pipe(
            full_prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            return_full_text=False,
            truncation=True,
        )
        text = outputs[0]["generated_text"].strip()
        return text


def get_local_llm(settings):
    return LocalTransformersGenerator(settings)


def keyword_tokens(question: str) -> List[str]:
    tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9']+", question.lower())
    return [t for t in tokens if len(t) >= 4 and t not in STOPWORDS_IT]


def best_sentences(text: str, question: str, limit: int = 2) -> List[str]:
    tokens = keyword_tokens(question)
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    scored = []

    for sentence in sentences:
        sent = sentence.strip()
        if len(sent) < 40:
            continue
        score = sum(1 for tok in tokens if tok in sent.lower())
        scored.append((score, len(sent), sent))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

    selected: List[str] = []
    for score, _, sent in scored:
        if score <= 0 and selected:
            continue
        if sent not in selected:
            selected.append(sent)
        if len(selected) >= limit:
            break

    if not selected:
        text = re.sub(r"\s+", " ", text).strip()
        short_text = text[:380] + ("…" if len(text) > 380 else "")
        return [short_text]

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
        "\nQuesta risposta è stata costruita in modalità estrattiva di fallback."
    )
    return "\n".join(lines)


def postprocess_answer(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


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

    if mode == "sintesi_locale_light" and llm is not None:
        prompt = build_user_prompt(question, results)
        try:
            answer = llm.generate(
                prompt=prompt,
                max_new_tokens=settings.llm_max_new_tokens,
                temperature=settings.llm_temperature,
                do_sample=settings.llm_do_sample,
            )
            answer = postprocess_answer(answer)

            if not answer:
                raise RuntimeError("Il modello ha restituito una risposta vuota.")
        except Exception as exc:
            answer = build_extractive_answer(question, results)
            warning = (
                f"Sintesi locale non disponibile in runtime: {exc}. "
                "Ho usato la modalità estrattiva."
            )
    else:
        answer = build_extractive_answer(question, results)
        if mode == "sintesi_locale_light" and llm is None:
            warning = (
                f"Sintesi locale non caricata. Motivo: {llm_error}. "
                "Ho usato la modalità estrattiva."
                if llm_error
                else "Sintesi locale non caricata. Ho usato la modalità estrattiva."
            )

    return {
        "answer_markdown": answer,
        "results": results,
        "warning": warning,
    }
