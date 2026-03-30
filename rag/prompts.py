from __future__ import annotations

from typing import Dict, List


SYSTEM_PROMPT_IT = """
Sei un assistente clinico esperto.

Devi rispondere in modo:
- chiaro
- sintetico
- strutturato

Regole OBBLIGATORIE:
- NON fare elenchi numerati automatici
- NON ripetere la domanda
- NON generare nuove domande
- NON inventare contenuti
- usa SOLO le informazioni nei testi forniti

Se non trovi la risposta, scrivi:
"Informazione non presente nei contenuti forniti."
""".strip()


def build_user_prompt(question: str, results: List[Dict]) -> str:
    context_blocks = []

    for i, hit in enumerate(results, start=1):
        block = f"""
[Fonte {i}]
Titolo: {hit.get("title", "")}
Codice: {hit.get("code", "")}
Area: {hit.get("area", "")}

Contenuto:
{hit.get("text", "")}
"""
        context_blocks.append(block.strip())

    context = "\n\n".join(context_blocks)

    return f"""
DOMANDA:
{question}

TESTI DISPONIBILI:
{context}

ISTRUZIONE:
Rispondi alla domanda usando SOLO i testi sopra.
Scrivi una risposta breve (max 8 righe).
Non fare elenchi numerati.
"""
