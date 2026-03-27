SYSTEM_PROMPT_IT = """Sei un assistente documentale per una guida sulle competenze infermieristiche.
Rispondi solo usando il contesto fornito.
Non inventare nulla.
Se il contesto non basta, dillo esplicitamente.
Rispondi in italiano chiaro, sintetico ma preciso.
Quando utile, organizza la risposta in punti.
"""


def build_context_block(results):
    parts = []
    for i, hit in enumerate(results, start=1):
        header = f"[Fonte {i}]"
        meta = []
        if hit.get("title"):
            meta.append(f"Titolo: {hit['title']}")
        if hit.get("area"):
            meta.append(f"Area: {hit['area']}")
        if hit.get("dimension"):
            meta.append(f"Dimensione: {hit['dimension']}")
        if hit.get("code"):
            meta.append(f"Codice: {hit['code']}")
        meta.append(f"Pagine: {hit['page_start']}-{hit['page_end']}")
        parts.append("\n".join([header, *meta, hit["text"]]))
    return "\n\n".join(parts)


def build_user_prompt(question, results):
    context = build_context_block(results)
    return f"""Domanda dell'utente:
{question}

Contesto disponibile:
{context}

Istruzioni:
- usa solo il contesto;
- se la domanda chiede differenze tra livelli di Benner, esplicita l'evoluzione;
- se la domanda riguarda descrittori, separa attitudini, motivazioni, skills e conoscenze quando presenti;
- se mancano informazioni, dichiaralo chiaramente.

Risposta:
"""
