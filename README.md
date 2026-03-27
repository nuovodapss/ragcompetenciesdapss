# Competenze RAG — Streamlit

Prototipo RAG leggero, interamente **senza API key**, per interrogare un PDF di competenze in Streamlit con stile grafico coerente con l'app APPGrade.

## Cosa fa

- upload del PDF
- parsing del testo
- chunking **per competenza** quando riconosce pattern come `COL1 — Titolo`
- embeddings locali con `sentence-transformers`
- retrieval top-k in memoria
- risposta:
  - **LLM locale** via `llama-cpp-python` + modello GGUF pubblico
  - oppure **estrattiva** se il modello non si carica o vuoi stare più leggero

## Struttura repo

```text
competenze_rag_app/
├── app.py
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml
├── assets/
│   └── style.css
├── utils/
│   └── style.py
└── rag/
    ├── config.py
    ├── parser.py
    ├── chunker.py
    ├── embedder.py
    ├── retriever.py
    ├── prompts.py
    └── generator.py
```

## Avvio locale

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy su GitHub + Streamlit Community Cloud

1. crea un repository GitHub
2. carica tutti questi file
3. apri Streamlit Community Cloud
4. collega il repo
5. come entrypoint scegli `app.py`

## Note pratiche

### 1) Modalità più robusta
Se vuoi minimizzare i problemi di deploy, in sidebar passa a **Modalità risposta = estrattiva**. Il retrieval resta identico e non usi il modello generativo.

### 2) Modalità LLM locale
La modalità `llm_locale` prova a scaricare un modello GGUF pubblico da Hugging Face. I parametri sono configurabili da sidebar o variabili d'ambiente:

- `LLM_REPO_ID`
- `LLM_FILENAME`
- `LLM_N_CTX`
- `LLM_MAX_TOKENS`
- `LLM_TEMPERATURE`

### 3) Embeddings consigliati
Default:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

### 4) Limiti attesi
Su Streamlit Community Cloud questo va pensato come **RAG leggero**:
- pochi documenti
- PDF medio-piccoli
- top-k basso
- modello GGUF molto piccolo

## Suggerimento operativo
Per il tuo caso io terrei:
- top-k = 4
- soglia = 0.20–0.25
- modalità iniziale = `estrattiva`
- attivazione `llm_locale` solo dopo che il deploy è stabile

