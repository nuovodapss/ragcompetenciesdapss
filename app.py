import hashlib
from typing import Dict, List

import streamlit as st

from rag.chunker import build_chunks_from_pages
from rag.config import AppSettings
from rag.embedder import build_embeddings, get_embedder
from rag.generator import answer_question, get_local_llm
from rag.parser import extract_pages_from_pdf_bytes
from rag.retriever import filter_chunks, search_chunks
from utils.style import inject_css


st.set_page_config(page_title="Competenze RAG", page_icon="🧠", layout="wide")
inject_css()


@st.cache_data(show_spinner=False)
def cached_parse_and_chunk(file_bytes: bytes):
    pages = extract_pages_from_pdf_bytes(file_bytes)
    chunks = build_chunks_from_pages(pages)
    return pages, chunks


@st.cache_data(show_spinner="Calcolo embeddings in corso...")
def cached_embeddings(file_hash: str, texts: tuple, model_name: str):
    _ = file_hash
    embedder = get_embedder(model_name)
    return build_embeddings(list(texts), embedder)


@st.cache_resource(show_spinner=False)
def cached_llm(settings_tuple: tuple):
    settings = AppSettings.from_tuple(settings_tuple)
    return get_local_llm(settings)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def chunks_to_text_tuple(chunks: List[Dict]) -> tuple:
    return tuple(chunk.get("text", "") for chunk in chunks)


def reset_index_state():
    for key in [
        "indexed_file_hash",
        "indexed_chunks",
        "indexed_vectors",
        "indexed_pages",
        "llm_runtime_error",
    ]:
        if key in st.session_state:
            del st.session_state[key]


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.markdown("## Impostazioni")

settings = AppSettings()

embedding_model_name = st.sidebar.text_input(
    "Modello embedding",
    value=settings.embedding_model_name,
)

top_k = st.sidebar.slider(
    "Top-k retrieval",
    min_value=1,
    max_value=8,
    value=settings.top_k,
    step=1,
)

min_score = st.sidebar.slider(
    "Soglia minima similarità",
    min_value=0.0,
    max_value=1.0,
    value=float(settings.min_score),
    step=0.05,
)

generation_mode = st.sidebar.radio(
    "Modalità risposta",
    options=["sintesi_locale_light", "estrattiva"],
    index=0 if settings.generation_mode == "sintesi_locale_light" else 1,
    help="'sintesi_locale_light' usa un piccolo modello locale via transformers. 'estrattiva' usa solo retrieval.",
)

with st.sidebar.expander("Impostazioni sintesi locale", expanded=False):
    llm_repo_id = st.text_input(
        "HF repo modello",
        value=settings.llm_repo_id,
        help="Repository Hugging Face del modello locale piccolo.",
    )
    llm_max_new_tokens = st.slider(
        "Max nuovi token risposta",
        min_value=80,
        max_value=400,
        value=settings.llm_max_new_tokens,
        step=20,
    )
    llm_temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=float(settings.llm_temperature),
        step=0.05,
    )

local_settings = AppSettings(
    embedding_model_name=embedding_model_name,
    top_k=top_k,
    min_score=min_score,
    generation_mode=generation_mode,
    llm_repo_id=llm_repo_id,
    llm_max_new_tokens=llm_max_new_tokens,
    llm_temperature=llm_temperature,
    llm_do_sample=False,
)

# =========================================================
# HEADER
# =========================================================
st.markdown("# 🧠 Competenze RAG")
st.markdown(
    "Carica un PDF, indicizzalo e interroga il documento con retrieval semantico e sintesi grounded."
)

mc1, mc2, mc3 = st.columns(3)
mc1.metric("Embedding", "Locale")
mc2.metric("Top-k", str(top_k))
mc3.metric(
    "Modalità",
    "Sintesi locale" if generation_mode == "sintesi_locale_light" else "Estrattiva",
)

st.markdown("---")

# =========================================================
# UPLOAD
# =========================================================
uploaded_file = st.file_uploader(
    "Carica il PDF delle competenze",
    type=["pdf"],
    help="PDF singolo. L'indicizzazione viene fatta in memoria.",
)

if uploaded_file is None:
    st.info("Carica un PDF per iniziare.")
    st.stop()

file_bytes = uploaded_file.read()
file_hash = sha256_bytes(file_bytes)

if st.session_state.get("indexed_file_hash") != file_hash:
    reset_index_state()

left, right = st.columns([1.2, 1])

with left:
    if st.button("Indicizza documento", use_container_width=True):
        with st.spinner("Parsing e chunking del PDF..."):
            pages, chunks = cached_parse_and_chunk(file_bytes)

        if not chunks:
            st.error("Non sono riuscito a estrarre chunk utili dal PDF.")
            st.stop()

        texts = chunks_to_text_tuple(chunks)
        vectors = cached_embeddings(file_hash, texts, embedding_model_name)

        st.session_state["indexed_file_hash"] = file_hash
        st.session_state["indexed_chunks"] = chunks
        st.session_state["indexed_vectors"] = vectors
        st.session_state["indexed_pages"] = pages
        st.session_state["llm_runtime_error"] = None

        st.success(f"Documento indicizzato. Chunk creati: {len(chunks)}")

with right:
    if st.button("Reindicizza da zero", use_container_width=True):
        reset_index_state()
        st.rerun()

if "indexed_chunks" not in st.session_state:
    st.warning("Premi 'Indicizza documento' per costruire l'indice.")
    st.stop()

chunks = st.session_state["indexed_chunks"]
vectors = st.session_state["indexed_vectors"]
pages = st.session_state["indexed_pages"]

# =========================================================
# CARICAMENTO LLM
# =========================================================
llm = None
llm_error = None

if generation_mode == "sintesi_locale_light":
    try:
        with st.spinner("Caricamento sintesi locale..."):
            llm = cached_llm(local_settings.to_tuple())
    except Exception as exc:
        llm_error = str(exc)
        st.session_state["llm_runtime_error"] = llm_error

# =========================================================
# DOMANDA
# =========================================================
st.markdown("## Fai una domanda")

question = st.text_area(
    "Domanda",
    placeholder="Es. Quali descrittori caratterizzano questa competenza? Come evolve nei livelli di Benner?",
    height=120,
)

col_a, col_b = st.columns([1, 3])
ask = col_a.button("Genera risposta", use_container_width=True)

if ask:
    if not question.strip():
        st.warning("Scrivi una domanda.")
        st.stop()

    with st.spinner("Recupero dei passaggi più rilevanti..."):
raw_results = search_chunks(
    question=question,
    chunks=chunks,
    embeddings=vectors,
    model_name=embedding_model_name,
    top_k=top_k,
)

    output = answer_question(
        question=question,
        results=results,
        mode=generation_mode,
        llm=llm,
        settings=local_settings,
        llm_error=llm_error,
    )

    st.markdown("## Risposta")
    st.markdown(output["answer_markdown"])

    if output.get("warning"):
        st.warning(output["warning"])

    st.markdown("## Fonti recuperate")

    if not output["results"]:
        st.info("Nessun chunk sopra soglia.")
    else:
        for i, hit in enumerate(output["results"], start=1):
            title = hit.get("title") or "Chunk"
            code = hit.get("code") or "-"
            area = hit.get("area") or "-"
            score = hit.get("score", 0.0)
            text = hit.get("text", "")

            st.markdown(
                f"""
<div class="source-card">
    <div class="source-card-header">
        <strong>{i}. {title}</strong>
    </div>
    <div class="source-card-meta">
        <span><strong>Codice:</strong> {code}</span> ·
        <span><strong>Area:</strong> {area}</span> ·
        <span><strong>Score:</strong> {score:.3f}</span>
    </div>
    <div class="source-card-body">{text}</div>
</div>
                """,
                unsafe_allow_html=True,
            )

# =========================================================
# DEBUG
# =========================================================
with st.expander("Debug indice", expanded=False):
    st.write(f"File hash: `{file_hash}`")
    st.write(f"Pagine estratte: {len(pages)}")
    st.write(f"Chunk indicizzati: {len(chunks)}")
    if st.session_state.get("llm_runtime_error"):
        st.write(
            f"Errore runtime sintesi locale: {st.session_state['llm_runtime_error']}"
        )
