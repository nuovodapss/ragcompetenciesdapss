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


# SIDEBAR
st.sidebar.markdown("## Impostazioni")

settings = AppSettings()

embedding_model_name = st.sidebar.text_input("Modello embedding", value=settings.embedding_model_name)
top_k = st.sidebar.slider("Top-k retrieval", 1, 8, settings.top_k)
min_score = st.sidebar.slider("Soglia minima", 0.0, 1.0, float(settings.min_score), 0.05)

generation_mode = st.sidebar.radio("Modalità", ["sintesi_locale_light", "estrattiva"])

with st.sidebar.expander("LLM"):
    llm_repo_id = st.text_input("HF repo", value=settings.llm_repo_id)
    llm_max_new_tokens = st.slider("Max tokens", 80, 400, settings.llm_max_new_tokens)
    llm_temperature = st.slider("Temperature", 0.0, 1.0, float(settings.llm_temperature), 0.05)


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


# HEADER
st.title("🧠 Competenze RAG")

uploaded_file = st.file_uploader("Carica PDF", type=["pdf"])

if uploaded_file is None:
    st.stop()

file_bytes = uploaded_file.read()
file_hash = sha256_bytes(file_bytes)

if st.session_state.get("indexed_file_hash") != file_hash:
    reset_index_state()

if st.button("Indicizza documento"):
    with st.spinner("Parsing..."):
        pages, chunks = cached_parse_and_chunk(file_bytes)

    texts = chunks_to_text_tuple(chunks)
    vectors = cached_embeddings(file_hash, texts, embedding_model_name)

    st.session_state["indexed_file_hash"] = file_hash
    st.session_state["indexed_chunks"] = chunks
    st.session_state["indexed_vectors"] = vectors
    st.session_state["indexed_pages"] = pages

if "indexed_chunks" not in st.session_state:
    st.stop()

chunks = st.session_state["indexed_chunks"]
vectors = st.session_state["indexed_vectors"]

# LLM
llm = None
llm_error = None

if generation_mode == "sintesi_locale_light":
    try:
        llm = cached_llm(local_settings.to_tuple())
    except Exception as e:
        llm_error = str(e)

# QUERY
question = st.text_area("Domanda")

if st.button("Genera risposta"):

    with st.spinner("Ricerca..."):
        raw_results = search_chunks(
            question=question,
            chunks=chunks,
            embeddings=vectors,
            model_name=embedding_model_name,
            top_k=top_k,
        )

        results = filter_chunks(raw_results, min_score=min_score)

    output = answer_question(
        question=question,
        results=results,
        mode=generation_mode,
        llm=llm,
        settings=local_settings,
        llm_error=llm_error,
    )

    st.markdown(output["answer_markdown"])
