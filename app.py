import hashlib
from pathlib import Path
from typing import Dict, List, Optional

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


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "indexed_chunks" not in st.session_state:
    st.session_state.indexed_chunks = []
if "chunk_embeddings" not in st.session_state:
    st.session_state.chunk_embeddings = None
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None
if "doc_hash" not in st.session_state:
    st.session_state.doc_hash = None
if "index_ready" not in st.session_state:
    st.session_state.index_ready = False


settings = AppSettings()

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.title("Competenze RAG")
    st.caption("RAG locale, senza API key")

    st.markdown("### Retrieval")
    top_k = st.slider("Top-k chunk", min_value=2, max_value=8, value=settings.top_k, step=1)
    min_score = st.slider("Soglia minima similarità", min_value=0.0, max_value=1.0, value=settings.min_score, step=0.05)

    st.markdown("### Embeddings")
    embedding_model_name = st.text_input(
        "Modello embeddings",
        value=settings.embedding_model_name,
        help="Consigliato: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )

    st.markdown("### Generazione")
 generation_mode = st.radio(
    "Modalità risposta",
    options=["sintesi_locale_light", "estrattiva"],
    index=0 if settings.generation_mode == "sintesi_locale_light" else 1,
    help="'sintesi_locale_light' usa un piccolo modello locale via transformers. 'estrattiva' funziona anche senza LLM.",
)

with st.expander("Impostazioni sintesi locale", expanded=False):
    llm_repo_id = st.text_input(
        "HF repo modello",
        value=settings.llm_repo_id,
        help="Repo pubblico Hugging Face del modello small instruct.",
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
        value=settings.llm_temperature,
        step=0.05,
    )


# -----------------------------
# HEADER / HERO
# -----------------------------
st.title("Applicativo RAG — Guida Competenze")
st.markdown(
    """
    Interroga un PDF di competenze in stile **APPGrade**: upload del documento, indicizzazione locale,
    retrieval dei passaggi più pertinenti e risposta grounded sul testo recuperato.
    """
)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        """
        <div class="card">
        <h2>1. Carica PDF</h2>
        <p>Upload del documento da interrogare direttamente nell'app, senza API key esterne.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        """
        <div class="card">
        <h2>2. Indicizza</h2>
        <p>Chunking per competenza quando possibile, altrimenti fallback a chunk generici sovrapposti.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        """
        <div class="card">
        <h2>3. Interroga</h2>
        <p>Risposta con fonti esplicite, metadati di area/dimensione/codice e similarità del retrieval.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("### Documento")

left, right = st.columns([1.15, 0.85])
with left:
    uploaded_file = st.file_uploader("Carica il PDF delle competenze", type=["pdf"])
with right:
    st.markdown(
        """
        <div class="card compact-card">
        <h2>Flusso consigliato</h2>
        <p>1) Carica il PDF<br>2) Clicca <b>Indicizza documento</b><br>3) Fai una domanda clinico-organizzativa o per codice competenza</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()
    st.session_state.doc_name = uploaded_file.name
    st.session_state.doc_hash = file_hash

    if st.button("Indicizza documento", use_container_width=False):
        with st.spinner("Lettura PDF e costruzione indice..."):
            pages, chunks = cached_parse_and_chunk(file_bytes)
            embed_matrix = cached_embeddings(
                file_hash,
                tuple(chunk["text_for_embedding"] for chunk in chunks),
                embedding_model_name,
            )

        st.session_state.indexed_chunks = chunks
        st.session_state.chunk_embeddings = embed_matrix
        st.session_state.index_ready = True
        st.session_state.chat_history = []

        st.success(
            f"Documento indicizzato: {uploaded_file.name} • pagine: {len(pages)} • chunk: {len(chunks)}"
        )


if st.session_state.index_ready:
    indexed_chunks: List[Dict] = st.session_state.indexed_chunks

    available_areas = sorted({c.get("area") for c in indexed_chunks if c.get("area")})
    available_dimensions = sorted({c.get("dimension") for c in indexed_chunks if c.get("dimension")})
    available_codes = sorted({c.get("code") for c in indexed_chunks if c.get("code")})

    st.markdown("### Esplora e interroga")

    fc1, fc2, fc3 = st.columns([1, 1, 1.2])
    with fc1:
        selected_areas = st.multiselect("Filtra per area", options=available_areas)
    with fc2:
        selected_dimensions = st.multiselect("Filtra per dimensione", options=available_dimensions)
    with fc3:
        code_search = st.text_input("Filtra per codice o parola chiave", placeholder="Es. COL1, ventilazione, territor...")

    filtered_chunks = filter_chunks(
        chunks=indexed_chunks,
        areas=selected_areas,
        dimensions=selected_dimensions,
        code_or_keyword=code_search,
    )

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Documento", st.session_state.doc_name or "—")
    mc2.metric("Chunk attivi", len(filtered_chunks))
mc3.metric("Modalità", "Sintesi locale" if generation_mode == "sintesi_locale_light" else "Estrattiva")

    question = st.text_area(
        "Fai una domanda sul documento",
        height=120,
        placeholder="Es. Quali descrittori operativi caratterizzano la competenza COL1?\nOppure: come cambia la competenza tra novizio ed esperto?",
    )

    ask_clicked = st.button("Genera risposta", type="primary")

    if ask_clicked:
        if not question.strip():
            st.warning("Scrivi prima una domanda.")
        elif not filtered_chunks:
            st.warning("Nessun chunk corrisponde ai filtri impostati.")
        else:
            filtered_ids = {c["chunk_id"] for c in filtered_chunks}
            candidate_indices = [i for i, c in enumerate(indexed_chunks) if c["chunk_id"] in filtered_ids]

            results = search_chunks(
                question=question,
                chunks=indexed_chunks,
                candidate_indices=candidate_indices,
                embeddings_matrix=st.session_state.chunk_embeddings,
                embedder=get_embedder(embedding_model_name),
                top_k=top_k,
                min_score=min_score,
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

            llm = None
            llm_error: Optional[str] = None
if generation_mode == "sintesi_locale_light":
                try:
                    llm = cached_llm(local_settings.to_tuple())
                except Exception as exc:
                    llm_error = str(exc)

            answer_payload = answer_question(
                question=question,
                results=results,
                mode=generation_mode,
                llm=llm,
                settings=local_settings,
                llm_error=llm_error,
            )

            st.session_state.chat_history.append(
                {
                    "question": question,
                    "answer": answer_payload,
                }
            )

    if st.session_state.chat_history:
        st.markdown("### Conversazione")
        for item in reversed(st.session_state.chat_history):
            with st.chat_message("user"):
                st.write(item["question"])
            with st.chat_message("assistant"):
                st.markdown(
                    f"<div class='answer-box'>{item['answer']['answer_markdown']}</div>",
                    unsafe_allow_html=True,
                )
                if item["answer"].get("warning"):
                    st.info(item["answer"]["warning"])

                st.markdown("**Fonti recuperate**")
                for hit in item["answer"]["results"]:
                    meta_bits = []
                    if hit.get("area"):
                        meta_bits.append(f"Area: {hit['area']}")
                    if hit.get("dimension"):
                        meta_bits.append(f"Dimensione: {hit['dimension']}")
                    if hit.get("code"):
                        meta_bits.append(f"Codice: {hit['code']}")
                    pages_label = f"pp. {hit['page_start']}-{hit['page_end']}" if hit['page_start'] != hit['page_end'] else f"p. {hit['page_start']}"
                    meta_bits.append(pages_label)
                    meta_bits.append(f"similarità: {hit['score']:.2f}")
                    meta = " • ".join(meta_bits)

                    with st.expander(f"{hit['title']} — {meta}"):
                        st.markdown(
                            f"<div class='source-card'><div class='source-meta'>{meta}</div><div>{hit['display_text']}</div></div>",
                            unsafe_allow_html=True,
                        )

else:
    st.info("Carica un PDF e clicca 'Indicizza documento' per iniziare.")


st.markdown("---")
st.caption(
    "Direzione Aziendale delle Professioni Sanitarie e Sociosanitarie • prototipo RAG locale Streamlit"
)
