from __future__ import annotations

import re
from typing import Dict, List

from rag.parser import PageText

PAGE_MARKER_TEMPLATE = "[[PAGE={page}]]"
HEADING_PATTERN = re.compile(
    r"(?m)^(?P<code>[A-Z][A-Z0-9]{1,10}\d{1,3})\s+[—-]\s+(?P<title>.+)$"
)
METADATA_PATTERN = re.compile(
    r"Area:\s*(?P<area>.*?)\s*\|\s*Dimensione:\s*(?P<dimension>.*?)\s*\|\s*Codice:\s*(?P<code>[A-Z0-9]+)",
    flags=re.IGNORECASE | re.DOTALL,
)


def normalize_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def attach_page_markers(pages: List[PageText]) -> str:
    parts = []
    for page in pages:
        parts.append(PAGE_MARKER_TEMPLATE.format(page=page.page_number))
        parts.append("\n")
        parts.append(normalize_text(page.text))
        parts.append("\n\n")
    return "".join(parts)


def strip_page_markers(text: str) -> str:
    text = re.sub(r"\[\[PAGE=\d+\]\]", "", text)
    text = re.sub(r"(?m)^Pagina\s+\d+\s*$", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_page_range(text_with_markers: str):
    pages = [int(x) for x in re.findall(r"\[\[PAGE=(\d+)\]\]", text_with_markers)]
    if not pages:
        return 1, 1
    return min(pages), max(pages)


def infer_title_from_text(block_text: str) -> str:
    first_line = next((ln.strip() for ln in block_text.splitlines() if ln.strip()), "Chunk documento")
    return first_line[:140]


def parse_metadata(block_text: str, fallback_code: str, fallback_title: str) -> Dict:
    meta_match = METADATA_PATTERN.search(block_text)
    if meta_match:
        area = meta_match.group("area").strip()
        dimension = meta_match.group("dimension").strip()
        code = meta_match.group("code").strip()
    else:
        area = None
        dimension = None
        code = fallback_code

    return {
        "area": area,
        "dimension": dimension,
        "code": code,
        "title": fallback_title.strip(),
    }


def split_competency_chunks(marked_text: str) -> List[Dict]:
    matches = list(HEADING_PATTERN.finditer(marked_text))
    chunks: List[Dict] = []

    if not matches:
        return chunks

    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(marked_text)
        raw_block = marked_text[start:end].strip()
        page_start, page_end = extract_page_range(raw_block)
        cleaned_block = strip_page_markers(raw_block)

        fallback_code = match.group("code").strip()
        fallback_title = match.group("title").strip()
        metadata = parse_metadata(cleaned_block, fallback_code=fallback_code, fallback_title=fallback_title)

        chunks.append(
            {
                "chunk_id": f"chunk_{idx+1:04d}",
                "chunk_type": "competenza",
                "title": metadata["title"],
                "area": metadata["area"],
                "dimension": metadata["dimension"],
                "code": metadata["code"],
                "page_start": page_start,
                "page_end": page_end,
                "text": cleaned_block,
                "text_for_embedding": build_embedding_text(metadata, cleaned_block),
            }
        )

    return chunks


def build_embedding_text(metadata: Dict, text: str) -> str:
    bits = []
    if metadata.get("title"):
        bits.append(f"Titolo: {metadata['title']}")
    if metadata.get("area"):
        bits.append(f"Area: {metadata['area']}")
    if metadata.get("dimension"):
        bits.append(f"Dimensione: {metadata['dimension']}")
    if metadata.get("code"):
        bits.append(f"Codice: {metadata['code']}")
    bits.append(text)
    return "\n".join(bits)


def split_generic_chunks(marked_text: str, chunk_chars: int = 1800, overlap: int = 250) -> List[Dict]:
    cleaned_text = strip_page_markers(marked_text)
    chunks: List[Dict] = []
    start = 0
    idx = 0
    while start < len(cleaned_text):
        end = min(start + chunk_chars, len(cleaned_text))
        segment = cleaned_text[start:end]
        if end < len(cleaned_text):
            cut = max(segment.rfind("\n\n"), segment.rfind(". "))
            if cut > chunk_chars * 0.55:
                end = start + cut + 1
                segment = cleaned_text[start:end]
        idx += 1
        chunks.append(
            {
                "chunk_id": f"generic_{idx:04d}",
                "chunk_type": "generico",
                "title": infer_title_from_text(segment),
                "area": None,
                "dimension": None,
                "code": None,
                "page_start": 1,
                "page_end": 1,
                "text": segment.strip(),
                "text_for_embedding": segment.strip(),
            }
        )
        if end >= len(cleaned_text):
            break
        start = max(0, end - overlap)
    return chunks


def build_chunks_from_pages(pages: List[PageText]) -> List[Dict]:
    marked_text = attach_page_markers(pages)
    competency_chunks = split_competency_chunks(marked_text)
    if competency_chunks and len(competency_chunks) >= 3:
        return competency_chunks
    return split_generic_chunks(marked_text)
