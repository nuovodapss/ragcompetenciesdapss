from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import List

from pypdf import PdfReader


@dataclass
class PageText:
    page_number: int
    text: str


def extract_pages_from_pdf_bytes(file_bytes: bytes) -> List[PageText]:
    reader = PdfReader(BytesIO(file_bytes))
    pages: List[PageText] = []
    for idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(PageText(page_number=idx, text=text))
    return pages
