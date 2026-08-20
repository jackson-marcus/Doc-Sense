"""PDF loading with digital-vs-scanned detection.

A page is "digital" if pypdf can extract a reasonable amount of text from it;
otherwise it is treated as scanned and routed through OCR. A single PDF can mix
both kinds of pages.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader

from docsense.settings import get_config

logger = logging.getLogger(__name__)


@dataclass
class Page:
    number: int  # 1-based
    text: str
    source: str  # "digital" | "ocr"
    ocr_confidence: float | None = None


@dataclass
class Document:
    doc_id: str
    path: str
    pages: list[Page] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)

    @property
    def n_ocr_pages(self) -> int:
        return sum(1 for p in self.pages if p.source == "ocr")


def extract_digital_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Return (page_number, extracted_text) for every page via pypdf."""
    reader = PdfReader(str(pdf_path))
    return [(i + 1, (page.extract_text() or "").strip()) for i, page in enumerate(reader.pages)]


def is_scanned_page(text: str, min_chars: int | None = None) -> bool:
    if min_chars is None:
        min_chars = get_config()["ingestion"]["min_chars_per_page"]
    return len(text.strip()) < min_chars


def load_document(pdf_path: Path, ocr_scanned: bool = True) -> Document:
    """Load a PDF, OCR-ing pages that have no usable text layer."""
    doc = Document(doc_id=pdf_path.stem, path=str(pdf_path))
    digital = extract_digital_pages(pdf_path)
    scanned_pages = [num for num, text in digital if is_scanned_page(text)]

    ocr_results: dict[int, tuple[str, float]] = {}
    if scanned_pages and ocr_scanned:
        from docsense.ingestion.ocr import ocr_pdf_pages

        logger.info("%s: OCR on %d scanned page(s)", pdf_path.name, len(scanned_pages))
        ocr_results = ocr_pdf_pages(pdf_path, scanned_pages)

    for num, text in digital:
        if num in ocr_results:
            ocr_text, confidence = ocr_results[num]
            doc.pages.append(Page(num, ocr_text, "ocr", ocr_confidence=confidence))
        else:
            doc.pages.append(Page(num, text, "digital"))
    return doc
