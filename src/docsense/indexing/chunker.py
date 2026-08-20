"""Chunking: recursive character splitting that respects paragraph/sentence
boundaries, with page-level metadata preserved for citations."""

from __future__ import annotations

from dataclasses import dataclass

from docsense.ingestion.loader import Document
from docsense.settings import get_config

SEPARATORS = ["\n\n", "\n", ". ", " "]


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    page: int
    text: str
    source: str  # digital | ocr


def _atomize(text: str, chunk_size: int) -> list[str]:
    """Break text into pieces no larger than chunk_size, splitting at the
    coarsest separator that helps and hard-cutting unbreakable runs."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []
    for sep in SEPARATORS:
        parts = [p for p in text.split(sep) if p.strip()]
        if len(parts) > 1:
            atoms: list[str] = []
            for part in parts:
                atoms.extend(_atomize(part, chunk_size))
            return atoms
    return [text[i : i + chunk_size].strip() for i in range(0, len(text), chunk_size)]


def _split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Greedily pack atoms into chunks of at most chunk_size characters, with
    an overlap tail carried between consecutive chunks for context."""
    atoms = _atomize(text.strip(), chunk_size)
    chunks: list[str] = []
    current = ""
    for atom in atoms:
        candidate = f"{current} {atom}".strip() if current else atom
        if len(candidate) > chunk_size and current:
            chunks.append(current)
            tail = current[-overlap:].strip() if overlap else ""
            current = f"{tail} {atom}".strip()
            if len(current) > chunk_size:  # atom nearly fills a chunk: drop the tail
                current = atom
        else:
            current = candidate
    if current.strip():
        chunks.append(current)
    return chunks


def chunk_document(
    doc: Document, chunk_size: int | None = None, overlap: int | None = None
) -> list[Chunk]:
    cfg = get_config()["indexing"]
    chunk_size = chunk_size or cfg["chunk_size"]
    overlap = overlap if overlap is not None else cfg["chunk_overlap"]

    chunks: list[Chunk] = []
    for page in doc.pages:
        for i, piece in enumerate(_split(page.text, chunk_size, overlap)):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}:p{page.number}:c{i}",
                    doc_id=doc.doc_id,
                    page=page.number,
                    text=piece,
                    source=page.source,
                )
            )
    return chunks
