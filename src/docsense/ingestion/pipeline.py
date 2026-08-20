"""End-to-end ingestion: PDF -> pages (OCR as needed) -> chunks -> ChromaDB.

Usage:
    python -m docsense.ingestion.pipeline path/to/file.pdf [more.pdf ...]
    python -m docsense.ingestion.pipeline --all     # ingest data/raw_pdfs
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from docsense.indexing.chunker import chunk_document
from docsense.indexing.store import upsert_chunks
from docsense.ingestion.loader import Document, load_document
from docsense.retrieval.hybrid import invalidate_bm25_cache
from docsense.settings import get_config, resolve_path

logger = logging.getLogger(__name__)


def ingest_pdf(pdf_path: Path) -> tuple[Document, int]:
    """Ingest one PDF; returns (document, n_chunks_indexed)."""
    doc = load_document(pdf_path)
    chunks = chunk_document(doc)
    n = upsert_chunks(chunks)
    invalidate_bm25_cache()
    logger.info(
        "%s: %d pages (%d OCR), %d chunks indexed", doc.doc_id, len(doc.pages), doc.n_ocr_pages, n
    )
    return doc, n


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdfs", nargs="*", type=Path)
    parser.add_argument("--all", action="store_true", help="ingest every PDF in data/raw_pdfs")
    args = parser.parse_args()

    paths = list(args.pdfs)
    if args.all:
        paths.extend(sorted(resolve_path(get_config()["ingestion"]["raw_dir"]).glob("*.pdf")))
    if not paths:
        parser.error("No PDFs given. Pass paths or --all.")
    for path in paths:
        ingest_pdf(path)


if __name__ == "__main__":
    main()
