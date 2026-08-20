"""Shared fixtures: on-the-fly PDFs, isolated Chroma dir, stub embeddings.

Embeddings are replaced with a deterministic bag-of-words hash projection so
tests never download the sentence-transformers model; BM25 and Chroma still
run for real.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np
import pytest

import docsense.indexing.store as store
import docsense.retrieval.hybrid as hybrid
from docsense.settings import get_config

DIM = 64


def _stub_embed(texts: list[str]) -> list[list[float]]:
    out = []
    for text in texts:
        vec = np.zeros(DIM)
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            digest = int(hashlib.md5(token.encode()).hexdigest(), 16)
            vec[digest % DIM] += 1.0
        norm = np.linalg.norm(vec)
        out.append((vec / norm if norm else vec).tolist())
    return out


@pytest.fixture(autouse=True)
def isolated_index(tmp_path, monkeypatch):
    """Point Chroma at a temp dir and stub embeddings; reset all caches."""
    cfg = get_config()
    original = cfg["indexing"]["chroma_dir"]
    cfg["indexing"]["chroma_dir"] = str(tmp_path / "chroma")

    monkeypatch.setattr(store, "embed_texts", _stub_embed)
    monkeypatch.setattr(hybrid, "embed_query", lambda q: _stub_embed([q])[0])

    store._client.cache_clear()
    hybrid._bm25_index.cache_clear()
    yield
    cfg["indexing"]["chroma_dir"] = original
    store._client.cache_clear()
    hybrid._bm25_index.cache_clear()


@pytest.fixture()
def digital_pdf(tmp_path):
    """A 2-page digital PDF with known text, generated via reportlab."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    path = tmp_path / "acme-report.pdf"
    c = canvas.Canvas(str(path), pagesize=letter)
    c.drawString(72, 720, "ACME Corporation Annual Report 2025.")
    c.drawString(72, 700, "Total revenue was 12.5 million dollars in fiscal year 2025.")
    c.showPage()
    c.drawString(72, 720, "Risk factors include supply chain concentration in a single region.")
    c.drawString(72, 700, "The company employs 340 people worldwide.")
    c.showPage()
    c.save()
    return path


@pytest.fixture()
def scanned_pdf(tmp_path):
    """An image-only PDF (no text layer) — rendered text on a PIL image."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (1700, 2200), "white")
    draw = ImageDraw.Draw(img)
    draw.text((100, 200), "INVOICE NUMBER 4711", fill="black")
    draw.text((100, 300), "Amount due: 1,250.00 USD", fill="black")
    path = tmp_path / "scan-invoice.pdf"
    img.save(path, "PDF")
    return path


def make_corpus() -> list:
    """Small chunk corpus with distinct topics for retrieval tests."""
    from docsense.indexing.chunker import Chunk

    docs = {
        "acme-10k": [
            "Total revenue was 12.5 million dollars in fiscal year 2025.",
            "Risk factors include supply chain concentration in a single region.",
            "The board approved a share buyback program of 2 million dollars.",
        ],
        "globex-10k": [
            "Globex reported a net loss of 3 million dollars for the year.",
            "The company opened a new research facility in Berlin.",
            "Employee headcount grew to 5,000 across all offices.",
        ],
    }
    chunks = []
    for doc_id, texts in docs.items():
        for i, text in enumerate(texts):
            chunks.append(Chunk(f"{doc_id}:p{i + 1}:c0", doc_id, i + 1, text, "digital"))
    return chunks
