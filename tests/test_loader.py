import pytest

from docsense.ingestion.loader import extract_digital_pages, is_scanned_page, load_document
from docsense.ingestion.ocr import tesseract_available


def test_digital_pdf_extracts_text(digital_pdf):
    pages = extract_digital_pages(digital_pdf)
    assert len(pages) == 2
    assert "12.5 million" in pages[0][1]
    assert "supply chain" in pages[1][1]


def test_digital_pages_not_marked_scanned(digital_pdf):
    for _, text in extract_digital_pages(digital_pdf):
        assert not is_scanned_page(text, min_chars=40)


def test_image_pdf_detected_as_scanned(scanned_pdf):
    pages = extract_digital_pages(scanned_pdf)
    assert len(pages) == 1
    assert is_scanned_page(pages[0][1], min_chars=40)


def test_load_document_digital_bypasses_ocr(digital_pdf):
    doc = load_document(digital_pdf, ocr_scanned=False)
    assert doc.n_ocr_pages == 0
    assert all(p.source == "digital" for p in doc.pages)
    assert "ACME" in doc.text


@pytest.mark.skipif(not tesseract_available(), reason="Tesseract binary not installed")
def test_scanned_pdf_ocr_recovers_text(scanned_pdf):
    doc = load_document(scanned_pdf, ocr_scanned=True)
    assert doc.n_ocr_pages == 1
    text = doc.text.upper()
    assert "INVOICE" in text or "4711" in text
