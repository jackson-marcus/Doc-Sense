"""OCR engine: pdf2image + Tesseract, with per-page confidence scores.

Requires the Tesseract binary (and Poppler for pdf2image). Both are installed
in the Docker image; for local Windows use see the README. All call sites go
through ocr_pdf_pages/ocr_image so the engine can be swapped in one place.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

from docsense.settings import get_config

logger = logging.getLogger(__name__)


def tesseract_available() -> bool:
    import pytesseract

    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def ocr_image(image: Image.Image, lang: str | None = None) -> tuple[str, float]:
    """OCR a PIL image; returns (text, mean_word_confidence 0-100)."""
    import pytesseract

    if lang is None:
        lang = get_config()["ingestion"]["ocr_lang"]
    data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)
    words, confidences = [], []
    for word, conf in zip(data["text"], data["conf"], strict=True):
        if word.strip():
            words.append(word)
            conf_val = float(conf)
            if conf_val >= 0:  # -1 marks non-word boxes
                confidences.append(conf_val)
    text = " ".join(words)
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return text, round(mean_conf, 2)


def ocr_pdf_pages(pdf_path: Path, page_numbers: list[int]) -> dict[int, tuple[str, float]]:
    """OCR specific pages (1-based) of a PDF; returns {page: (text, confidence)}."""
    from pdf2image import convert_from_path

    dpi = get_config()["ingestion"]["ocr_dpi"]
    results: dict[int, tuple[str, float]] = {}
    for num in page_numbers:
        images = convert_from_path(str(pdf_path), dpi=dpi, first_page=num, last_page=num)
        if not images:
            logger.warning("%s: could not render page %d", pdf_path.name, num)
            continue
        results[num] = ocr_image(images[0])
    return results
