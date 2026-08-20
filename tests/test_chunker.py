from docsense.indexing.chunker import chunk_document
from docsense.ingestion.loader import Document, Page


def _doc(text: str, page: int = 1) -> Document:
    return Document(doc_id="test-doc", path="x.pdf", pages=[Page(page, text, "digital")])


def test_short_page_is_single_chunk():
    chunks = chunk_document(_doc("A short paragraph."), chunk_size=500, overlap=50)
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "test-doc:p1:c0"
    assert chunks[0].page == 1


def test_long_text_is_split_with_bounded_size():
    text = "\n\n".join(f"Paragraph {i}: " + "word " * 60 for i in range(10))
    chunks = chunk_document(_doc(text), chunk_size=400, overlap=50)
    assert len(chunks) > 3
    assert all(len(c.text) <= 400 + 50 for c in chunks)


def test_empty_page_produces_no_chunks():
    assert chunk_document(_doc("   \n  ")) == []


def test_chunk_ids_are_unique():
    text = "\n\n".join(f"Sentence number {i}. " * 20 for i in range(8))
    chunks = chunk_document(_doc(text), chunk_size=300, overlap=30)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_multi_page_metadata():
    doc = Document(
        doc_id="d",
        path="x.pdf",
        pages=[Page(1, "Page one text.", "digital"), Page(2, "Page two text.", "ocr")],
    )
    chunks = chunk_document(doc, chunk_size=500, overlap=0)
    assert {c.page for c in chunks} == {1, 2}
    assert chunks[1].source == "ocr"
