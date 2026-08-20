"""Streamlit demo: upload PDFs, ask questions, see cited answers.

Talks to the docsense package directly (no API hop) so it also works as a
standalone Hugging Face Space.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from docsense.indexing.store import list_documents
from docsense.ingestion.pipeline import ingest_pdf
from docsense.llm.factory import get_provider
from docsense.rag.chain import ask_stream
from docsense.settings import get_config, resolve_path

st.set_page_config(page_title="docsense", page_icon="📄", layout="wide")
st.title("📄 docsense")
st.caption("Ask questions about your documents — scanned or digital, with citations")

with st.sidebar:
    st.subheader("LLM backend")
    provider_name = st.radio(
        "Provider",
        ["claude", "ollama", "fake"],
        index=0,
        horizontal=True,
        help="Hot-swappable: same pipeline, different engine.",
    )
    st.subheader("Indexed documents")
    docs = list_documents()
    if docs:
        for doc_id, n in sorted(docs.items()):
            st.markdown(f"- **{doc_id}** ({n} chunks)")
    else:
        st.info("No documents yet — upload a PDF below.")

    st.subheader("Upload")
    uploaded = st.file_uploader("PDF (digital or scanned)", type="pdf")
    if uploaded is not None and st.button("Ingest"):
        raw_dir = resolve_path(get_config()["ingestion"]["raw_dir"])
        raw_dir.mkdir(parents=True, exist_ok=True)
        dest = raw_dir / Path(uploaded.name).name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.getvalue())
        Path(tmp.name).replace(dest)
        with st.spinner("Ingesting (OCR runs if pages are scanned)…"):
            doc, n_chunks = ingest_pdf(dest)
        st.success(
            f"{doc.doc_id}: {len(doc.pages)} pages ({doc.n_ocr_pages} OCR), {n_chunks} chunks"
        )
        st.rerun()

if "history" not in st.session_state:
    st.session_state.history = []

for entry in st.session_state.history:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])

if question := st.chat_input("Ask about the indexed documents…"):
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            provider = get_provider(provider_name)
            hits, token_iter = ask_stream(question, provider=provider)
        except Exception as exc:
            st.error(f"{provider_name} backend unavailable: {exc}")
            st.stop()

        answer = st.write_stream(token_iter)
        with st.expander(f"Sources ({len(hits)} chunks, provider: {provider.name})"):
            for hit in hits:
                c = hit.chunk
                st.markdown(f"**[{c.doc_id}, p.{c.page}]** · score {hit.score} · {c.source}")
                st.text(c.text[:400])
    st.session_state.history.append({"role": "assistant", "content": answer})
