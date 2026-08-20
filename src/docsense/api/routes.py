"""API routes: /upload, /ask (SSE streaming), /documents, /health."""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from docsense.indexing.store import list_documents
from docsense.ingestion.pipeline import ingest_pdf
from docsense.llm.factory import get_provider
from docsense.rag.chain import ask_stream
from docsense.settings import get_config, get_settings, resolve_path

logger = logging.getLogger(__name__)
router = APIRouter()


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    provider: str | None = Field(default=None, description="claude | ollama | fake")
    top_k: int | None = Field(default=None, ge=1, le=20)


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "provider": get_settings().llm_provider}


@router.get("/documents")
def documents() -> dict[str, int]:
    return list_documents()


@router.post("/upload")
async def upload(file: UploadFile) -> dict:
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=422, detail="Only PDF files are accepted")
    raw_dir = resolve_path(get_config()["ingestion"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    # Sanitize: keep the basename only.
    dest = raw_dir / Path(file.filename).name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)
    tmp_path.replace(dest)
    doc, n_chunks = ingest_pdf(dest)
    return {
        "doc_id": doc.doc_id,
        "pages": len(doc.pages),
        "ocr_pages": doc.n_ocr_pages,
        "chunks_indexed": n_chunks,
    }


@router.post("/ask")
def ask_endpoint(request: AskRequest) -> EventSourceResponse:
    """Stream the answer via SSE: first a `sources` event, then `token` events."""
    try:
        provider = get_provider(request.provider)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    def event_stream():
        hits, token_iter = ask_stream(request.question, provider=provider, top_k=request.top_k)
        yield {
            "event": "sources",
            "data": json.dumps(
                [
                    {
                        "doc_id": h.chunk.doc_id,
                        "page": h.chunk.page,
                        "score": h.score,
                        "preview": h.chunk.text[:200],
                    }
                    for h in hits
                ]
            ),
        }
        try:
            for token in token_iter:
                yield {"event": "token", "data": token}
        except Exception as exc:  # provider/network failure mid-stream
            logger.exception("Streaming failure")
            yield {"event": "error", "data": str(exc)}
        yield {"event": "done", "data": provider.name}

    return EventSourceResponse(event_stream())
