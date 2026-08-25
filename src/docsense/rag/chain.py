"""The RAG chain: backed by the Declarative DAG Pipeline."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from docsense.dag.pipeline import run_doc_qa_dag
from docsense.llm.base import LLMProvider
from docsense.llm.factory import get_provider
from docsense.retrieval.hybrid import Hit, retrieve
from docsense.settings import get_config, resolve_path

SYSTEM = "You answer questions about documents precisely and always cite your sources."


@dataclass
class RagAnswer:
    answer: str
    hits: list[Hit]
    provider: str


def _load_prompt() -> str:
    return resolve_path(get_config()["rag"]["prompt_path"]).read_text(encoding="utf-8")


def build_context(hits: list[Hit], max_chars: int) -> str:
    parts, used = [], 0
    for hit in hits:
        c = hit.chunk
        block = f"--- [{c.doc_id}, p.{c.page}] ---\n{c.text}\n"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n".join(parts)


def build_prompt(question: str, hits: list[Hit]) -> str:
    cfg = get_config()["rag"]
    return _load_prompt().format(
        context=build_context(hits, cfg["max_context_chars"]), question=question
    )


def ask(question: str, provider: LLMProvider | None = None, top_k: int | None = None) -> RagAnswer:
    """Answer a question using the declarative DAG pipeline."""
    prov = provider or get_provider()
    k = top_k or get_config()["retrieval"].get("top_k", 5)
    results = run_doc_qa_dag(question=question, provider=prov, top_k=k)
    hits: list[Hit] = results["rrf_merge"].data
    answer: str = results["llm_synthesis"].data
    return RagAnswer(answer=answer, hits=hits, provider=prov.name)


def ask_stream(
    question: str, provider: LLMProvider | None = None, top_k: int | None = None
) -> tuple[list[Hit], Iterator[str]]:
    """Return retrieved hits immediately plus a lazy answer-chunk iterator."""
    provider = provider or get_provider()
    hits = retrieve(question, top_k=top_k)
    prompt = build_prompt(question, hits)
    max_tokens = get_config()["rag"]["max_answer_tokens"]
    stream = provider.stream(prompt, system=SYSTEM, max_tokens=max_tokens)
    return hits, stream
