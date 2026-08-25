"""Declarative DAG Architecture - Standard Document Intelligence Nodes.

Discrete, modular nodes for hybrid retrieval, context packing, prompt assembly,
and LLM synthesis.
"""

from __future__ import annotations

from typing import Any

from docsense.dag.node import NodeContext
from docsense.indexing.chunker import Chunk
from docsense.llm.base import LLMProvider
from docsense.retrieval.hybrid import Hit, bm25_search, dense_search
from docsense.settings import get_config, resolve_path


class DenseRetrievalNode:
    """DAG Node: Embed question and retrieve semantic nearest neighbor chunks."""

    def __init__(self, candidates: int = 10) -> None:
        self._candidates = candidates

    @property
    def name(self) -> str:
        return "dense_retrieval"

    @property
    def dependencies(self) -> list[str]:
        return []

    def execute(self, inputs: dict[str, Any], context: NodeContext) -> list[tuple[str, Chunk]]:
        question = inputs["question"]
        cand = context.params.get("candidates", self._candidates)
        collection = context.params.get("collection_name")
        try:
            return dense_search(question, k=cand, collection_name=collection)
        except Exception:
            return []


class BM25RetrievalNode:
    """DAG Node: Lexical keyword search over indexed chunks."""

    def __init__(self, candidates: int = 10) -> None:
        self._candidates = candidates

    @property
    def name(self) -> str:
        return "bm25_retrieval"

    @property
    def dependencies(self) -> list[str]:
        return []

    def execute(self, inputs: dict[str, Any], context: NodeContext) -> list[tuple[str, Chunk]]:
        question = inputs["question"]
        cand = context.params.get("candidates", self._candidates)
        collection = context.params.get("collection_name")
        try:
            return bm25_search(question, k=cand, collection_name=collection)
        except Exception:
            return []


class RRFMergeNode:
    """DAG Node: Reciprocal Rank Fusion of dense and lexical candidate lists."""

    def __init__(self, top_k: int = 5, rrf_k: int = 60) -> None:
        self._top_k = top_k
        self._rrf_k = rrf_k

    @property
    def name(self) -> str:
        return "rrf_merge"

    @property
    def dependencies(self) -> list[str]:
        return ["dense_retrieval", "bm25_retrieval"]

    def execute(self, inputs: dict[str, Any], context: NodeContext) -> list[Hit]:
        dense_results: list[tuple[str, Chunk]] = inputs.get("dense_retrieval", [])
        bm25_results: list[tuple[str, Chunk]] = inputs.get("bm25_retrieval", [])
        top_k = context.params.get("top_k", self._top_k)
        rrf_k = context.params.get("rrf_k", self._rrf_k)

        ranked_lists = [dense_results, bm25_results]
        fused: dict[str, float] = {}
        by_id: dict[str, Chunk] = {}

        for ranked in ranked_lists:
            for rank, (cid, chunk) in enumerate(ranked):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
                by_id[cid] = chunk

        best = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [Hit(chunk=by_id[cid], score=round(score, 6)) for cid, score in best]


class ContextAssemblyNode:
    """DAG Node: Assemble grounded context window from top retrieved hits."""

    @property
    def name(self) -> str:
        return "context_assembly"

    @property
    def dependencies(self) -> list[str]:
        return ["rrf_merge"]

    def execute(self, inputs: dict[str, Any], context: NodeContext) -> str:
        hits: list[Hit] = inputs["rrf_merge"]
        max_chars = context.params.get(
            "max_context_chars", get_config()["rag"]["max_context_chars"]
        )
        parts, used = [], 0
        for hit in hits:
            c = hit.chunk
            block = f"--- [{c.doc_id}, p.{c.page}] ---\n{c.text}\n"
            if used + len(block) > max_chars:
                break
            parts.append(block)
            used += len(block)
        return "\n".join(parts)


class PromptFormatNode:
    """DAG Node: Format grounded prompt template with assembled context."""

    @property
    def name(self) -> str:
        return "prompt_format"

    @property
    def dependencies(self) -> list[str]:
        return ["context_assembly"]

    def execute(self, inputs: dict[str, Any], context: NodeContext) -> str:
        context_str = inputs["context_assembly"]
        question = inputs["question"]
        prompt_tmpl = resolve_path(get_config()["rag"]["prompt_path"]).read_text(encoding="utf-8")
        return prompt_tmpl.format(context=context_str, question=question)


class LLMSynthesisNode:
    """DAG Node: Complete answer with citations using configured LLM provider."""

    def __init__(self, provider: LLMProvider | None = None) -> None:
        self._provider = provider

    @property
    def name(self) -> str:
        return "llm_synthesis"

    @property
    def dependencies(self) -> list[str]:
        return ["prompt_format"]

    def execute(self, inputs: dict[str, Any], context: NodeContext) -> str:
        from docsense.llm.factory import get_provider

        provider = self._provider or context.params.get("provider") or get_provider()
        prompt = inputs["prompt_format"]
        max_tokens = get_config()["rag"]["max_answer_tokens"]
        system_msg = "You answer questions about documents precisely and always cite your sources."
        return provider.complete(prompt, system=system_msg, max_tokens=max_tokens)
