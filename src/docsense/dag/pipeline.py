"""Declarative DAG Architecture - Standard Document QA Pipeline Builder."""

from __future__ import annotations

from typing import Any

from docsense.dag.graph import DAGGraph
from docsense.dag.node import NodeContext
from docsense.dag.nodes import (
    BM25RetrievalNode,
    ContextAssemblyNode,
    DenseRetrievalNode,
    LLMSynthesisNode,
    PromptFormatNode,
    RRFMergeNode,
)
from docsense.llm.base import LLMProvider


def build_doc_qa_dag(
    provider: LLMProvider | None = None, candidates: int = 10, top_k: int = 5
) -> DAGGraph:
    """Construct the complete Declarative Document QA DAG."""
    dag = DAGGraph()
    dag.add_node(DenseRetrievalNode(candidates=candidates))
    dag.add_node(BM25RetrievalNode(candidates=candidates))
    dag.add_node(RRFMergeNode(top_k=top_k))
    dag.add_node(ContextAssemblyNode())
    dag.add_node(PromptFormatNode())
    dag.add_node(LLMSynthesisNode(provider=provider))
    return dag


def run_doc_qa_dag(
    question: str,
    provider: LLMProvider | None = None,
    top_k: int = 5,
    extra_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute the Document QA DAG for a query."""
    dag = build_doc_qa_dag(provider=provider, top_k=top_k)
    params = {"top_k": top_k}
    if provider:
        params["provider"] = provider
    if extra_params:
        params.update(extra_params)
    context = NodeContext(params=params)
    return dag.execute(initial_inputs={"question": question}, context=context)
