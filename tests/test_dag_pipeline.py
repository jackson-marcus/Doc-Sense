"""Unit tests for the Declarative DAG Pipeline Architecture."""

import pytest

from docsense.dag.graph import CyclicDependencyError, DAGGraph, MissingDependencyError
from docsense.dag.pipeline import run_doc_qa_dag


class MockNode:
    def __init__(self, name: str, dependencies: list[str], compute_fn):
        self._name = name
        self._deps = dependencies
        self._fn = compute_fn

    @property
    def name(self) -> str:
        return self._name

    @property
    def dependencies(self) -> list[str]:
        return self._deps

    def execute(self, inputs, context):
        return self._fn(inputs, context)


def test_dag_topological_sort():
    dag = DAGGraph()
    dag.add_node(MockNode("c", ["a", "b"], lambda inp, ctx: inp["a"] + inp["b"]))
    dag.add_node(MockNode("a", [], lambda inp, ctx: 10))
    dag.add_node(MockNode("b", ["a"], lambda inp, ctx: inp["a"] * 2))

    order = dag.topological_order()
    assert order == ["a", "b", "c"]

    res = dag.execute()
    assert res["a"].data == 10
    assert res["b"].data == 20
    assert res["c"].data == 30


def test_dag_detects_cycle():
    dag = DAGGraph()
    dag.add_node(MockNode("a", ["b"], lambda inp, ctx: None))
    dag.add_node(MockNode("b", ["a"], lambda inp, ctx: None))

    with pytest.raises(CyclicDependencyError):
        dag.topological_order()


def test_dag_detects_missing_dependency():
    dag = DAGGraph()
    dag.add_node(MockNode("a", ["non_existent"], lambda inp, ctx: None))

    with pytest.raises(MissingDependencyError):
        dag.topological_order()


def test_doc_qa_dag_execution(fake_llm):
    results = run_doc_qa_dag(
        question="What are the warranty terms?",
        provider=fake_llm,
        top_k=2,
    )
    assert "dense_retrieval" in results
    assert "bm25_retrieval" in results
    assert "rrf_merge" in results
    assert "context_assembly" in results
    assert "prompt_format" in results
    assert "llm_synthesis" in results
    assert len(results["llm_synthesis"].data) > 0
