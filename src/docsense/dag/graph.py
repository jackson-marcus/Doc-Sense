"""Declarative DAG Architecture - Graph Runner & Topological Orchestrator.

Validates acyclicity, resolves dependencies, and runs nodes in topological order.
"""

from __future__ import annotations

import collections
import time
from typing import Any

from docsense.dag.node import DAGNode, NodeContext, NodeResult


class CyclicDependencyError(ValueError):
    """Raised when a circular dependency is detected in the DAG."""


class MissingDependencyError(KeyError):
    """Raised when a node depends on an unprovided or unregistered node."""


class DAGGraph:
    """Directed Acyclic Graph orchestrator for declarative pipeline execution."""

    def __init__(self) -> None:
        self._nodes: dict[str, DAGNode] = {}

    def add_node(self, node: DAGNode) -> DAGGraph:
        """Register a node in the DAG."""
        self._nodes[node.name] = node
        return self

    def topological_order(self) -> list[str]:
        """Compute the topological sort of registered nodes using Kahn's algorithm."""
        in_degree: dict[str, int] = {name: 0 for name in self._nodes}
        adj: dict[str, list[str]] = collections.defaultdict(list)

        for name, node in self._nodes.items():
            for dep in node.dependencies:
                if dep not in self._nodes:
                    raise MissingDependencyError(
                        f"Node '{name}' depends on unregistered node '{dep}'"
                    )
                adj[dep].append(name)
                in_degree[name] += 1

        queue = collections.deque([name for name, deg in in_degree.items() if deg == 0])
        order: list[str] = []

        while queue:
            curr = queue.popleft()
            order.append(curr)
            for neighbor in adj[curr]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(self._nodes):
            raise CyclicDependencyError("Circular dependency detected in pipeline DAG")

        return order

    def execute(
        self, initial_inputs: dict[str, Any] | None = None, context: NodeContext | None = None
    ) -> dict[str, NodeResult]:
        """Execute all nodes in topological dependency order."""
        context = context or NodeContext()
        order = self.topological_order()
        results: dict[str, NodeResult] = {}
        intermediates: dict[str, Any] = dict(initial_inputs or {})

        for name in order:
            node = self._nodes[name]
            node_inputs = {
                dep: intermediates[dep] for dep in node.dependencies if dep in intermediates
            }
            for k, v in (initial_inputs or {}).items():
                if k not in node_inputs:
                    node_inputs[k] = v

            start_t = time.perf_counter()
            out = node.execute(node_inputs, context)
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0

            node_result = NodeResult(
                node_name=name,
                data=out,
                execution_time_ms=round(elapsed_ms, 3),
            )
            results[name] = node_result
            intermediates[name] = out

        return results
