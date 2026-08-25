"""Declarative DAG Architecture - Node and Result Definitions.

Pure, typed node contracts for directed acyclic graph execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class NodeContext:
    """Execution context passed into DAG nodes."""

    params: dict[str, Any] = field(default_factory=dict)
    cache: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NodeResult:
    """Output result produced by a DAG node."""

    node_name: str
    data: Any
    execution_time_ms: float = 0.0
    cached: bool = False


class DAGNode(Protocol):
    """Protocol for a single executable stage in the DAG."""

    @property
    def name(self) -> str: ...

    @property
    def dependencies(self) -> list[str]: ...

    def execute(self, inputs: dict[str, Any], context: NodeContext) -> Any: ...
