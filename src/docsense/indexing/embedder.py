"""Embedding wrapper: fastembed (ONNX runtime) — no torch dependency.

ONNX MiniLM is faster than the torch pipeline on CPU, keeps the Docker image
~700MB smaller, and sidesteps torch's MSVC-runtime requirements on Windows.
"""

from __future__ import annotations

import functools

import numpy as np

from docsense.settings import get_config


@functools.lru_cache(maxsize=1)
def _model():
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=get_config()["indexing"]["embedding_model"])


def embed_texts(texts: list[str]) -> list[list[float]]:
    out = []
    for vec in _model().embed(texts):
        arr = np.asarray(vec, dtype=np.float32)
        norm = np.linalg.norm(arr)
        out.append((arr / norm if norm else arr).tolist())
    return out


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
