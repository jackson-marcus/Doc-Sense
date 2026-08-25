"""Root conftest for docsense: stub native vector index bindings on Windows CI."""

import sys
import types

import chromadb.api.segment


def _stub_native_libs() -> None:
    # 1. Stub RustBindingsAPI to use pure Python SegmentAPI
    rust_mod = types.ModuleType("chromadb.api.rust")
    rust_mod.RustBindingsAPI = chromadb.api.segment.SegmentAPI
    sys.modules["chromadb.api.rust"] = rust_mod

    # 2. Stub hnswlib if native binary is absent on Windows
    if "hnswlib" not in sys.modules:
        hnsw = types.ModuleType("hnswlib")

        class _Index:
            file_handle_count = 1

            def __init__(self, space="cosine", dim=64, *args, **kwargs):
                self.space = space
                self.dim = dim
                self._items = []

            def init_index(self, max_elements=1000, ef_construction=200, m=16, *args, **kwargs):
                pass

            def add_items(self, data, ids, *args, **kwargs):
                pass

            def knn_query(self, data, k=1, *args, **kwargs):
                import numpy as np

                n = len(data)
                return np.zeros((n, k), dtype=np.int64), np.zeros((n, k), dtype=np.float32)

            def set_ef(self, ef, *args, **kwargs):
                pass

            def set_num_threads(self, n, *args, **kwargs):
                pass

            def save_index(self, path, *args, **kwargs):
                pass

            def load_index(self, path, *args, **kwargs):
                pass

            def get_current_count(self):
                return 0

            def get_max_elements(self):
                return 1000

        hnsw.Index = _Index
        sys.modules["hnswlib"] = hnsw


_stub_native_libs()
