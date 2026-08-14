"""检索器：把 Embedder + VectorStore 组装成对外的检索能力。"""
from __future__ import annotations

from stockg.domain import Embedder, RetrievedContext, Retriever, VectorStore


class SimpleRetriever(Retriever):
    """轻量自研检索器：query 向量化后调用向量库做 top-k 召回。"""

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        query_prefix: str = "",
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._query_prefix = query_prefix

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedContext]:
        q = (self._query_prefix + query) if self._query_prefix else query
        vec = self._embedder.embed([q])[0]
        return self._store.search(vec, top_k)
