"""向量库：基于 Chroma 的持久化实现（HNSW + 余弦相似度）。"""

from __future__ import annotations

from pathlib import Path

from chromadb import PersistentClient
from chromadb.config import Settings

from stockg.domain import Chunk, RetrievedContext, VectorStore

DEFAULT_COLLECTION = "stockg_rag"


class ChromaVectorStore(VectorStore):
    """用 Chroma 做向量存储，本地持久化 + 余弦检索。

    与 VectorStore 端口对齐：add 接收已向量化的 Chunk，search 接收查询向量。
    模型名写入 collection metadata，避免不同 Embedder 的向量混用。
    """

    def __init__(self, path: str, collection_name: str = DEFAULT_COLLECTION) -> None:
        self._path = Path(path)
        self._path.mkdir(parents=True, exist_ok=True)
        # 持久化客户端：数据落盘到 path 目录；关闭遥测避免联网
        self._client = PersistentClient(
            path=str(self._path),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: list[Chunk], model_name: str) -> None:
        existing = self._collection.metadata or {}
        if existing.get("model") and existing["model"] != model_name:
            raise RuntimeError(
                f"向量库由模型「{existing['model']}」生成，与当前「{model_name}」不一致。"
                f"请删除 {self._path} 后重新入库，避免向量空间不匹配。"
            )
        if not chunks:
            return
        ids = [f"{c.source}__{i}" for i, c in enumerate(chunks)]
        embeddings = [c.embedding for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [{"source": c.source} for c in chunks]
        # upsert：重复入库同名材料时覆盖而非报错
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        self._collection.modify(metadata={"model": model_name})

    def search(self, query_vec: list[float], top_k: int) -> list[RetrievedContext]:
        total = self._collection.count()
        if total == 0:
            return []
        top_k = min(top_k, total)
        res = self._collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        results: list[RetrievedContext] = []
        for doc, meta, dist in zip(docs, metas, dists):
            # cosine 空间下 distance = 1 - cosine_similarity
            score = 1.0 - float(dist)
            results.append(
                RetrievedContext(
                    text=doc,
                    source=(meta or {}).get("source", "unknown"),
                    score=score,
                )
            )
        return results

    def count(self) -> int:
        return self._collection.count()
