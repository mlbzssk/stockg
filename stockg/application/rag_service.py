"""应用服务: 编排 RAG 的「入库」与「检索」用例。"""
from __future__ import annotations

from pathlib import Path

from stockg.domain import (
    DocumentLoader,
    Embedder,
    RetrievedContext,
    Retriever,
    TextSplitter,
    VectorStore,
)


class RagIngestionService:
    """用例: 把 PDF / 文本材料切分、向量化并写入向量库。"""

    def __init__(
        self,
        loader: DocumentLoader,
        splitter: TextSplitter,
        embedder: Embedder,
        store: VectorStore,
    ) -> None:
        self._loader = loader
        self._splitter = splitter
        self._embedder = embedder
        self._store = store

    def ingest_file(
        self, path: str, chunk_size: int = 400, chunk_overlap: int = 50
    ) -> int:
        doc = self._loader.load(path)
        chunks = self._splitter.split(doc, chunk_size, chunk_overlap)
        texts = [c.text for c in chunks]
        vectors = self._embedder.embed(texts)
        for chunk, vec in zip(chunks, vectors):
            chunk.embedding = vec
        self._store.add(chunks, self._embedder.name)
        return len(chunks)

    def ingest_path(
        self, path: str, chunk_size: int = 400, chunk_overlap: int = 50
    ) -> int:
        """接受文件或目录；目录会递归处理 .pdf / .txt / .md。"""
        p = Path(path)
        if p.is_dir():
            total = 0
            for f in sorted(p.rglob("*")):
                if f.suffix.lower() in {".pdf", ".txt", ".md"}:
                    total += self.ingest_file(str(f), chunk_size, chunk_overlap)
            return total
        return self.ingest_file(path, chunk_size, chunk_overlap)


class RagQueryService:
    """用例: 基于检索器做知识库问答（取回上下文）。"""

    def __init__(self, retriever: Retriever) -> None:
        self._retriever = retriever

    def retrieve(self, question: str, top_k: int = 3) -> list[RetrievedContext]:
        return self._retriever.retrieve(question, top_k)
