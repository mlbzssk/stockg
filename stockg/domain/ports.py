"""领域端口(抽象接口 / DIP 依赖倒置点)。

应用层只依赖这些抽象, 具体实现由 infrastructure 层提供并通过依赖注入传入。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from stockg.domain.entities import StockSnapshot, Document, Chunk, RetrievedContext


class StockDataRepository(ABC):
    """负责获取股票行情与新闻的端口（A股=akshare / 美股=finnhub 各实现一份）。"""

    @abstractmethod
    def fetch_snapshot(self, symbol: str) -> StockSnapshot:
        """抓取指定代码的实时行情 + 新闻快照。"""
        ...
    async def fetch_snapshot_async(self, symbol: str) -> StockSnapshot:
        import asyncio
        return await asyncio.to_thread(self.fetch_snapshot, symbol)


# ==========================================
# RAG 端口
# ==========================================
class DocumentLoader(ABC):
    """加载原始文档（PDF / 文本）为纯文本。"""

    @abstractmethod
    def load(self, path: str) -> Document:
        ...


class TextSplitter(ABC):
    """把文档切分为适合检索的块。"""

    @abstractmethod
    def split(
        self, doc: Document, chunk_size: int = 400, chunk_overlap: int = 50
    ) -> List[Chunk]:
        ...


class Embedder(ABC):
    """文本向量化。"""

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        ...

    @property
    def name(self) -> str:
        return "unknown"


class VectorStore(ABC):
    """向量库的抽象：写入块、按向量召回、统计数量。"""

    @abstractmethod
    def add(self, chunks: List[Chunk], model_name: str) -> None:
        ...

    @abstractmethod
    def search(self, query_vec: List[float], top_k: int) -> List[RetrievedContext]:
        ...

    @abstractmethod
    def count(self) -> int:
        ...


class Retriever(ABC):
    """对外提供「自然语言查询 -> 相关上下文」的检索能力。"""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievedContext]:
        ...

