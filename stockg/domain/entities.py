"""领域实体 / 值对象。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class StockQuote:
    """单只股票的实时行情。"""

    symbol: str
    name: str
    price: float
    change_pct: float


@dataclass
class StockNews:
    """单只股票的相关新闻。"""

    symbol: str
    titles: List[str] = field(default_factory=list)


@dataclass
class StockSnapshot:
    """某一时刻某只股票的行情 + 新闻聚合, 作为分析服务的输入。"""

    symbol: str
    name: str
    price: float
    change_pct: float
    news: List[str] = field(default_factory=list)


@dataclass
class AnalysisReport:
    """大模型产出的分析报告。"""

    symbol: str
    name: str
    content: str


# ==========================================
# RAG 相关值对象
# ==========================================
@dataclass
class Document:
    """一份待入库的原始文档（PDF / 文本）。"""

    source: str
    text: str


@dataclass
class Chunk:
    """切分后的文本块，已（或待）向量化。"""

    text: str
    source: str
    embedding: Optional[List[float]] = None


@dataclass
class RetrievedContext:
    """从向量库召回的上下文片段。"""

    text: str
    source: str
    score: float


@dataclass
class AgentRunResult:
    """Agent 一次运行的最终结构化结果，供上层调用与评测消费。"""

    status: str
    rating: str
    reason: str
    fetched_context: list[str]
    cost: dict
