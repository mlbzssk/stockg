"""领域层: 实体与端口(接口), 不依赖任何外部框架或 IO。

对外统一从 ``stockg.domain`` 导入, 无需感知内部模块名。
"""

from importlib import import_module
from typing import Any

from stockg.domain.entities import (
    AnalysisReport,
    Chunk,
    Document,
    RetrievedContext,
    StockNews,
    StockQuote,
    StockSnapshot,
)
from stockg.domain.ports import (
    DocumentLoader,
    Embedder,
    Retriever,
    StockDataRepository,
    TextSplitter,
    VectorStore,
)

# Agent 入口依赖 openai / langgraph 等重依赖, 懒加载避免 import 本包时拉起
_LAZY_EXPORTS = {
    "run_industrial_agent": "agent",
    "run_research_agent": "research_agent",
    "run_industrial_agent_async": "agent",
    "run_research_agent_async": "research_agent",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value


__all__ = [
    "AnalysisReport",
    "Chunk",
    "Document",
    "DocumentLoader",
    "Embedder",
    "RetrievedContext",
    "Retriever",
    "StockDataRepository",
    "StockNews",
    "StockQuote",
    "StockSnapshot",
    "TextSplitter",
    "VectorStore",
    "run_industrial_agent",
    "run_research_agent",
    "run_industrial_agent_async",
    "run_research_agent_async",
]
