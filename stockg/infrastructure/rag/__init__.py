"""RAG 基础设施实现：PDF 加载、切分、向量化、向量库、检索。

各模块依赖 chromadb / numpy / torch 等重库, 懒加载按需引入。
"""

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "BGE_QUERY_PREFIX": "embedder",
    "build_embedder": "embedder",
    "ChromaVectorStore": "chroma_store",
    "RecursiveCharacterSplitter": "chunker",
    "build_loader": "pdf_loader",
    "SimpleRetriever": "retriever",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value


__all__ = [
    "BGE_QUERY_PREFIX",
    "ChromaVectorStore",
    "RecursiveCharacterSplitter",
    "SimpleRetriever",
    "build_embedder",
    "build_loader",
]
