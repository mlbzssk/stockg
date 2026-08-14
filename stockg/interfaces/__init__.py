"""接口层: 对外入口 (CLI / API 等)。

cli 间接依赖 RAG 重库, 懒加载避免 import 本包时拉起。
"""

from importlib import import_module
from typing import Any


def __getattr__(name: str) -> Any:
    if name != "main":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.cli"), name)
    globals()[name] = value
    return value


__all__ = ["main"]
