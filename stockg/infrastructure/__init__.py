"""基础设施层: 领域端口的具体适配器 (akshare/东财, DeepSeek 等)。

对外统一从 ``stockg.infrastructure`` 导入; 仓储实现依赖 akshare 等重库,
通过模块级 ``__getattr__`` 懒加载, 避免 import 本包时拉起重型依赖。
"""

from importlib import import_module
from typing import Any

from stockg.infrastructure.config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    FINNHUB_API_KEY,
    RAG_STORE_PATH,
)

_LAZY_EXPORTS = {
    "AkshareStockRepository": "akshare_stock_repository",
    "FinnhubStockRepository": "finnhub_stock_repository",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value


__all__ = [
    "AkshareStockRepository",
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_MODEL",
    "FINNHUB_API_KEY",
    "FinnhubStockRepository",
    "RAG_STORE_PATH",
]
