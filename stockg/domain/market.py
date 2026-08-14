"""股票代码与市场的确定性解析规则。"""

from __future__ import annotations

import re
from typing import Literal

Market = Literal["a", "us"]
RequestedMarket = Literal["a", "us", "auto"]

_A_SHARE_TICKER = re.compile(r"^[0368]\d{5}$")
_US_SHARE_TICKER = re.compile(r"^[A-Z][A-Z0-9.-]{0,9}$")


def detect_market_from_ticker(ticker: str) -> Market | None:
    """根据标准股票代码识别市场，无法确定时返回 ``None``。"""
    normalized_ticker = ticker.strip().upper()
    if _A_SHARE_TICKER.fullmatch(normalized_ticker):
        return "a"
    if _US_SHARE_TICKER.fullmatch(normalized_ticker):
        return "us"
    return None


def resolve_market(
    requested_market: RequestedMarket,
    ticker: str,
    suggested_market: object,
) -> Market:
    """按调用方、标准代码、模型建议的顺序确定市场。"""
    if requested_market == "a":
        return "a"
    if requested_market == "us":
        return "us"

    detected_market = detect_market_from_ticker(ticker)
    if detected_market is not None:
        return detected_market

    if suggested_market == "a":
        return "a"
    if suggested_market == "us":
        return "us"
    raise ValueError("无法根据股票代码或模型结果确定市场")
