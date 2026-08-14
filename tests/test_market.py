"""股票市场解析回归测试。"""

import pytest

from stockg.domain.market import detect_market_from_ticker, resolve_market


@pytest.mark.parametrize("ticker", ["601208", "600519", "000001", "300750", "688981"])
def test_detect_market_identifies_a_share_tickers(ticker: str) -> None:
    assert detect_market_from_ticker(ticker) == "a"


@pytest.mark.parametrize("ticker", ["AAPL", "TSLA", "BRK.B"])
def test_detect_market_identifies_us_share_tickers(ticker: str) -> None:
    assert detect_market_from_ticker(ticker) == "us"


def test_auto_market_overrides_incorrect_model_market_for_a_share() -> None:
    assert resolve_market("auto", "601208", "us") == "a"


def test_explicit_market_has_highest_priority() -> None:
    assert resolve_market("a", "AAPL", "us") == "a"


def test_ambiguous_ticker_uses_valid_model_market() -> None:
    assert resolve_market("auto", "贵州茅台", "a") == "a"


def test_ambiguous_ticker_rejects_invalid_model_market() -> None:
    with pytest.raises(ValueError, match="无法.*确定市场"):
        resolve_market("auto", "贵州茅台", "invalid")
