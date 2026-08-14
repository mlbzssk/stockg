from types import SimpleNamespace

import pytest

from stockg.infrastructure import akshare_stock_repository as akshare_repository
from stockg.infrastructure import finnhub_stock_repository as finnhub_repository


def test_a_share_symbol_helpers_select_exchange_prefixes() -> None:
    assert akshare_repository._to_secid("600519") == "1.600519"
    assert akshare_repository._to_secid("000001") == "0.000001"
    assert akshare_repository._to_sina_prefix("600519") == "sh"
    assert akshare_repository._to_sina_prefix("300750") == "sz"


def test_sina_quote_parser_calculates_change_percentage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = SimpleNamespace(
        text='var hq_str_sh600519="贵州茅台,100.00,0,105.00,0";',
        encoding=None,
    )
    monkeypatch.setattr(
        akshare_repository.requests, "get", lambda *args, **kwargs: response
    )

    result = akshare_repository.AkshareStockRepository()._fetch_from_sina("600519")

    assert result == ("贵州茅台", 105.0, 5.0)
    assert response.encoding == "gbk"


def test_sina_quote_parser_rejects_empty_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = SimpleNamespace(text='var hq_str_sh600519="";', encoding=None)
    monkeypatch.setattr(
        akshare_repository.requests, "get", lambda *args, **kwargs: response
    )

    with pytest.raises(RuntimeError, match="未返回 600519"):
        akshare_repository.AkshareStockRepository()._fetch_from_sina("600519")


def test_akshare_repository_falls_back_to_eastmoney_and_returns_news(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = akshare_repository.AkshareStockRepository()
    monkeypatch.setattr(
        repository,
        "_fetch_from_sina",
        lambda symbol: (_ for _ in ()).throw(RuntimeError("sina failed")),
    )
    monkeypatch.setattr(
        repository,
        "_fetch_from_eastmoney",
        lambda symbol: ("贵州茅台", 1500.0, 1.25),
    )
    monkeypatch.setattr(repository, "_fetch_news", lambda symbol, name: ["新闻一"])

    snapshot = repository.fetch_snapshot("600519")

    assert snapshot.symbol == "600519"
    assert snapshot.name == "贵州茅台"
    assert snapshot.price == 1500.0
    assert snapshot.change_pct == 1.25
    assert snapshot.news == ["新闻一"]


def test_finnhub_repository_returns_degraded_snapshot_on_quote_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(path: str, params: dict) -> dict:
        raise RuntimeError("network down")

    monkeypatch.setattr(finnhub_repository, "_finnhub_get", fail)

    snapshot = finnhub_repository.FinnhubStockRepository().fetch_snapshot("AAPL")

    assert snapshot.price == 0.0
    assert snapshot.change_pct == 0.0
    assert snapshot.news == ["行情获取失败: network down"]


def test_finnhub_repository_returns_no_data_message_for_zero_price(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        finnhub_repository,
        "_finnhub_get",
        lambda path, params: {"c": 0.0, "dp": 0.0},
    )

    snapshot = finnhub_repository.FinnhubStockRepository().fetch_snapshot("UNKNOWN")

    assert snapshot.price == 0.0
    assert "暂无行情数据" in snapshot.news[0]


def test_finnhub_repository_limits_news_and_ignores_missing_headline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict]] = []

    def fake_get(path: str, params: dict):
        calls.append((path, params))
        if path == "quote":
            return {"c": 220, "dp": 1.5}
        return [{"headline": f"news-{index}"} for index in range(20)] + [{}]

    monkeypatch.setattr(finnhub_repository, "_finnhub_get", fake_get)

    snapshot = finnhub_repository.FinnhubStockRepository().fetch_snapshot("AAPL")

    assert snapshot.price == 220.0
    assert snapshot.change_pct == 1.5
    assert snapshot.news == [f"news-{index}" for index in range(15)]
    assert calls[0] == ("quote", {"symbol": "AAPL"})
    assert calls[1][0] == "company-news"
    assert calls[1][1]["symbol"] == "AAPL"
