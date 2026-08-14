import asyncio

from stockg.domain.entities import AgentRunResult, Chunk, StockNews, StockSnapshot
from stockg.domain.ports import StockDataRepository


class FakeStockRepository(StockDataRepository):
    def __init__(self, snapshot: StockSnapshot) -> None:
        self.snapshot = snapshot
        self.requested_symbols: list[str] = []

    def fetch_snapshot(self, symbol: str) -> StockSnapshot:
        self.requested_symbols.append(symbol)
        return self.snapshot


def test_entity_collection_defaults_are_not_shared() -> None:
    first_news = StockNews(symbol="600519")
    second_news = StockNews(symbol="000001")
    first_snapshot = StockSnapshot("600519", "贵州茅台", 1.0, 0.0)
    second_snapshot = StockSnapshot("000001", "平安银行", 2.0, 0.0)

    first_news.titles.append("新闻")
    first_snapshot.news.append("快讯")

    assert second_news.titles == []
    assert second_snapshot.news == []


def test_chunk_embedding_defaults_to_none() -> None:
    chunk = Chunk(text="正文", source="report.md")

    assert chunk.embedding is None


def test_agent_run_result_preserves_structured_fields() -> None:
    result = AgentRunResult(
        status="success",
        rating="观望",
        reason="估值偏高",
        fetched_context=["行情"],
        cost={"tool_tokens": 20, "cost_yuan": 0.01},
    )

    assert result.rating == "观望"
    assert result.fetched_context == ["行情"]
    assert result.cost["tool_tokens"] == 20


def test_default_async_repository_adapter_calls_sync_implementation() -> None:
    snapshot = StockSnapshot("AAPL", "Apple", 220.0, 1.5, ["news"])
    repository = FakeStockRepository(snapshot)

    result = asyncio.run(repository.fetch_snapshot_async("AAPL"))

    assert result is snapshot
    assert repository.requested_symbols == ["AAPL"]
