import asyncio
import contextvars

import pytest

from stockg.domain import research_agent
from stockg.domain.entities import RetrievedContext, StockSnapshot
from stockg.domain.session import (
    SessionContext,
    get_current_session,
    reset_current_session,
    set_current_session,
)


class FakeRepository:
    def __init__(self, snapshot: StockSnapshot) -> None:
        self.snapshot = snapshot
        self.calls: list[str] = []

    async def fetch_snapshot_async(self, symbol: str) -> StockSnapshot:
        self.calls.append(symbol)
        await asyncio.sleep(0)
        return self.snapshot


class FakeRetriever:
    def __init__(self, contexts: list[RetrievedContext]) -> None:
        self.contexts = contexts
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int) -> list[RetrievedContext]:
        self.calls.append((query, top_k))
        return self.contexts


class FakeGraph:
    def __init__(self, result: dict[str, object]) -> None:
        self.result = result
        self.states: list[dict[str, object]] = []

    async def ainvoke(self, state: dict[str, object]) -> dict[str, object]:
        self.states.append(state)
        return self.result


def test_repository_for_selects_market_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    a_repository = object()
    us_repository = object()
    monkeypatch.setattr(research_agent, "sh_active_repository", a_repository)
    monkeypatch.setattr(research_agent, "us_active_repository", us_repository)

    assert research_agent._repository_for("a") is a_repository
    assert research_agent._repository_for("us") is us_repository
    with pytest.raises(ValueError, match="market 必须是 a 或 us"):
        research_agent._repository_for("hk")


def test_fetch_snapshot_uses_market_scoped_cache_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = StockSnapshot("600519", "贵州茅台", 1_500.0, 1.2, ["news"])
    repository = FakeRepository(snapshot)
    monkeypatch.setattr(research_agent, "sh_active_repository", repository)

    async def scenario() -> None:
        context = SessionContext(thread_id="cache-test")
        token = set_current_session(context)
        try:
            first, second = await asyncio.gather(
                research_agent._fetch_snapshot_async("600519", "a"),
                research_agent._fetch_snapshot_async("600519", "a"),
            )
        finally:
            reset_current_session(token)
        assert first is snapshot
        assert second is snapshot

    asyncio.run(scenario())

    assert repository.calls == ["600519"]


def test_retrieve_knowledge_formats_contexts_and_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retriever = FakeRetriever(
        [
            RetrievedContext("第一段", "a.pdf", 0.91234),
            RetrievedContext("第二段", "b.md", 0.8),
        ]
    )
    monkeypatch.setattr(research_agent, "_get_rag_retriever", lambda: retriever)

    result = research_agent._retrieve_knowledge("AAPL", top_k=2)

    assert result == (
        "知识库检索结果:\n"
        "[来源: a.pdf | 相似度: 0.912]\n第一段\n\n"
        "[来源: b.md | 相似度: 0.800]\n第二段"
    )
    assert retriever.calls == [
        ("从本地知识库搜索股票AAPL相关的知识，包括过往研报、公告、新闻等", 2)
    ]


def test_retrieve_knowledge_returns_import_hint_when_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        research_agent,
        "_get_rag_retriever",
        lambda: FakeRetriever([]),
    )

    result = research_agent._retrieve_knowledge("600519", top_k=3)

    assert "知识库为空" in result
    assert "ingest" in result


def test_fetch_market_data_builds_price_and_news(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = StockSnapshot("600519", "贵州茅台", 1_500.0, -0.5, ["新闻一", "新闻二"])

    async def fake_fetch(ticker: str, market: str) -> StockSnapshot:
        assert (ticker, market) == ("600519", "a")
        return snapshot

    monkeypatch.setattr(research_agent, "_fetch_snapshot_async", fake_fetch)

    result = asyncio.run(
        research_agent.fetch_market_data(
            {"ticker": "600519", "market": "a", "rag_top_k": 3, "errors": []}
        )
    )

    assert result == {
        "price": "600519（贵州茅台）当前最新股价: 1500.0 (涨跌幅: -0.5%)",
        "news": "600519 最新相关新闻:\n- 新闻一\n- 新闻二",
        "errors": [],
    }


def test_fetch_market_data_degrades_to_structured_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_fetch(ticker: str, market: str) -> StockSnapshot:
        raise RuntimeError("upstream unavailable")

    monkeypatch.setattr(research_agent, "_fetch_snapshot_async", fail_fetch)

    result = asyncio.run(
        research_agent.fetch_market_data(
            {"ticker": "AAPL", "market": "us", "rag_top_k": 3, "errors": []}
        )
    )

    assert result["errors"] == ["AAPL 行情与新闻获取失败: upstream unavailable"]
    assert "upstream unavailable" in str(result["price"])
    assert "快照不可用" in str(result["news"])


def test_build_result_includes_defaults_and_warnings() -> None:
    result = research_agent.build_result(
        {
            "ticker": "AAPL",
            "market": "us",
            "rag_top_k": 3,
            "errors": ["行情失败", "RAG 失败"],
            "price": "220.0",
        }
    )

    assert result["research_msg"] == (
        "\n当前股价: 220.0"
        "\n最新新闻: 未获取"
        "\n知识库: 未获取"
        "\n资料收集告警: 行情失败 | RAG 失败"
    )


@pytest.mark.parametrize(
    ("ticker", "market", "top_k", "message"),
    [
        ("AAPL", "auto", 3, "market 必须是 a 或 us"),
        ("AAPL", "us", 0, "rag_top_k 必须大于 0"),
        ("   ", "us", 3, "ticker 不能为空"),
    ],
)
def test_run_research_agent_async_validates_inputs(
    ticker: str,
    market: str,
    top_k: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        asyncio.run(research_agent.run_research_agent_async(ticker, market, top_k))


def test_run_research_agent_async_normalizes_input_and_uses_fallback_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = FakeGraph({"research_msg": "final", "errors": []})
    monkeypatch.setattr(research_agent, "research_graph", graph)

    def exercise() -> None:
        result = asyncio.run(
            research_agent.run_research_agent_async("  aapl  ", "us", rag_top_k=4)
        )
        assert result == "final"
        with pytest.raises(LookupError):
            get_current_session()

    contextvars.Context().run(exercise)

    assert graph.states == [
        {"ticker": "AAPL", "market": "us", "rag_top_k": 4, "errors": []}
    ]


def test_sync_entry_rejects_running_event_loop() -> None:
    async def scenario() -> None:
        with pytest.raises(RuntimeError, match="请改用 await"):
            research_agent.run_research_agent("AAPL", "us")

    asyncio.run(scenario())
