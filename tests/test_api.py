import asyncio
from uuid import UUID

import pytest
from fastapi import HTTPException

from stockg.domain.entities import AgentRunResult
from stockg.interfaces import api


def test_health_returns_ok() -> None:
    assert api.health() == {"status": "ok"}


def test_chat_generates_session_id_and_maps_agent_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, str]] = []

    async def fake_run(message: str, *, market: str, thread_id: str) -> AgentRunResult:
        calls.append((message, market, thread_id))
        return AgentRunResult(
            status="success",
            rating="观望",
            reason="风险收益均衡",
            fetched_context=["行情", "新闻"],
            cost={"tool_tokens": 100, "cost_yuan": 0.01},
        )

    monkeypatch.setattr(api, "run_industrial_agent_async", fake_run)

    response = asyncio.run(api.chat(api.ChatRequest(message="分析 AAPL", market="us")))

    UUID(response.session_id)
    assert calls == [("分析 AAPL", "us", response.session_id)]
    assert response.status == "success"
    assert response.message == "风险收益均衡"
    assert response.rating == "观望"
    assert response.fetched_context == ["行情", "新闻"]
    assert response.cost == {"tool_tokens": 100, "cost_yuan": 0.01}


def test_chat_reuses_provided_session_id(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run(message: str, *, market: str, thread_id: str) -> AgentRunResult:
        assert thread_id == "existing-session"
        return AgentRunResult("success", "买入", "理由", [], {})

    monkeypatch.setattr(api, "run_industrial_agent_async", fake_run)

    response = asyncio.run(
        api.chat(
            api.ChatRequest(
                message="继续分析",
                market="a",
                session_id="existing-session",
            )
        )
    )

    assert response.session_id == "existing-session"


def test_chat_hides_internal_agent_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fail_run(message: str, *, market: str, thread_id: str) -> AgentRunResult:
        raise RuntimeError("secret upstream detail")

    monkeypatch.setattr(api, "run_industrial_agent_async", fail_run)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(api.chat(api.ChatRequest(message="分析", market="auto")))

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Agent 处理失败"
    assert "secret upstream detail" not in exc_info.value.detail
