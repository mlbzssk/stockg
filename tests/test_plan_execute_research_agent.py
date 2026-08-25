import asyncio
import contextvars

import pytest

from stockg.domain import plan_execute_research_agent as agent
from stockg.domain.session import get_current_session


def _state() -> agent.PlanExecuteResearchState:
    return {
        "ticker": "AAPL",
        "market": "us",
        "objective": "收集经营风险和潜在催化剂",
        "rag_top_k": 3,
        "plan": [],
        "observations": [],
        "errors": [],
        "execution_round": 0,
        "is_complete": False,
    }


def test_parse_steps_accepts_supported_tools() -> None:
    steps = agent._parse_steps(
        {
            "steps": [
                {
                    "step_id": "market",
                    "tool": "fetch_market_data",
                    "instruction": "获取行情",
                },
                {
                    "step_id": "risk",
                    "tool": "search_knowledge",
                    "instruction": "检索经营风险",
                },
            ]
        }
    )

    assert [step["tool"] for step in steps] == [
        "fetch_market_data",
        "search_knowledge",
    ]


def test_parse_steps_rejects_unknown_tool() -> None:
    with pytest.raises(ValueError, match="tool 必须是"):
        agent._parse_steps(
            {
                "steps": [
                    {
                        "step_id": "invalid",
                        "tool": "run_command",
                        "instruction": "执行命令",
                    }
                ]
            }
        )


def test_planner_falls_back_when_model_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_request(*args, **kwargs):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(agent, "_request_json", fail_request)

    result = asyncio.run(agent.planner(_state()))

    assert result["is_complete"] is False
    assert [step["tool"] for step in result["plan"]] == [
        "fetch_market_data",
        "search_knowledge",
    ]


def test_executor_combines_parallel_step_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_fetch(state):
        return {"price": "价格资料", "news": "新闻资料", "errors": []}

    monkeypatch.setattr(agent, "fetch_market_data", fake_fetch)
    monkeypatch.setattr(
        agent,
        "search_knowledge",
        lambda query, top_k: f"知识资料:{query}:{top_k}",
    )
    state = _state()
    state["plan"] = [
        agent._market_step(),
        agent._knowledge_step("检索风险", "risk"),
        agent._knowledge_step("检索催化剂", "catalyst"),
    ]

    result = asyncio.run(agent.executor(state))

    assert result["price"] == "价格资料"
    assert result["news"] == "新闻资料"
    assert "检索风险" in result["knowledge_base"]
    assert "检索催化剂" in result["knowledge_base"]
    assert result["execution_round"] == 1
    assert len(result["observations"]) == 3


def test_replanner_stops_at_maximum_execution_rounds() -> None:
    state = _state()
    state["execution_round"] = agent.MAX_EXECUTION_ROUNDS

    result = asyncio.run(agent.replanner(state))

    assert result == {
        "plan": [],
        "is_complete": True,
        "completion_reason": "已达到最大执行轮数",
    }


def test_async_entry_normalizes_input_and_cleans_fallback_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeGraph:
        async def ainvoke(self, state, config):
            captured["state"] = state
            captured["config"] = config
            return {**state, "research_msg": "完成"}

    monkeypatch.setattr(agent, "plan_execute_research_graph", FakeGraph())

    def exercise() -> None:
        result = asyncio.run(
            agent.run_plan_execute_research_agent_async(
                "  aapl  ",
                "us",
                objective="  检索风险  ",
                rag_top_k=2,
            )
        )
        assert result == "完成"
        with pytest.raises(LookupError):
            get_current_session()

    contextvars.Context().run(exercise)

    state = captured["state"]
    assert state["ticker"] == "AAPL"
    assert state["objective"] == "检索风险"
    assert state["rag_top_k"] == 2
    assert captured["config"] == {"recursion_limit": 12}


def test_main_runs_standalone_with_detected_market(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[tuple[str, str, str, int]] = []

    def fake_run(
        ticker: str,
        market: str,
        objective: str,
        rag_top_k: int,
    ) -> str:
        calls.append((ticker, market, objective, rag_top_k))
        return "独立流程完成"

    monkeypatch.setattr(agent, "run_plan_execute_research_agent", fake_run)

    agent.main(["aapl", "--objective", "检索风险", "--rag-top-k", "2"])

    assert calls == [("aapl", "us", "检索风险", 2)]
    assert capsys.readouterr().out == "独立流程完成\n"
