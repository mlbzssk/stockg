import contextvars
import json
from types import SimpleNamespace

import pytest

from stockg.domain import agent
from stockg.domain import session as session_module
from stockg.domain.session import SessionStore, get_current_session


def _tool_call(name: str, arguments: dict, call_id: str = "call-1") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
    }


def _usage(prompt_tokens: int = 10, completion_tokens: int = 5) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


def test_client_factories_require_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent, "DEEPSEEK_API_KEY", "")
    agent._get_client.cache_clear()
    agent._get_async_client.cache_clear()

    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        agent._get_client()
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        agent._get_async_client()


def test_industrial_agent_returns_structured_final_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SessionStore()
    monkeypatch.setattr(session_module, "session_store", store)
    monkeypatch.setattr(agent, "_get_client", lambda: object())
    calls: list[dict] = []

    def fake_stream(client, **kwargs):
        calls.append(kwargs)
        return (
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    _tool_call(
                        "submit_final_report",
                        {"investment_rating": "观望", "analysis_reason": "估值合理"},
                    )
                ],
            },
            _usage(),
            "tool_calls",
        )

    monkeypatch.setattr(agent, "stream_completion", fake_stream)

    def exercise():
        result = agent.run_industrial_agent(
            "分析 AAPL",
            thread_id="thread-1",
            market="us",
            max_iterations=2,
        )
        with pytest.raises(LookupError):
            get_current_session()
        return result

    result = contextvars.Context().run(exercise)

    assert result.status == "success"
    assert result.rating == "观望"
    assert result.reason == "估值合理"
    assert result.fetched_context == ["[submit_final_report 已提交最终报告。评级: 观望]"]
    assert result.cost == {"tool_tokens": 15, "cost_yuan": 0.0001}
    assert calls[0]["tool_choice"] == "auto"
    assert calls[0]["messages"][0]["role"] == "system"
    assert "本轮 market 参数：us" in calls[0]["messages"][1]["content"]


def test_industrial_agent_returns_failed_result_after_length_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SessionStore()
    monkeypatch.setattr(session_module, "session_store", store)
    monkeypatch.setattr(agent, "_get_client", lambda: object())
    monkeypatch.setattr(
        agent,
        "stream_completion",
        lambda client, **kwargs: (
            {"role": "assistant", "content": "partial", "tool_calls": None},
            _usage(),
            "length",
        ),
    )

    result = contextvars.Context().run(
        lambda: agent.run_industrial_agent(
            "分析 600519",
            thread_id="length-test",
            market="a",
            max_iterations=1,
        )
    )

    assert result.status == "failed"
    assert result.reason == "agent run out of iterations"
    assert result.cost == {"tool_tokens": 0, "cost_yuan": 0.0}
    messages = store.get("length-test").messages
    assert messages[-1]["content"] == "输出过长被截断，请直接调用工具，不要输出长文本。"


def test_industrial_agent_rejects_invalid_market_before_model_call() -> None:
    with pytest.raises(ValueError, match="market 必须是 a、us 或 auto"):
        agent.run_industrial_agent("分析", market="hk")


def test_get_stock_info_formats_supported_markets_and_skips_other_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = SimpleNamespace(
        json=lambda: {
            "QuotationCodeTable": {
                "Data": [
                    {"Code": "600519", "Name": "贵州茅台", "Classify": "AStock"},
                    {"Code": "AAPL", "Name": "Apple", "Classify": "UsStock"},
                    {"Code": "00700", "Name": "腾讯控股", "Classify": "HKStock"},
                    {"Code": "FUND", "Name": "基金", "Classify": "Fund"},
                ]
            }
        }
    )
    import requests

    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: response)

    result = agent.get_stock_info("科技")

    assert result == "找到:贵州茅台(600519), A股|Apple(AAPL), 美股|腾讯控股(00700), 港股"


def test_get_stock_info_returns_readable_message_when_no_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = SimpleNamespace(json=lambda: {"QuotationCodeTable": {"Data": []}})
    import requests

    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: response)

    assert agent.get_stock_info("missing") == "未找到匹配'missing'的股票，请确认名称或代码"
