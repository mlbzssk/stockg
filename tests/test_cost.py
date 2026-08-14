from types import SimpleNamespace

import pytest

from stockg.domain.cost import CostBudgetExceeded, CostTracker


def _usage(
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def test_record_ignores_missing_usage() -> None:
    tracker = CostTracker()

    tracker.record(None, label="main#1")

    assert tracker.total_tokens == 0
    assert tracker.total_cost_yuan == 0.0
    assert tracker.calls == []


def test_record_accumulates_tokens_cost_and_call_details() -> None:
    tracker = CostTracker(max_total_tokens=10_000, max_total_cost_yuan=10.0)

    tracker.record(_usage(1_000, 500, 1_500), label="main#1")
    tracker.record(_usage(200, 100), label="main#2")

    assert tracker.prompt_tokens == 1_200
    assert tracker.completion_tokens == 600
    assert tracker.total_tokens == 1_800
    assert tracker.total_cost_yuan == pytest.approx(0.0072)
    assert tracker.calls == [
        {
            "label": "main#1",
            "prompt_tokens": 1_000,
            "completion_tokens": 500,
            "total_tokens": 1_500,
            "cost_yuan": 0.006,
        },
        {
            "label": "main#2",
            "prompt_tokens": 200,
            "completion_tokens": 100,
            "total_tokens": 300,
            "cost_yuan": 0.0012,
        },
    ]


def test_record_raises_after_token_budget_is_exceeded() -> None:
    tracker = CostTracker(max_total_tokens=10, max_total_cost_yuan=10.0)

    tracker.record(_usage(5, 5, 10), label="within-budget")

    with pytest.raises(CostBudgetExceeded, match="Total tokens exceeded 10"):
        tracker.record(_usage(1, 0, 1), label="over-budget")

    assert tracker.total_tokens == 11
    assert len(tracker.calls) == 2


def test_record_raises_after_cost_budget_is_exceeded() -> None:
    tracker = CostTracker(max_total_tokens=2_000_000, max_total_cost_yuan=1.0)

    with pytest.raises(CostBudgetExceeded, match="Total cost exceeded"):
        tracker.record(_usage(1_000_000, 0), label="expensive-call")


def test_reset_clears_all_accumulated_state() -> None:
    tracker = CostTracker(max_total_tokens=10_000, max_total_cost_yuan=10.0)
    tracker.record(_usage(100, 50), label="main#1")

    tracker.reset()

    assert tracker.prompt_tokens == 0
    assert tracker.completion_tokens == 0
    assert tracker.total_tokens == 0
    assert tracker.total_cost_yuan == 0.0
    assert tracker.calls == []
