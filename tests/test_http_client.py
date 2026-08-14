import asyncio

import pytest

from stockg.infrastructure import http_client


class FakeResponse:
    def __init__(self, status_code: int, text: str, payload: dict) -> None:
        self.status_code = status_code
        self.text = text
        self._payload = payload

    def json(self) -> dict:
        return self._payload


def test_get_json_retries_non_json_response_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter(
        [
            FakeResponse(502, "Bad Gateway", {}),
            FakeResponse(200, '{"ok": true}', {"ok": True}),
        ]
    )
    sleeps: list[float] = []
    monkeypatch.setattr(
        http_client.requests, "get", lambda *args, **kwargs: next(responses)
    )
    monkeypatch.setattr(http_client.random, "uniform", lambda start, end: 0.0)
    monkeypatch.setattr(http_client.time, "sleep", sleeps.append)

    result = http_client.get_json(
        "https://example.com", {"q": "stock"}, max_retries=2, sleep=2
    )

    assert result == {"ok": True}
    assert sleeps == [1.0]


def test_get_json_reports_last_exception_after_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise TimeoutError("timed out")

    monkeypatch.setattr(http_client.requests, "get", fail)
    monkeypatch.setattr(http_client.random, "uniform", lambda start, end: 0.0)
    monkeypatch.setattr(http_client.time, "sleep", lambda delay: None)

    with pytest.raises(RuntimeError, match="TimeoutError: timed out"):
        http_client.get_json("https://example.com", {}, max_retries=3, sleep=0)

    assert calls == 3


def test_get_json_async_retries_and_uses_async_sleep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcomes: list[FakeResponse | Exception] = [
        TimeoutError("temporary"),
        FakeResponse(200, '{"value": 1}', {"value": 1}),
    ]
    sleeps: list[float] = []

    class FakeAsyncClient:
        def __init__(self, timeout: float) -> None:
            assert timeout == 15.0

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback) -> None:
            return None

        async def get(self, *args, **kwargs) -> FakeResponse:
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(http_client.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(http_client.random, "uniform", lambda start, end: 0.0)
    monkeypatch.setattr(http_client.asyncio, "sleep", fake_sleep)

    result = asyncio.run(
        http_client.get_json_async(
            "https://example.com",
            {"q": "stock"},
            max_retries=2,
            sleep=4,
        )
    )

    assert result == {"value": 1}
    assert sleeps == [2.0]
