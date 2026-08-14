import contextvars

import pytest

from stockg.domain import session
from stockg.domain.session import (
    SessionContext,
    SessionStore,
    get_current_session,
    reset_current_session,
    set_current_session,
)


def test_session_context_uses_independent_mutable_defaults() -> None:
    first = SessionContext(thread_id="first")
    second = SessionContext(thread_id="second")

    first.messages.append({"role": "user", "content": "hello"})
    first.snapshot_cache[("a", "600519")] = object()
    first.cost_tracker.total_tokens = 12

    assert second.messages == []
    assert second.snapshot_cache == {}
    assert second.cost_tracker.total_tokens == 0
    assert first.cache_lock is not second.cache_lock


def test_touch_updates_last_active(monkeypatch: pytest.MonkeyPatch) -> None:
    context = SessionContext(thread_id="thread", last_active=100.0)
    monkeypatch.setattr(session.time, "time", lambda: 125.0)

    context.touch()

    assert context.last_active == 125.0


def test_store_reuses_existing_session_and_refreshes_activity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 100.0
    monkeypatch.setattr(session.time, "time", lambda: now)
    store = SessionStore(ttl_seconds=60)

    first = store.get_or_create("thread-1")
    now = 120.0
    second = store.get_or_create("thread-1")

    assert second is first
    assert second.last_active == 120.0
    assert store.get("thread-1") is first


def test_store_generates_thread_id_when_not_provided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session.uuid, "uuid4", lambda: "generated-id")
    store = SessionStore()

    context = store.get_or_create()

    assert context.thread_id == "generated-id"
    assert store.get("generated-id") is context


def test_store_removes_expired_sessions_before_creating_new_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 1_000.0
    monkeypatch.setattr(session.time, "time", lambda: now)
    store = SessionStore(ttl_seconds=10)
    expired = store.get_or_create("expired")
    active = store.get_or_create("active")
    expired.last_active = 980.0
    active.last_active = 995.0

    store.get_or_create("new")

    assert store.get("expired") is None
    assert store.get("active") is active
    assert store.get("new") is not None


def test_context_variable_can_be_bound_and_reset() -> None:
    context = SessionContext(thread_id="isolated")

    def exercise_context() -> None:
        with pytest.raises(LookupError):
            get_current_session()
        token = set_current_session(context)
        assert get_current_session() is context
        reset_current_session(token)
        with pytest.raises(LookupError):
            get_current_session()

    contextvars.Context().run(exercise_context)
