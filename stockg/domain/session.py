"""会话管理: 按 thread_id 隔离 Agent 的短期记忆与运行时状态。

- SessionContext: 单次会话的全部状态 (messages / snapshot_cache / cost_tracker)
- SessionStore:   进程级 thread_id -> SessionContext 映射, 带 TTL 自动回收
- ContextVar:     让子 agent (research_agent) 隐式拿到当前会话, 无需层层传参

生产环境多实例部署时, 把 SessionStore 换成 Redis 后端即可,
接口不变; 也可后续接入 LangGraph Checkpointer 作为等价替代。
"""
from __future__ import annotations

import asyncio
import contextvars
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from stockg.domain.cost import CostTracker


@dataclass
class SessionContext:
    """单次会话的全部可变状态, 按 thread_id 隔离。"""
    thread_id: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    snapshot_cache: dict[tuple[str, str], Any] = field(default_factory=dict)
    cost_tracker: CostTracker = field(default_factory=CostTracker)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    cache_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def touch(self) -> None:
        self.last_active = time.time()


class SessionStore:
    """进程级会话存储, 带 TTL。生产可替换为 Redis 实现。"""

    def __init__(self, ttl_seconds: int = 3600) -> None:
        self._sessions: dict[str, SessionContext] = {}
        self._ttl_seconds = ttl_seconds

    def get_or_create(self, thread_id: Optional[str] = None) -> SessionContext:
        self._gc()
        if thread_id is None:
            thread_id = str(uuid.uuid4())
        ctx = self._sessions.get(thread_id)
        if ctx is None:
            ctx = SessionContext(thread_id=thread_id)
            self._sessions[thread_id] = ctx
        ctx.touch()
        return ctx

    def get(self, thread_id: str) -> Optional[SessionContext]:
        return self._sessions.get(thread_id)

    def _gc(self) -> None:
        now = time.time()
        expired = [
            tid for tid, ctx in self._sessions.items()
            if now - ctx.last_active > self._ttl_seconds
        ]
        for tid in expired:
            del self._sessions[tid]


# 进程级单例
session_store = SessionStore()

# ContextVar: 子 agent 通过它隐式访问当前会话状态
_current_session: contextvars.ContextVar[SessionContext] = contextvars.ContextVar(
    "stockg_session"
)


def get_current_session() -> SessionContext:
    """获取当前协程绑定的会话上下文。

    子 agent (research_agent) 被主 agent 调用时, 主 agent 已 set 好,
    这里直接取。无上下文时抛 LookupError, 调用方可 fallback。
    """
    return _current_session.get()


def set_current_session(ctx: SessionContext):
    """绑定会话到当前 ContextVar, 返回 token 用于 finally reset。"""
    return _current_session.set(ctx)


def reset_current_session(token) -> None:
    _current_session.reset(token)
