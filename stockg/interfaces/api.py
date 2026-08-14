from __future__ import annotations

import logging
import uuid
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from stockg.domain import run_industrial_agent_async
from stockg.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="stovkg", description="A股/美股智能分析 Agent")


class ChatRequest(BaseModel):
    message: str = Field(..., description="用户的自然语言问题，如「茅台还能买吗？」")
    market: Literal["a", "us", "auto"] = Field(
        default="auto",
        description="auto 表示根据用户问题和股票搜索结果自动识别；a/us 表示强制指定市场",
    )
    session_id: str | None = Field(default=None, description="会话 ID; 省略则开新会话, 传入同一 id 可多轮对话")

class ChatResponse(BaseModel):
    status: str
    session_id: str              # 回传 session_id, 前端下次请求带上即可多轮
    message: str                 # Agent 的自然语言回复（即 reason）
    rating: str | None = None    # 评级，如果识别到了
    fetched_context: list[str]
    cost: dict

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    session_id = req.session_id or str(uuid.uuid4())
    logger.info("收到 Agent 请求: session_id=%s market=%s", session_id, req.market)
    try:
        result = await run_industrial_agent_async(
            req.message, market=req.market, thread_id=session_id
        )
    except Exception as exc:
        logger.exception(
            "Agent 请求失败: session_id=%s market=%s",
            session_id,
            req.market,
        )
        raise HTTPException(status_code=500, detail="Agent 处理失败") from exc
    logger.info(
        "Agent 请求完成: session_id=%s status=%s rating=%s tokens=%s cost_yuan=%s",
        session_id,
        result.status,
        result.rating,
        result.cost.get("tool_tokens"),
        result.cost.get("cost_yuan"),
    )
    return ChatResponse(
        session_id=session_id,
        status=result.status,
        message=result.reason,
        rating=result.rating,
        fetched_context=result.fetched_context,
        cost=result.cost
    )

@app.get("/health")
def health():
    return {"status": "ok"}