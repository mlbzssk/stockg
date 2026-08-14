from __future__ import annotations

import asyncio
import json
import logging
from functools import lru_cache

from langfuse import observe
from langfuse.openai import AsyncOpenAI, OpenAI

from stockg.domain.entities import AgentRunResult
from stockg.domain.market import RequestedMarket, resolve_market
from stockg.infrastructure import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
)
from stockg.infrastructure.llm_stream import stream_completion, stream_completion_async

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_client() -> OpenAI:
    """在首次模型调用时创建同步客户端。"""
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("未设置 DEEPSEEK_API_KEY 环境变量")
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


@lru_cache(maxsize=1)
def _get_async_client() -> AsyncOpenAI:
    """在首次模型调用时创建异步客户端。"""
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("未设置 DEEPSEEK_API_KEY 环境变量")
    return AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


tools = [
    {
        "type": "function",
        "function": {
            "name": "submit_final_report",
            "description": "提交最终的分析报告",
            "parameters": {
                "type": "object",
                "properties": {
                    "investment_rating": {
                        "type": "string",
                        "enum": ["买入", "观望", "卖出"],
                        "description": "最终评级",
                    },
                    "analysis_reason": {
                        "type": "string",
                        "description": "详细的分析理由",
                    },
                },
                "required": ["investment_rating", "analysis_reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "research_stock",
            "description": "调用资料搜索 agent，获取指定市场股票的相关资料",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "标准股票代码，A股如 600519，美股如 AAPL、TSLA",
                    },
                    "market": {
                        "type": "string",
                        "enum": ["a", "us"],
                        "description": "从用户问题和股票搜索结果中识别出的市场",
                    },
                },
                "required": ["ticker", "market"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_info",
            "description": "获取股票信息,比如id和名字，市场代码等",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "股票代码，A股如 600519，美股如 AAPL、TSLA",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
]


# ==========================================
# 工业级 Loop Engine
# ==========================================
@observe(name="industrial_agent")
def run_industrial_agent(
    user_query: str,
    thread_id: str | None = None,
    market: RequestedMarket = "auto",
    max_iterations: int = 5,
) -> AgentRunResult:
    """统一的股票分析 Agent。

    market: "a" -> A股（akshare 行情），"us" -> 美股（finnhub 行情），"auto" -> 自动推断。
    除行情数据源不同外，知识库检索与最终报告流程完全一致。
    """
    from stockg.domain.session import (
        reset_current_session,
        session_store,
        set_current_session,
    )

    if market not in {"a", "us", "auto"}:
        raise ValueError("market 必须是 a、us 或 auto")

    ctx = session_store.get_or_create(thread_id)
    _token = set_current_session(ctx)
    fetched_context: list[str] = []
    tracker = ctx.cost_tracker

    logger.info(
        "股票分析 Agent 启动: session_id=%s market=%s max_iterations=%s",
        ctx.thread_id,
        market,
        max_iterations,
    )
    # 多轮: 仅新会话才注入 system; 已有历史则只追加本轮 user 消息
    if not ctx.messages:
        ctx.messages.append(
            {
                "role": "system",
                "content": """你是一个高级金融 Agent，支持 A 股和美股分析。

            你可以使用以下工具:
            1. get_stock_info(ticker): 根据名称或代码搜索股票，并确认标准代码和市场。
            2. research_stock(ticker, market): 获取股票行情和新闻。
            3. submit_final_report: 提交最终分析报告。

            规则:
            1. 每条用户消息都会包含本轮 market 参数。
            2. market=a 或 market=us 时，将它作为调用方强制指定的市场。
            3. market=auto 时，根据用户问题识别股票和市场。
            4. 用户提供股票名称、代码不标准、存在歧义或市场不确定时，必须先调用 get_stock_info，并等待查询结果后再调用 research_stock。
            5. 调用 research_stock 时，ticker 必须是标准代码，market 必须是 a 或 us。
            6. A 股代码使用六位数字，例如 600519；美股代码使用英文代码，例如 AAPL。
            7. research_stock 只调用一次，拿到数据后下一步必须调用 submit_final_report。
            8. 数据收集完毕后不能直接输出文本，只能调用 submit_final_report。""",
            }
        )
    ctx.messages.append(
        {
            "role": "user",
            "content": f"本轮 market 参数：{market}\n用户问题：{user_query}",
        }
    )
    messages = ctx.messages  # 直接引用, loop 内 append 自动持久化到会话
    for iteration in range(max_iterations):
        logger.debug(
            "Agent 开始迭代: session_id=%s iteration=%s",
            ctx.thread_id,
            iteration + 1,
        )
        tracker.check()
        agent_msg, usage, finish_reason = stream_completion(
            _get_client(),
            model=DEEPSEEK_MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=4096,
        )
        if finish_reason == "length":
            logger.warning(
                "模型输出被截断: session_id=%s iteration=%s",
                ctx.thread_id,
                iteration + 1,
            )
            messages.append(
                {
                    "role": "user",
                    "content": "输出过长被截断，请直接调用工具，不要输出长文本。",
                }
            )
            continue
        messages.append(agent_msg)
        tracker.record(usage, label=f"main#{iteration + 1}")

        final_report = None
        for tool_call in agent_msg["tool_calls"] or []:
            func_name = tool_call["function"]["name"]
            func_args = json.loads(tool_call["function"]["arguments"])

            if func_name == "submit_final_report":
                rating = func_args.get("investment_rating", "未知评级")
                reason = func_args.get("analysis_reason", "未知理由")
                logger.info(
                    "Agent 已生成最终报告: session_id=%s rating=%s iteration=%s",
                    ctx.thread_id,
                    rating,
                    iteration + 1,
                )
                result = f"已提交最终报告。评级: {rating}"
                final_report = (rating, reason)

            elif func_name == "research_stock":
                from stockg.domain.research_agent import run_research_agent

                resolved_market = resolve_market(
                    market,
                    func_args["ticker"],
                    func_args.get("market"),
                )
                logger.info(
                    "Agent 调用资料检索: session_id=%s ticker=%s market=%s",
                    ctx.thread_id,
                    func_args["ticker"],
                    resolved_market,
                )
                result = run_research_agent(func_args["ticker"], market=resolved_market)
            elif func_name == "get_stock_info":
                logger.info(
                    "Agent 查询股票信息: session_id=%s ticker=%s",
                    ctx.thread_id,
                    func_args["ticker"],
                )
                result = get_stock_info(func_args["ticker"])
            else:
                logger.warning(
                    "Agent 请求未知工具: session_id=%s tool=%s",
                    ctx.thread_id,
                    func_name,
                )
                result = f"未知工具: {func_name}，可用工具: research_stock / submit_final_report"

            fetched_context.append(f"[{func_name} {result}]")
            messages.append(
                {
                    "tool_call_id": tool_call["id"],
                    "role": "tool",
                    "name": func_name,
                    "content": result,
                }
            )

        if final_report is not None:
            rating, reason = final_report
            logger.info(
                "Agent 运行成功: session_id=%s iterations=%s total_tokens=%s cost_yuan=%.4f",
                ctx.thread_id,
                iteration + 1,
                tracker.total_tokens,
                tracker.total_cost_yuan,
            )
            reset_current_session(_token)
            return AgentRunResult(
                status="success",
                rating=rating,
                reason=reason,
                fetched_context=fetched_context,
                cost={
                    "tool_tokens": tracker.total_tokens,
                    "cost_yuan": round(tracker.total_cost_yuan, 4),
                },
            )
    logger.warning(
        "Agent 达到最大迭代次数: session_id=%s max_iterations=%s total_tokens=%s",
        ctx.thread_id,
        max_iterations,
        tracker.total_tokens,
    )
    reset_current_session(_token)
    return AgentRunResult(
        status="failed",
        rating="未知评级",
        reason="agent run out of iterations",
        fetched_context=fetched_context,
        cost={
            "tool_tokens": tracker.total_tokens,
            "cost_yuan": round(tracker.total_cost_yuan, 4),
        },
    )


@observe(name="industrial_agent_async")
async def run_industrial_agent_async(
    user_query: str,
    thread_id: str | None = None,
    market: RequestedMarket = "auto",
    max_iterations: int = 5,
) -> AgentRunResult:
    from stockg.domain.session import (
        reset_current_session,
        session_store,
        set_current_session,
    )

    if market not in {"a", "us", "auto"}:
        raise ValueError("market 必须是 a、us 或 auto")

    ctx = session_store.get_or_create(thread_id)
    _token = set_current_session(ctx)
    fetched_context: list[str] = []
    tracker = ctx.cost_tracker

    logger.info(
        "异步股票分析 Agent 启动: session_id=%s market=%s max_iterations=%s",
        ctx.thread_id,
        market,
        max_iterations,
    )
    if not ctx.messages:
        ctx.messages.append(
            {
                "role": "system",
                "content": """你是一个高级金融 Agent，支持 A 股和美股分析。

            你可以使用以下工具:
            1. get_stock_info(ticker): 根据名称或代码搜索股票，并确认标准代码和市场。
            2. research_stock(ticker, market): 获取股票行情和新闻。
            3. submit_final_report: 提交最终分析报告。

            规则:
            1. 每条用户消息都会包含本轮 market 参数。
            2. market=a 或 market=us 时，将它作为调用方强制指定的市场。
            3. market=auto 时，根据用户问题识别股票和市场。
            4. 用户提供股票名称、代码不标准、存在歧义或市场不确定时，必须先调用 get_stock_info，并等待查询结果后再调用 research_stock。
            5. 调用 research_stock 时，ticker 必须是标准代码，market 必须是 a 或 us。
            6. A 股代码使用六位数字，例如 600519；美股代码使用英文代码，例如 AAPL。
            7. research_stock 只调用一次，拿到数据后下一步必须调用 submit_final_report。
            8. 数据收集完毕后不能直接输出文本，只能调用 submit_final_report。""",
            }
        )
    ctx.messages.append(
        {
            "role": "user",
            "content": f"本轮 market 参数：{market}\n用户问题：{user_query}",
        }
    )
    messages = ctx.messages
    for iteration in range(max_iterations):
        logger.debug(
            "异步 Agent 开始迭代: session_id=%s iteration=%s",
            ctx.thread_id,
            iteration + 1,
        )
        tracker.check()
        agent_msg, usage, finish_reason = await stream_completion_async(
            _get_async_client(),
            model=DEEPSEEK_MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=4096,
        )
        if finish_reason == "length":
            logger.warning(
                "异步模型输出被截断: session_id=%s iteration=%s",
                ctx.thread_id,
                iteration + 1,
            )
            messages.append(
                {
                    "role": "user",
                    "content": "输出过长被截断，请直接调用工具，不要输出长文本。",
                }
            )
            continue
        messages.append(agent_msg)
        tracker.record(usage, label=f"main#{iteration + 1}")

        final_report = None
        for tool_call in agent_msg["tool_calls"] or []:
            func_name = tool_call["function"]["name"]
            func_args = json.loads(tool_call["function"]["arguments"])

            if func_name == "submit_final_report":
                rating = func_args.get("investment_rating", "未知评级")
                reason = func_args.get("analysis_reason", "未知理由")
                logger.info(
                    "异步 Agent 已生成最终报告: session_id=%s rating=%s iteration=%s",
                    ctx.thread_id,
                    rating,
                    iteration + 1,
                )
                result = f"已提交最终报告。评级: {rating}"
                final_report = (rating, reason)

            elif func_name == "research_stock":
                from stockg.domain.research_agent import run_research_agent_async

                resolved_market = resolve_market(
                    market,
                    func_args["ticker"],
                    func_args.get("market"),
                )
                logger.info(
                    "异步 Agent 调用资料检索: session_id=%s ticker=%s market=%s",
                    ctx.thread_id,
                    func_args["ticker"],
                    resolved_market,
                )
                result = await run_research_agent_async(
                    func_args["ticker"], market=resolved_market
                )
            elif func_name == "get_stock_info":
                logger.info(
                    "异步 Agent 查询股票信息: session_id=%s ticker=%s",
                    ctx.thread_id,
                    func_args["ticker"],
                )
                # get_stock_info 内部是 requests，丢线程池
                result = await asyncio.to_thread(get_stock_info, func_args["ticker"])
            else:
                logger.warning(
                    "异步 Agent 请求未知工具: session_id=%s tool=%s",
                    ctx.thread_id,
                    func_name,
                )
                result = f"未知工具: {func_name}，可用工具: research_stock / submit_final_report"

            fetched_context.append(f"[{func_name} {result}]")
            messages.append(
                {
                    "tool_call_id": tool_call["id"],
                    "role": "tool",
                    "name": func_name,
                    "content": result,
                }
            )

        if final_report is not None:
            rating, reason = final_report
            logger.info(
                "Agent 运行成功: session_id=%s iterations=%s total_tokens=%s cost_yuan=%.4f",
                ctx.thread_id,
                iteration + 1,
                tracker.total_tokens,
                tracker.total_cost_yuan,
            )
            reset_current_session(_token)
            return AgentRunResult(
                status="success",
                rating=rating,
                reason=reason,
                fetched_context=fetched_context,
                cost={
                    "tool_tokens": tracker.total_tokens,
                    "cost_yuan": round(tracker.total_cost_yuan, 4),
                },
            )

    logger.warning(
        "异步 Agent 达到最大迭代次数: session_id=%s max_iterations=%s total_tokens=%s",
        ctx.thread_id,
        max_iterations,
        tracker.total_tokens,
    )
    reset_current_session(_token)
    return AgentRunResult(
        status="failed",
        rating="未知评级",
        reason="agent run out of iterations",
        fetched_context=fetched_context,
        cost={
            "tool_tokens": tracker.total_tokens,
            "cost_yuan": round(tracker.total_cost_yuan, 4),
        },
    )


@observe(name="get_stock_info")
def get_stock_info(ticker: str) -> str:
    import requests

    url = "https://searchapi.eastmoney.com/api/suggest/get"
    params = {"input": ticker, "count": 5, "type": 14}
    headers = {
        "Referer": "https://www.eastmoney.com/",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    }
    try:
        r = requests.get(url, params=params, headers=headers, timeout=5)
        data = r.json()
        results = data.get("QuotationCodeTable", {}).get("Data", [])
        if not results:
            return f"未找到匹配'{ticker}'的股票，请确认名称或代码"
        lines = []
        for item in results:
            code = item.get("Code", "")
            name = item.get("Name", "")
            classify = item.get("Classify", "")
            if classify == "AStock":
                market = "A股"
            elif classify == "UsStock":
                market = "美股"
            elif classify == "HKStock":
                market = "港股"
            else:
                continue  # 跳过基金/债券等非股票
            lines.append(f"{name}({code}), {market}")
        if not lines:
            return "未找到类型"
        return "找到:" + "|".join(lines)
    except Exception as exc:
        logger.exception("股票信息查询失败: ticker=%s", ticker)
        return f"获取股票信息失败: {exc}"


if __name__ == "__main__":
    from stockg.logging_config import configure_logging

    configure_logging()
    logger.info("启动股票分析 Agent 调试入口")
    result = run_industrial_agent("茅台最近走势怎么样？还能买吗？", thread_id=None)
    logger.info(
        "股票分析 Agent 调试完成: status=%s rating=%s", result.status, result.rating
    )
