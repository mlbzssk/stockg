from __future__ import annotations

import asyncio
import logging
import operator
import threading
from typing import Annotated, NotRequired, Required, TypedDict

from langfuse import observe
from langgraph.graph import END, START, StateGraph

from stockg.domain.entities import StockSnapshot
from stockg.domain.market import Market
from stockg.domain.session import (
    SessionContext,
    get_current_session,
    reset_current_session,
    set_current_session,
)
from stockg.infrastructure import AkshareStockRepository, FinnhubStockRepository

logger = logging.getLogger(__name__)

CacheKey = tuple[Market, str]


class ResearchState(TypedDict, total=False):
    """LangGraph 可序列化状态；节点按阶段增量写入字段。"""

    ticker: Required[str]
    market: Required[Market]
    rag_top_k: Required[int]
    errors: Required[Annotated[list[str], operator.add]]
    price: NotRequired[str]
    news: NotRequired[str]
    knowledge_base: NotRequired[str]
    research_msg: NotRequired[str]


sh_active_repository = AkshareStockRepository()
us_active_repository = FinnhubStockRepository()
_rag_retriever = None
_rag_lock = threading.Lock()


def _get_rag_retriever():
    global _rag_retriever
    if _rag_retriever is None:
        with _rag_lock:
            if _rag_retriever is None:
                from stockg.infrastructure import RAG_STORE_PATH
                from stockg.infrastructure.rag import (
                    BGE_QUERY_PREFIX,
                    ChromaVectorStore,
                    SimpleRetriever,
                    build_embedder,
                )

                embedder = build_embedder()
                store = ChromaVectorStore(RAG_STORE_PATH)
                _rag_retriever = SimpleRetriever(
                    embedder=embedder,
                    store=store,
                    query_prefix=BGE_QUERY_PREFIX,
                )
    return _rag_retriever


def _repository_for(market: str):
    if market == "a":
        return sh_active_repository
    if market == "us":
        return us_active_repository
    raise ValueError("market 必须是 a 或 us")


async def _fetch_snapshot_async(ticker: str, market: Market) -> StockSnapshot:
    """按已识别市场抓取并缓存快照；缓存键包含市场，支持跨市场并发。"""
    ctx = get_current_session()
    cache_key: CacheKey = (market, ticker)
    if cache_key not in ctx.snapshot_cache:
        async with ctx.cache_lock:
            if cache_key not in ctx.snapshot_cache:
                repository = _repository_for(market)
                ctx.snapshot_cache[cache_key] = await repository.fetch_snapshot_async(
                    ticker
                )
    return ctx.snapshot_cache[cache_key]


def _retrieve_knowledge(ticker: str, top_k: int) -> str:
    query = f"从本地知识库搜索股票{ticker}相关的知识，包括过往研报、公告、新闻等"
    retriever = _get_rag_retriever()
    results = retriever.retrieve(query, top_k=top_k)
    if not results:
        return (
            "知识库为空或未检索到相关内容。请先运行 "
            "`python -m stockg.interfaces.cli ingest <pdf目录或文件>` 导入材料。"
        )
    lines = [
        f"[来源: {result.source} | 相似度: {result.score:.3f}]\n{result.text}"
        for result in results
    ]
    return "知识库检索结果:\n" + "\n\n".join(lines)


@observe(name="research_fetch_market_data")
async def fetch_market_data(state: ResearchState) -> dict[str, object]:
    """获取一个行情快照，并从中同时生成价格和新闻。"""
    ticker = state["ticker"]
    try:
        snapshot = await _fetch_snapshot_async(ticker, state["market"])
        if snapshot.price:
            price = (
                f"{ticker}（{snapshot.name}）当前最新股价: {snapshot.price} "
                f"(涨跌幅: {snapshot.change_pct}%)"
            )
        else:
            price = f"{ticker} 暂无行情数据(可能代码无效或未开盘)"

        if snapshot.news:
            news = f"{ticker} 最新相关新闻:\n" + "\n".join(
                f"- {title}" for title in snapshot.news
            )
        else:
            news = f"{ticker} 暂无相关新闻"
        return {"price": price, "news": news, "errors": []}
    except Exception as exc:  # noqa: BLE001 - 单个数据源失败时让图继续汇总
        logger.warning(
            "行情与新闻获取失败，Research Agent 将继续汇总: ticker=%s market=%s error=%s",
            ticker,
            state["market"],
            exc,
        )
        error = f"{ticker} 行情与新闻获取失败: {exc}"
        return {
            "price": error,
            "news": f"{ticker} 新闻获取失败: 行情快照不可用",
            "errors": [error],
        }


@observe(name="research_retrieve_knowledge")
async def retrieve_knowledge(state: ResearchState) -> dict[str, object]:
    """在线程池执行本地向量检索，避免阻塞 API 事件循环。"""
    ticker = state["ticker"]
    try:
        knowledge = await asyncio.to_thread(
            _retrieve_knowledge,
            ticker,
            state["rag_top_k"],
        )
        return {"knowledge_base": knowledge, "errors": []}
    except Exception as exc:  # noqa: BLE001 - RAG 失败不阻断行情分析
        logger.warning(
            "知识库检索失败，Research Agent 将继续汇总: ticker=%s error=%s",
            ticker,
            exc,
        )
        error = f"知识库检索失败: {exc}"
        return {"knowledge_base": error, "errors": [error]}


@observe(name="research_build_result")
def build_result(state: ResearchState) -> dict[str, str]:
    """将并行节点的结构化结果汇总为主 Agent 当前所需的文本协议。"""
    research_msg = (
        f"\n当前股价: {state.get('price', '未获取')}"
        f"\n最新新闻: {state.get('news', '未获取')}"
        f"\n知识库: {state.get('knowledge_base', '未获取')}"
    )
    if state.get("errors"):
        research_msg += "\n资料收集告警: " + " | ".join(state["errors"])
    return {"research_msg": research_msg}


def _build_research_graph():
    workflow = StateGraph(ResearchState)
    workflow.add_node("fetch_market_data", fetch_market_data)
    workflow.add_node("retrieve_knowledge", retrieve_knowledge)
    workflow.add_node("build_result", build_result)

    workflow.add_edge(START, "fetch_market_data")
    workflow.add_edge(START, "retrieve_knowledge")
    workflow.add_edge(
        ["fetch_market_data", "retrieve_knowledge"],
        "build_result",
    )
    workflow.add_edge("build_result", END)
    return workflow.compile()


# 图结构固定，模块加载时只编译一次；每次请求仅传入独立 state。
research_graph = _build_research_graph()


@observe(name="run_research_agent_async")
async def run_research_agent_async(
    ticker: str,
    market: Market,
    rag_top_k: int = 3,
) -> str:
    """运行唯一的异步 Research LangGraph。"""
    if market not in {"a", "us"}:
        raise ValueError("market 必须是 a 或 us")
    if rag_top_k <= 0:
        raise ValueError("rag_top_k 必须大于 0")

    normalized_ticker = ticker.strip().upper()
    if not normalized_ticker:
        raise ValueError("ticker 不能为空")

    session_token = None
    try:
        ctx = get_current_session()
    except LookupError:
        ctx = SessionContext(thread_id="research-fallback")
        session_token = set_current_session(ctx)

    cache_key: CacheKey = (market, normalized_ticker)
    async with ctx.cache_lock:
        ctx.snapshot_cache.pop(cache_key, None)

    initial_state: ResearchState = {
        "ticker": normalized_ticker,
        "market": market,
        "rag_top_k": rag_top_k,
        "errors": [],
    }

    logger.info(
        "Research LangGraph 启动: session_id=%s ticker=%s market=%s rag_top_k=%s",
        ctx.thread_id,
        normalized_ticker,
        market,
        rag_top_k,
    )
    try:
        final_state = await research_graph.ainvoke(initial_state)
        logger.info(
            "Research LangGraph 完成: session_id=%s ticker=%s errors=%s",
            ctx.thread_id,
            normalized_ticker,
            len(final_state.get("errors", [])),
        )
        return final_state["research_msg"]
    finally:
        if session_token is not None:
            reset_current_session(session_token)


@observe(name="run_research_agent")
def run_research_agent(
    ticker: str,
    market: Market,
    rag_top_k: int = 3,
) -> str:
    """供同步 CLI 使用的薄适配器；业务编排只存在于异步图中。"""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            run_research_agent_async(ticker, market, rag_top_k=rag_top_k)
        )
    raise RuntimeError(
        "检测到正在运行的事件循环，请改用 await run_research_agent_async(...)"
    )


if __name__ == "__main__":
    from stockg.logging_config import configure_logging

    configure_logging()
    run_research_agent("600519", market="a")
