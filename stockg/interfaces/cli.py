"""命令行入口: 组装依赖并触发用例。

子命令:
  analyze <代码>              统一分析流程（A股/美股自动识别），内部走 Agent + RAG
  ingest  <pdf目录或文件>     把材料入库到本地向量库（Chroma / RAG）
  ask     <问题>             直接检索知识库并打印相关段落
"""

from __future__ import annotations

import argparse
import logging
import sys

from langfuse import get_client

from stockg.application import RagIngestionService, RagQueryService
from stockg.domain import run_industrial_agent
from stockg.domain.market import detect_market_from_ticker
from stockg.infrastructure import RAG_STORE_PATH
from stockg.infrastructure.rag import (
    BGE_QUERY_PREFIX,
    ChromaVectorStore,
    RecursiveCharacterSplitter,
    SimpleRetriever,
    build_embedder,
    build_loader,
)
from stockg.logging_config import configure_logging

logger = logging.getLogger(__name__)

DEFAULT_SYMBOL = "600519"  # 贵州茅台, 可换成你关注的 A 股代码


def _build_retriever():
    embedder = build_embedder()
    store = ChromaVectorStore(RAG_STORE_PATH)
    return SimpleRetriever(
        embedder=embedder, store=store, query_prefix=BGE_QUERY_PREFIX
    )


def _cmd_analyze(symbol: str) -> None:
    market = detect_market_from_ticker(symbol) or "us"
    label = "A股" if market == "a" else "美股"
    query = (
        f"请分析{label}标的 {symbol}，先获取其行情与最新新闻，"
        f"并结合本地知识库（如有相关资料）给出「买入 / 观望 / 卖出」的评级与理由。"
    )
    logger.info("识别为%s标的，启动统一分析 Agent: symbol=%s", label, symbol)
    result = run_industrial_agent(query, market=market, thread_id=None)
    logger.info("分析完成: status=%s rating=%s", result.status, result.rating)
    sys.stdout.write(
        f"状态: {result.status}\n评级: {result.rating}\n"
        f"理由: {result.reason}\n成本: {result.cost}\n"
    )


def _cmd_ingest(path: str) -> None:
    embedder = build_embedder()
    store = ChromaVectorStore(RAG_STORE_PATH)
    service = RagIngestionService(
        loader=build_loader(path),
        splitter=RecursiveCharacterSplitter(),
        embedder=embedder,
        store=store,
    )
    n = service.ingest_path(path)
    logger.info("入库完成: chunks=%s embedder=%s", n, embedder.name)


def _cmd_ask(question: str, top_k: int = 3) -> None:
    retriever = _build_retriever()
    service = RagQueryService(retriever)
    contexts = service.retrieve(question, top_k=top_k)
    if not contexts:
        logger.info("知识库检索完成: results=0")
        sys.stdout.write(
            "知识库为空或未检索到相关内容，请先运行 ingest 子命令导入材料。\n"
        )
        return
    logger.info("知识库检索完成: results=%s", len(contexts))
    sys.stdout.write(f"========== 知识库检索结果（{len(contexts)} 条）==========\n")
    for index, context in enumerate(contexts, 1):
        sys.stdout.write(
            f"\n[#{index}] 来源: {context.source} | 相似度: {context.score:.3f}\n"
            f"{context.text}\n"
        )


def main() -> None:
    """配置日志并分发 CLI 子命令。"""
    configure_logging()

    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        import uvicorn

        uvicorn.run("stockg.interfaces.api:app", host="0.0.0.0", port=8000, reload=True)
        return

    parser = argparse.ArgumentParser(description="stockg 股票分析 / 知识库 RAG")
    sub = parser.add_subparsers(dest="cmd")

    p_analyze = sub.add_parser("analyze", help="分析股票（A股/美股自动识别）")
    p_analyze.add_argument("symbol", nargs="?", default=DEFAULT_SYMBOL)

    p_ingest = sub.add_parser("ingest", help="把 PDF / 文本材料入库到向量库")
    p_ingest.add_argument("path", help="PDF 文件或目录（目录递归处理 .pdf/.txt/.md）")

    p_ask = sub.add_parser("ask", help="检索知识库")
    p_ask.add_argument("question", help="检索问题")
    p_ask.add_argument("--top-k", type=int, default=3)

    args = parser.parse_args()

    if args.cmd == "ingest":
        _cmd_ingest(args.path)
    elif args.cmd == "ask":
        _cmd_ask(args.question, top_k=args.top_k)
    else:
        _cmd_analyze(args.symbol if hasattr(args, "symbol") else DEFAULT_SYMBOL)
    get_client().flush()


if __name__ == "__main__":
    main()
