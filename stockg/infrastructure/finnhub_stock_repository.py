"""美股行情仓储：基于 Finnhub 实现 StockDataRepository 端口。"""

from __future__ import annotations

import json
import logging
import urllib.request
from datetime import date, timedelta
from urllib.parse import urlencode

from stockg.domain import StockDataRepository, StockSnapshot
from stockg.infrastructure.config import FINNHUB_API_KEY

logger = logging.getLogger(__name__)

FINNHUB_BASE = "https://finnhub.io/api/v1"


def _finnhub_get(path: str, params: dict):
    """调用 Finnhub REST 接口并返回解析后的 JSON。"""
    if not FINNHUB_API_KEY:
        raise RuntimeError(
            "未设置 FINNHUB_API_KEY，请在项目根目录 .env 或部署环境中配置"
        )
    params = dict(params)
    params["token"] = FINNHUB_API_KEY
    url = f"{FINNHUB_BASE}/{path}?{urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


class FinnhubStockRepository(StockDataRepository):
    """通过 Finnhub 获取美股快照（实时价 + 近 7 天新闻），与 A股仓储共用同一端口。"""

    def fetch_snapshot(self, symbol: str) -> StockSnapshot:
        logger.info("开始抓取美股快照: symbol=%s", symbol)
        try:
            data = _finnhub_get("quote", {"symbol": symbol})
            price = data.get("c") or 0.0
            change_pct = data.get("dp") or 0.0
        except Exception as exc:  # noqa: BLE001 - 行情失败要降级
            logger.warning("Finnhub 行情获取失败: symbol=%s error=%s", symbol, exc)
            return StockSnapshot(
                symbol=symbol,
                name=symbol,
                price=0.0,
                change_pct=0.0,
                news=[f"行情获取失败: {exc}"],
            )

        if not price:
            return StockSnapshot(
                symbol=symbol,
                name=symbol,
                price=0.0,
                change_pct=0.0,
                news=[f"{symbol} 暂无行情数据(可能代码无效或未开盘)"],
            )

        try:
            today = date.today()
            week_ago = today - timedelta(days=7)
            news = (
                _finnhub_get(
                    "company-news",
                    {
                        "symbol": symbol,
                        "from": week_ago.isoformat(),
                        "to": today.isoformat(),
                    },
                )
                or []
            )
            titles = [n.get("headline", "") for n in news][:15]
        except Exception as exc:  # noqa: BLE001 - 新闻非关键, 失败降级
            logger.warning("Finnhub 新闻获取失败: symbol=%s error=%s", symbol, exc)
            titles = [f"新闻获取失败: {exc}"]

        return StockSnapshot(
            symbol=symbol,
            name=symbol,
            price=float(price),
            change_pct=float(change_pct),
            news=titles,
        )
