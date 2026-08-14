"""股票数据仓储的具体实现: 新浪(主) + 东财(降级) 双数据源。"""

from __future__ import annotations

import asyncio
import logging
import re

import akshare as ak
import httpx
import requests

from stockg.domain import StockDataRepository, StockSnapshot
from stockg.infrastructure.http_client import get_json, get_json_async

logger = logging.getLogger(__name__)


def _to_secid(symbol: str) -> str:
    """根据股票代码判断市场, 生成东财 secid: 沪市 6 开头 -> 1.x, 深市/北交所 -> 0.x。"""
    return f"1.{symbol}" if symbol.startswith("6") else f"0.{symbol}"


def _to_sina_prefix(symbol: str) -> str:
    """生成新浪行情前缀: 沪市 6 开头 -> sh, 深市/创业板 -> sz。"""
    return "sh" if symbol.startswith("6") else "sz"


class AkshareStockRepository(StockDataRepository):
    """通过新浪接口获取股票快照, 失败时降级到东财。"""

    def fetch_snapshot(self, symbol: str) -> StockSnapshot:
        logger.info("开始抓取 A 股快照: symbol=%s", symbol)

        # 1. 先试新浪, 失败降级到东财
        try:
            name, price, change_pct = self._fetch_from_sina(symbol)
        except Exception as e:
            logger.warning(
                "新浪行情接口失败，降级到东方财富: symbol=%s error=%s",
                symbol,
                e,
            )
            name, price, change_pct = self._fetch_from_eastmoney(symbol)

        # 2. 抓取个股新闻
        logger.debug("开始抓取 A 股新闻: symbol=%s", symbol)
        news_titles = self._fetch_news(symbol, name)

        return StockSnapshot(
            symbol=symbol,
            name=name,
            price=price,
            change_pct=change_pct,
            news=news_titles,
        )

    def _fetch_from_eastmoney(self, symbol: str) -> tuple[str, float, float]:
        """东财实时行情接口: 返回 (名称, 现价, 涨跌幅%)。"""
        secid = _to_secid(symbol)
        url = "https://push2.eastmoney.com/api/qt/stock/get"
        params = {
            "secid": secid,
            "fields": "f43,f57,f58,f169",
            "invt": "2",
            "fltt": "2",
        }
        data = get_json(url, params, max_retries=3)["data"]
        return data["f58"], float(data["f43"]), float(data["f169"])

    def _fetch_from_sina(self, symbol: str) -> tuple[str, float, float]:
        """新浪行情接口(降级源): 返回 (名称, 现价, 涨跌幅%)。

        新浪返回格式:
            var hq_str_sh600519="贵州茅台,1325.000,1309.220,1348.860,...";
        字段顺序: 名称,昨收,今开,现价,最高,最低,...,日期,时间
        涨跌幅 = (现价 - 昨收) / 昨收 * 100
        """
        prefix = _to_sina_prefix(symbol)
        url = f"https://hq.sinajs.cn/list={prefix}{symbol}"
        headers = {
            "Referer": "https://finance.sina.com.cn",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        }
        r = requests.get(url, headers=headers, timeout=10)
        r.encoding = "gbk"
        match = re.search(r'"(.+)"', r.text)
        if not match or not match.group(1):
            raise RuntimeError(f"新浪接口未返回 {symbol} 的数据")
        fields = match.group(1).split(",")
        name = fields[0]
        yesterday_close = float(fields[1])
        current_price = float(fields[3])
        change_pct = (
            round((current_price - yesterday_close) / yesterday_close * 100, 2)
            if yesterday_close
            else 0.0
        )
        return name, current_price, change_pct

    def _fetch_news(self, symbol: str, name: str) -> list[str]:
        """获取个股新闻标题。

        优先用东财搜索接口(search-api-web, 非被封的 push2),
        失败降级到 akshare 的 stock_news_em。
        """
        try:
            return self._fetch_news_from_eastmoney_search(name)
        except Exception as e:
            logger.warning(
                "东方财富新闻接口失败，降级到 AkShare: symbol=%s error=%s",
                symbol,
                e,
            )
            try:
                news_df = ak.stock_news_em(symbol=symbol)
                return news_df["新闻标题"].head(5).tolist()
            except Exception as fallback_exc:
                logger.warning(
                    "AkShare 新闻降级源也失败: symbol=%s error=%s",
                    symbol,
                    fallback_exc,
                )
                return ["未能成功抓取新闻"]

    def _fetch_news_from_eastmoney_search(self, name: str) -> list[str]:
        """东财搜索接口: 按股票名称搜新闻标题。"""
        import json as _json

        url = "https://search-api-web.eastmoney.com/search/jsonp"
        param = {
            "uid": "",
            "keyword": name,
            "type": ["cmsArticleWebOld"],
            "client": "web",
            "clientType": "web",
            "clientVersion": "curr",
            "param": {
                "cmsArticleWebOld": {
                    "searchScope": "default",
                    "sort": "default",
                    "pageIndex": 1,
                    "pageSize": 5,
                }
            },
        }
        r = requests.get(
            url,
            params={"cb": "jQuery", "param": _json.dumps(param, ensure_ascii=False)},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        # 返回是 JSONP: jQuery({...}), 去掉外层
        text = r.text
        start = text.index("(") + 1
        end = text.rindex(")")
        data = _json.loads(text[start:end])
        articles = data.get("result", {}).get("cmsArticleWebOld", [])
        titles = [a["title"] for a in articles if "title" in a]
        # 去掉 HTML 高亮标签 <em>...</em>
        return (
            [re.sub(r"<[^>]+>", "", t) for t in titles]
            if titles
            else ["未能成功抓取新闻"]
        )

    async def fetch_snapshot_async(self, symbol: str) -> StockSnapshot:
        """异步抓取股票快照."""
        price_task = asyncio.create_task(self._fetch_price_async(symbol))
        news_task = asyncio.create_task(self._fetch_news_async(symbol))
        name, price, change_pct = await price_task
        news_titles = await news_task
        return StockSnapshot(
            symbol=symbol,
            name=name,
            price=price,
            change_pct=change_pct,
            news=news_titles,
        )

    async def _fetch_price_async(self, symbol: str) -> tuple[str, float, float]:
        """异步抓取股票价格."""
        try:
            return await self._fetch_from_sina_async(symbol)
        except Exception as e:
            logger.warning(
                "新浪行情接口失败，降级到东方财富: symbol=%s error=%s",
                symbol,
                e,
            )
            return await self._fetch_from_eastmoney_async(symbol)

    async def _fetch_from_eastmoney_async(
        self, symbol: str
    ) -> tuple[str, float, float]:
        """异步抓取股票价格."""
        secid = _to_secid(symbol)
        url = "https://push2.eastmoney.com/api/qt/stock/get"
        params = {
            "secid": secid,
            "fields": "f43,f57,f58,f169",
            "invt": "2",
            "fltt": "2",
        }
        data = (await get_json_async(url, params, max_retries=3))["data"]
        return data["f58"], float(data["f43"]), float(data["f169"])

    async def _fetch_from_sina_async(self, symbol: str) -> tuple[str, float, float]:
        """异步抓取股票价格."""
        prefix = _to_sina_prefix(symbol)
        url = f"https://hq.sinajs.cn/list={prefix}{symbol}"
        headers = {
            "Referer": "https://finance.sina.com.cn",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        }
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(url, headers=headers)
            r.encoding = "gbk"
        match = re.search(r'"(.+)"', r.text)
        if not match or not match.group(1):
            raise RuntimeError(f"新浪接口未返回 {symbol} 的数据")
        fields = match.group(1).split(",")
        name = fields[0]
        yesterday_close = float(fields[1])
        current_price = float(fields[3])
        change_pct = (
            round((current_price - yesterday_close) / yesterday_close * 100, 2)
            if yesterday_close
            else 0.0
        )
        return name, current_price, change_pct

    async def _fetch_news_async(self, symbol: str) -> list[str]:
        """async 抓新闻。优先东财 search(用 symbol)，失败降级 akshare(包 to_thread)。"""
        try:
            return await self._fetch_news_from_eastmoney_search_async(symbol)
        except Exception as e:
            logger.warning(
                "东方财富新闻接口失败，降级到 AkShare: symbol=%s error=%s",
                symbol,
                e,
            )
            try:
                # akshare 无 async API，丢线程池
                news_df = await asyncio.to_thread(ak.stock_news_em, symbol=symbol)
                return news_df["新闻标题"].head(5).tolist()
            except Exception as fallback_exc:
                logger.warning(
                    "AkShare 新闻降级源也失败: symbol=%s error=%s",
                    symbol,
                    fallback_exc,
                )
                return ["未能成功抓取新闻"]

    async def _fetch_news_from_eastmoney_search_async(self, symbol: str) -> list[str]:
        import json as _json

        url = "https://search-api-web.eastmoney.com/search/jsonp"
        # 注意：原方法用 name 作 keyword，async 路径改成 symbol（避免依赖行情结果）
        param = {
            "uid": "",
            "keyword": symbol,
            "type": ["cmsArticleWebOld"],
            "client": "web",
            "clientType": "web",
            "clientVersion": "curr",
            "param": {
                "cmsArticleWebOld": {
                    "searchScope": "default",
                    "sort": "default",
                    "pageIndex": 1,
                    "pageSize": 5,
                }
            },
        }
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(
                url,
                params={
                    "cb": "jQuery",
                    "param": _json.dumps(param, ensure_ascii=False),
                },
                headers={"User-Agent": "Mozilla/5.0"},
            )
        text = r.text
        start = text.index("(") + 1
        end = text.rindex(")")
        data = _json.loads(text[start:end])
        articles = data.get("result", {}).get("cmsArticleWebOld", [])
        titles = [a["title"] for a in articles if "title" in a]
        return (
            [re.sub(r"<[^>]+>", "", t) for t in titles]
            if titles
            else ["未能成功抓取新闻"]
        )
