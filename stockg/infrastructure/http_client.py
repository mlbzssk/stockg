"""对 requests 的封装: 带退避重试的 GET-JSON。"""

from __future__ import annotations

import asyncio
import logging
import random
import time

import httpx
import requests

logger = logging.getLogger(__name__)


def get_json(
    url: str,
    params: dict,
    max_retries: int = 6,
    sleep: float = 1.0,
    max_sleep: float = 30.0,
) -> dict:
    """带指数退避重试的 GET-JSON。

    本机出口经 Whistle 代理, 东财接口经常偶发 502; 指数退避 + 随机抖动可规避限流,
    避免固定间隔重试加剧封禁。
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Referer": "https://quote.eastmoney.com/",
    }
    last = None
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=15)
            if r.status_code == 200 and r.text.strip().startswith("{"):
                return r.json()
            last = f"HTTP {r.status_code}"
        except Exception as e:  # noqa: BLE001 - 统一重试
            last = f"{type(e).__name__}: {e}"
        if attempt < max_retries:
            # 指数退避: 1, 2, 4, 8, 16... 封顶 max_sleep; equal jitter 保证至少一半基础退避
            base = min(sleep * 2 ** (attempt - 1), max_sleep)
            delay = base / 2 + random.uniform(0, base / 2)
            logger.warning(
                "HTTP 请求失败，将重试: attempt=%s/%s error=%s delay=%.1fs",
                attempt,
                max_retries,
                last,
                delay,
            )
            time.sleep(delay)
        else:
            logger.error(
                "HTTP 请求失败，已达最大重试次数: attempt=%s/%s error=%s",
                attempt,
                max_retries,
                last,
            )
    raise RuntimeError(f"多次重试后仍无法获取 {url}：{last}")


async def get_json_async(
    url: str,
    params: dict,
    max_retries: int = 6,
    sleep: float = 1.0,
    max_sleep: float = 30.0,
) -> dict:
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Referer": "https://quote.eastmoney.com/",
    }
    last = None
    async with httpx.AsyncClient(timeout=15.0) as client:
        for attempt in range(1, max_retries + 1):
            try:
                r = await client.get(url, params=params, headers=headers)
                if r.status_code == 200 and r.text.strip().startswith("{"):
                    return r.json()
                last = f"HTTP {r.status_code}"
            except Exception as e:  # noqa: BLE001 - 统一重试
                last = f"{type(e).__name__}: {e}"
            if attempt < max_retries:
                # 指数退避: 1, 2, 4, 8, 16... 封顶 max_sleep; equal jitter 保证至少一半基础退避
                base = min(sleep * 2 ** (attempt - 1), max_sleep)
                delay = base / 2 + random.uniform(0, base / 2)
                logger.warning(
                    "HTTP 请求失败，将重试: attempt=%s/%s error=%s delay=%.1fs",
                    attempt,
                    max_retries,
                    last,
                    delay,
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    "HTTP 请求失败，已达最大重试次数: attempt=%s/%s error=%s",
                    attempt,
                    max_retries,
                    last,
                )
        raise RuntimeError(f"多次重试后仍无法获取 {url}：{last}")
