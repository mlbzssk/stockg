"""从项目根目录 ``.env`` 和进程环境读取应用配置。

进程环境变量的优先级高于 ``.env``；密钥没有源码默认值。
"""

import os
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ENV_FILE = _PROJECT_ROOT / ".env"


def load_environment(env_file: Path = _ENV_FILE) -> None:
    """加载环境文件，但不覆盖 Shell 或部署平台已注入的变量。"""
    _ = load_dotenv(dotenv_path=env_file, override=False, encoding="utf-8")


load_environment()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")

# RAG 向量库（Chroma）的持久化目录。
RAG_STORE_PATH = os.getenv("RAG_STORE_PATH", str(_PROJECT_ROOT / ".rag_store"))
