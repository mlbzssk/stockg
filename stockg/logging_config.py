"""应用日志配置。"""

from __future__ import annotations

import logging
import os

from stockg.infrastructure.config import load_environment

_DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(name)s - %(message)s"
_HANDLER_MARKER = "_stockg_handler"
_LOGGER_NAMES = ("stockg", "eval", "judges")


def configure_logging(level: str | None = None) -> None:
    """配置项目 stderr 日志；级别可由 ``LOG_LEVEL`` 环境变量覆盖。"""
    load_environment()
    level_name = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    log_level = getattr(logging, level_name, logging.INFO)
    if not isinstance(log_level, int):
        log_level = logging.INFO

    formatter = logging.Formatter(_DEFAULT_FORMAT)
    for logger_name in _LOGGER_NAMES:
        project_logger = logging.getLogger(logger_name)
        project_logger.setLevel(log_level)
        project_logger.propagate = False
        handler = next(
            (
                item
                for item in project_logger.handlers
                if getattr(item, _HANDLER_MARKER, False)
            ),
            None,
        )
        if handler is None:
            handler = logging.StreamHandler()
            setattr(handler, _HANDLER_MARKER, True)
            project_logger.addHandler(handler)
        handler.setLevel(log_level)
        handler.setFormatter(formatter)

    logging.captureWarnings(True)
