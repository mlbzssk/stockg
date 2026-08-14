import json
import logging

from stockg.domain import run_industrial_agent
from stockg.logging_config import configure_logging

logger = logging.getLogger("eval.capture")

CAPTURE_CASES = [
    {"id": "a-600519", "market": "a", "ticker": "600519"},  # 茅台
    {"id": "a-000001", "market": "a", "ticker": "000001"},  # 平安
    {"id": "us-aapl", "market": "us", "ticker": "AAPL"},
    {"id": "us-tsla", "market": "us", "ticker": "TSLA"},
    {"id": "a-999999", "market": "a", "ticker": "999999"},
]


def capture(out_path="eval/cases.jsonl"):
    cases = []
    for c in CAPTURE_CASES:
        label = "A股" if c["market"] == "a" else "美股"
        query = f"请分析{label}标的{c['ticker']}，先获取行情与新闻，给出买入/观望/卖出评级与理由"
        try:
            r = run_industrial_agent(query, market=c["market"], thread_id=None)
            cases.append(
                {
                    "id": c["id"],
                    "input": query,
                    "market": c["market"],
                    "status": r.status,
                    "rating": r.rating,
                    "reason": r.reason,
                    "fetched_context": r.fetched_context,
                    "cost": r.cost,
                }
            )
            logger.info(
                "用例采集完成: case_id=%s status=%s rating=%s cost=%s",
                c["id"],
                r.status,
                r.rating,
                r.cost,
            )
        except Exception as exc:
            cases.append(
                {
                    "id": c["id"],
                    "input": query,
                    "rating": "",
                    "market": c["market"],
                    "status": "failed",
                    "reason": str(exc),
                    "fetched_context": [],
                    "cost": {},
                }
            )
            logger.exception("用例采集失败: case_id=%s", c["id"])

        with open(out_path, "w", encoding="utf-8") as file:
            for case in cases:
                file.write(json.dumps(case, ensure_ascii=False) + "\n")
        logger.info("采集进度已保存: count=%s path=%s", len(cases), out_path)


if __name__ == "__main__":
    configure_logging()
    capture()
