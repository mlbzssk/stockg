import json
import logging

from eval.judge import LLMJudge
from stockg.logging_config import configure_logging

logger = logging.getLogger("eval.run_eval")


def load_cases(path="eval/cases.jsonl"):
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    configure_logging()
    judge = LLMJudge()
    cases = load_cases()
    results = []
    for c in cases:
        status = c.get("status", "success")
        if status != "success":
            logger.info("跳过评测用例: case_id=%s status=%s", c["id"], status)
            continue
        v = judge.judge(c["input"], c["fetched_context"], c["rating"], c["reason"])
        results.append((c["id"], v))
        logger.info(
            "评测用例完成: case_id=%s overall=%s passed=%s scores=%s",
            c["id"],
            v.overall,
            v.passed,
            v.scores,
        )
        for issue in v.issues:
            logger.info("评测问题: case_id=%s issue=%s", c["id"], issue)
    passed = sum(1 for _, v in results if v.passed)
    logger.info("评测完成: total=%s passed=%s", len(cases), passed)


if __name__ == "__main__":
    main()
