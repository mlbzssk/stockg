import json
from math import e
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from metrics import METRICS

def load_cases(path="eval/cases.jsonl"):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

CASES = load_cases()


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_report_quality(case):
    tc = LLMTestCase(
        input=case["input"],
        actual_output=f"评级:{case['rating']}\n理由:{case['reason']}",
        context=case["fetched_context"],
        retrieval_context=case["fetched_context"],
    )
    try:
        assert_test(tc, METRICS)
    except Exception as e:
        if case.get("expect_fail"):
            pytest.xfail(f"[{case['id']}] failed: {e}")
        raise e
    