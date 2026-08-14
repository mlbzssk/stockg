import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)
DEEPSEEK_PRICE = {"input": 2.0, "output": 8.0}


class CostBudgetExceeded(Exception):
    pass


@dataclass
class CostTracker:
    max_total_tokens: int = 50_000
    max_total_cost_yuan: float = 0.5
    max_iterations: int = 5

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_cost_yuan: float = 0.0
    calls: list[dict] = field(default_factory=list)

    def record(self, usage, label):
        if usage is None:
            return
        p = usage.prompt_tokens or 0
        c = usage.completion_tokens or 0
        t = usage.total_tokens or (p + c)
        cost = (p / 1_000_000) * DEEPSEEK_PRICE["input"] + (
            c / 1_000_000
        ) * DEEPSEEK_PRICE["output"]

        self.prompt_tokens += p
        self.completion_tokens += c
        self.total_tokens += t
        self.total_cost_yuan += cost

        self.calls.append(
            {
                "label": label,
                "prompt_tokens": p,
                "completion_tokens": c,
                "total_tokens": t,
                "cost_yuan": round(cost, 6),
            }
        )
        logger.debug(
            "模型成本: label=%s prompt_tokens=%s completion_tokens=%s total_tokens=%s accumulated_tokens=%s accumulated_cost_yuan=%.6f",
            label,
            p,
            c,
            t,
            self.total_tokens,
            self.total_cost_yuan,
        )
        self.check()

    def check(self):
        if self.total_tokens > self.max_total_tokens:
            raise CostBudgetExceeded(f"Total tokens exceeded {self.max_total_tokens}")
        if self.total_cost_yuan > self.max_total_cost_yuan:
            raise CostBudgetExceeded(f"Total cost exceeded ¥{self.max_total_cost_yuan}")

    def reset(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.total_cost_yuan = 0.0
        self.calls = []


global_tricker = CostTracker()
