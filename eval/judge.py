import json
import logging
from dataclasses import dataclass
from openai import OpenAI
from stockg.infrastructure import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL
)

logger = logging.getLogger("eval.judge")


@dataclass
class JudgeVerdict:
    scores: dict[str, int]   # {"rating_consistency":4,"groundedness":5,"completeness":3,"compliance":5}
    overall: float           # 四项均值
    passed: bool             # overall >= threshold
    reasoning: str           # 判官的 CoT 分析
    issues: list[str]        # 具体问题点

JUDGE_SYSTEM = '你是一个严苛的金融分析报告评审员，只输出JSON'
JUDGE_PROMPT = """请评审下面这份股票分析报告。
【用户问题】
{input}

【Agent实际抓取到的真实数据(行情/新闻/知识库)】
{fetched_context}

【待评审报告】
评级: {rating}
理由: {reason}

请先在 reasoning 字段写出你的分析过程，然后按以下四维度打分(1=极差，5=优秀)。
每个维度必须严格对照下方分值锚点判定，不要凭感觉打分。

=== 维度1: rating_consistency (评级自洽性) ===
5分: 评级与数据方向完全一致(涨+利好→买入; 跌+利空→卖出; 震荡无催化→观望)
4分: 方向正确, 但评级措辞与数据强度略有偏差
3分: 方向勉强成立, 但存在矛盾数据被忽略
2分: 评级与数据方向明显不符
1分: 评级与数据完全对立(如跌停+利空却给强烈买入)

=== 维度2: groundedness (事实忠实度) ===
5分: 理由中所有价格/涨跌幅/新闻均能在上方真实数据找到原文对应
4分: 数据基本准确, 有1处近似表述不影响结论
3分: 有1处数据无法在真实数据中找到, 但属合理常识推断
2分: 有1处明确捏造但非核心论点
1分: 核心论点基于捏造数据(如虚构重大新闻支撑买入)

=== 维度3: completeness (三要素完整性) ===
5分: 三要素全覆盖(价格解读+新闻情绪+风险/催化剂)
4分: 三要素都有, 但其中1项过于简略
3分: 覆盖2/3要素, 缺失1项
2分: 覆盖1/3要素, 只有价格或只有新闻
1分: 三要素全无, 只给评级无分析

=== 维度4: compliance (合规性) ===
5分: 客观中性+明确风险提示+无任何承诺性表述
4分: 无违规承诺, 有风险提示但措辞偏弱
3分: 无违规承诺, 但完全缺失风险提示
2分: 含轻微承诺性表述(如"大概率上涨"), 无风险提示
1分: 含明确违规承诺(如"必涨""保证收益""稳赚不赔")

严格只输出如下JSON，不要输出任何其他文字:
{{
    "reasoning": "逐维度分析过程, 每个维度说明依据哪条锚点判定",
    "scores": {{
        "rating_consistency": 1, "groundedness": 1, "completeness": 1, "compliance": 1
    }},
    "issues": ["问题1", "问题2"]
}}
"""

class LLMJudge:
    def __init__(self, model: str="deepseek-chat", threshold: float=3.5):
        self.model = model
        self.threshold = threshold
        self.client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL
        )
    
    def judge(self, input: str, fetched_context: list[str], rating: str, reason: str) -> JudgeVerdict:
        promopt = JUDGE_PROMPT.format(input=input, fetched_context=fetched_context, rating=rating, reason=reason)
        for attempt in range(2):
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role":"system", "content": JUDGE_SYSTEM},
                {"role":"user", "content": promopt}],
                temperature=0,
                response_format={"type": "json_object"}
            )
            raw = resp.choices[0].message.content
            try:
                data = json.loads(raw)
                break
            except json.JSONDecodeError:
                if attempt == 0:
                    logger.warning("Judge 返回的 JSON 解析失败，将重试")
                else:
                    logger.error("Judge 返回的 JSON 再次解析失败，使用默认评分")
                    data = {"reasoning": "JSON 解析失败", "scores": {"rating:consistency": 1.0, "groundedness": 1.0, "completeness": 1.0, "compliance": 1.0}, "issues": []}
        scores = {k: int(v) for k, v in data["scores"].items()}
        overall = sum(scores.values()) / len(scores)
        return JudgeVerdict(
            scores=scores,
            overall=round(overall, 2),
            passed=overall >= self.threshold,
            reasoning=data.get("reasoning", ""),
            issues=data.get("issues", [])
        )

