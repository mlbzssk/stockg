import os

from deepeval.metrics import FaithfulnessMetric, GEval
from deepeval.models import DeepSeekModel
from deepeval.test_case import SingleTurnParams

from stockg.infrastructure import DEEPSEEK_API_KEY

os.environ["DEEPSEEK_API_KEY"] = DEEPSEEK_API_KEY

judge = DeepSeekModel(
    model="deepseek-chat",
    api_key=DEEPSEEK_API_KEY,
    temperature=0,
)

rating_consistency = GEval(
    name="评级一致性",
    criteria="判断评级(买入/观望/卖出)是否与所给行情数据(股价、涨跌幅)和新闻面逻辑自洽。暴涨+利好却给卖出、暴跌+利空却给买入属严重不一致；数据中性给观望属合理。",
    evaluation_params=[
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.RETRIEVAL_CONTEXT,
    ],
    model=judge,
    threshold=0.6,
)

faithfulness = FaithfulnessMetric(threshold=0.7, model=judge)

completeness = GEval(
    name="三要素完整性",
    criteria=(
        "结合检索上下文，判断分析理由是否覆盖三要素：(1)价格/涨跌幅解读；"
        "(2)新闻面情绪；(3)短期风险或催化剂。若上下文明示某项数据不可用，"
        "报告准确说明该数据缺失及其对判断的影响，也视为覆盖该项，不应要求虚构数据；"
        "缺少可用数据时，等待后续行情或公司披露可视为合理催化条件。"
    ),
    evaluation_params=[
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.RETRIEVAL_CONTEXT,
    ],
    model=judge,
    threshold=0.6,
)

compliance = GEval(
    name="合规性",
    criteria="是否满足合规：(1)不含'保证收益/稳赚/必涨'等违规承诺(出现直接0分)；"
    "(2)客观不情绪化；(3)对高风险标的给出风险提示。",
    evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
    model=judge,
    threshold=0.8,
)

METRICS = [rating_consistency, faithfulness, completeness, compliance]
