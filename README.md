# stockg

基于实时行情、新闻、本地知识库与 DeepSeek 的 A 股 / 美股分析 Agent。项目提供 CLI 与 FastAPI 两种入口，可输出结构化的「买入 / 观望 / 卖出」评级、分析理由、工具调用上下文及模型成本估算。

> **免责声明**：本项目仅用于技术研究与信息辅助，不构成投资建议，不保证数据实时性、完整性或准确性，也不提供自动交易能力。

## 核心能力

- **双市场行情**：A 股通过新浪获取行情、失败时降级至东方财富；美股通过 Finnhub 获取行情与近 7 天新闻。
- **多 Agent 分析**：主 Agent 调度资料收集 Agent，并通过系统提示要求先获取行情、新闻和知识库资料，再提交结构化报告。
- **本地 RAG**：递归导入 PDF、Markdown、TXT，使用 `BAAI/bge-small-zh-v1.5` 向量化并持久化至 Chroma。
- **同步与异步入口**：CLI 使用同步流程，HTTP API 使用异步流程并并发收集资料。
- **多轮会话**：API 通过 `session_id` 复用上下文；会话保存在进程内，默认 1 小时过期。
- **可观测与成本控制**：集成 Langfuse；单会话默认限制 50,000 tokens、估算费用 ¥0.5。
- **报告评测**：支持自定义 LLM Judge 与 DeepEval，从评级自洽、事实忠实、内容完整、合规性四个维度评估报告。

## 工作流程

```text
用户问题
   │
   ▼
主分析 Agent ──► 识别/查询股票代码
   │
   ▼
资料收集 Agent ─┬─► 实时行情
                ├─► 最新新闻
                └─► 本地 Chroma 知识库
   │
   ▼
DeepSeek 综合分析
   │
   ▼
评级 + 理由 + 原始上下文 + 成本估算
```

### 独立 Plan-and-Execute 资料收集流程

`stockg.domain.plan_execute_research_agent` 提供独立的实验性资料收集流程，不接入现有主分析 Agent。流程由 Planner 生成最多 4 个受约束步骤，Executor 并发执行行情与知识库检索，Replanner 最多补充执行 1 轮；模型不可用时会降级为确定性计划。

```bash
uv run python -m stockg.domain.plan_execute_research_agent AAPL \
  --market us \
  --objective "收集近期经营风险和潜在催化剂" \
  --rag-top-k 3
```

不传参数时默认研究 A 股代码 `600519`。该入口只汇总资料，不生成投资评级，也不会改变 CLI 和 HTTP API 当前使用的主流程。

## 环境要求

- Python `>= 3.10`
- [uv](https://docs.astral.sh/uv/)（推荐）或 pip
- 可访问 DeepSeek、Finnhub、行情源及 Hugging Face（首次下载 BGE 模型）

## 快速开始

### 1. 安装

```bash
git clone <repository-url>
cd stockg
uv sync
```

也可使用 pip：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

`pyproject.toml` 与 `uv.lock` 是当前完整、可复现的依赖定义；根目录 `requirements` 仅保留了部分早期依赖，不建议作为首选安装方式。

### 2. 配置

应用启动时会自动加载项目根目录的 `.env`。先复制示例文件并填写密钥：

```bash
cp .env.example .env
```

至少需要配置 `DEEPSEEK_API_KEY`；获取真实美股行情和新闻时还需要 `FINNHUB_API_KEY`，未配置时流程会使用错误上下文降级继续。`DEEPSEEK_BASE_URL`、`DEEPSEEK_MODEL`、`RAG_STORE_PATH` 和 `LOG_LEVEL` 均可按需调整；Langfuse 配置为可选项。

Shell、CI、容器或部署平台注入的环境变量优先级高于 `.env`，因此生产环境无需创建 `.env`。修改 `.env` 后需要重启应用。

应用统一使用 Python 标准日志并输出到 stderr，格式包含时间、级别和模块名。默认级别为 `INFO`；排查 Agent 迭代、模型成本等细节时可临时设置 `LOG_LEVEL=DEBUG`。本地日志会记录会话 ID、市场、工具调用、重试、降级和成本摘要，但不会记录完整用户问题、会话消息或模型流式原文。启用 Langfuse 后，远程追踪可能包含模型消息和工具输入输出，生产环境应按隐私要求配置采集及数据保留策略。

> **安全提示**：不要提交真实 API Key。根目录 `.gitignore` 已忽略 `.env`、私钥、证书和常见本地凭据文件，`.env.example` 只包含占位符。生产环境应优先使用部署平台的密钥管理能力。如果源码中的旧凭据曾被提交、推送或共享，请立即在对应服务端轮换。

### 3. 分析股票

```bash
# A 股：6 位数字代码
uv run stockg analyze 600519

# 美股：英文 ticker
uv run stockg analyze AAPL
```

省略子命令或股票代码时，默认分析 `600519`：

```bash
uv run stockg
```

CLI 将 `0/3/6/8` 开头的 6 位数字识别为 A 股，其余输入按美股处理。

## 本地知识库

### 导入材料

```bash
# 单个文件
uv run stockg ingest ./docs/report.pdf

# 递归导入目录中的 .pdf / .txt / .md
uv run stockg ingest ./docs
```

默认按 400 字符切分、重叠 50 字符，数据持久化到 `.rag_store/`。首次使用会下载 `BAAI/bge-small-zh-v1.5`；若模型不可用，将自动退化为 256 维哈希向量，流程仍可运行，但检索质量会明显下降。

### 独立检索

```bash
uv run stockg ask "贵州茅台的主要经营风险" --top-k 5
```

分析 Agent 也会自动检索同一知识库；知识库为空不会阻塞行情分析。

## HTTP API

启动开发服务：

```bash
uv run stockg serve
```

服务默认监听 `0.0.0.0:8000`，并开启热重载：

- Swagger UI：<http://localhost:8000/docs>
- 健康检查：`GET /health`
- 对话分析：`POST /chat`

```bash
curl -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "message": "分析 AAPL 最近的表现并给出评级"
  }'
```

响应示例：

```json
{
  "status": "success",
  "session_id": "b4b79c67-...",
  "message": "分析理由",
  "rating": "观望",
  "fetched_context": ["..."],
  "cost": {
    "tool_tokens": 1234,
    "cost_yuan": 0.01
  }
}
```

继续传入响应中的 `session_id` 即可进行多轮对话：

```json
{
  "message": "再说明一下主要风险",
  "session_id": "b4b79c67-..."
}
```

`market` 是可选参数，默认值为 `auto`，通常无需传入。自动模式会优先根据标准股票代码确定市场（六位 A 股代码走 A 股数据源，英文 ticker 走美股数据源），名称、代码或市场不明确时再结合股票搜索和 Agent 结果判断。仅需强制指定市场时传入 `a` 或 `us`，显式值具有最高优先级。

## 单元测试

单元测试位于 `tests/`，覆盖领域实体、成本预算、会话隔离、市场识别、RAG 入库与检索、Research Agent、主 Agent、数据仓储降级、HTTP 重试、模型流式响应及 FastAPI 接口。测试通过 fake、mock 和临时目录隔离外部模型、网络、向量库与文件系统，不需要配置 API Key，也不会发起真实外部请求。

```bash
uv run pytest tests -q
```

提交前建议同时执行语法、离线单元测试和代码规范检查。Ruff 已作为开发依赖锁定，lint 会检查关键 PEP 8 问题、未定义或未使用符号以及导入顺序：

```bash
uv run python -m compileall -q stockg
uv run pytest tests
uv run ruff check .
uv run ruff format --check .
```

## 评测

评测依赖 DeepSeek，会产生模型调用费用。样例位于 `eval/cases.jsonl`，每行一个 JSON 对象，核心字段如下：

```json
{
  "id": "case-id",
  "status": "success",
  "input": "用户问题",
  "rating": "观望",
  "reason": "分析理由",
  "fetched_context": ["Agent 实际获取的数据"],
  "expect_fail": false
}
```

### 自定义 LLM Judge

按 1～5 分评估评级自洽性、事实忠实度、三要素完整性和合规性，四项平均分不低于 3.5 即通过：

```bash
uv run python -m eval.run_eval
```

### DeepEval + pytest

```bash
uv run pytest judges -v

# 生成独立 HTML 报告
uv run pytest judges -v \
  --html=eval_report.html \
  --self-contained-html
```

仓库中的 `eval_report.html` 是已有评测报告。`eval/capture.py` 是在线采集用例脚本，采集过程会逐条覆盖写入 `eval/cases.jsonl`，可直接供 `run_eval.py` 和 DeepEval 测试读取。

## 项目结构

```text
.
├── main.py                         # CLI 转发入口
├── pyproject.toml                  # 项目、依赖与 stockg 命令定义
├── stockg/
│   ├── domain/
│   │   ├── agent.py                # 主分析 Agent（同步/异步）
│   │   ├── research_agent.py       # LangGraph 异步编排行情、新闻与 RAG 资料收集
│   │   ├── entities.py             # 领域实体与结果结构
│   │   ├── ports.py                # 仓储、Loader、Embedder 等抽象端口
│   │   ├── session.py              # 进程内会话与快照缓存
│   │   └── cost.py                 # Token 与费用预算
│   ├── application/rag_service.py  # RAG 入库与检索用例
│   ├── infrastructure/
│   │   ├── config.py               # 模型、数据源与存储配置
│   │   ├── akshare_stock_repository.py
│   │   ├── finnhub_stock_repository.py
│   │   └── rag/                    # 加载、切分、向量化、Chroma、召回
│   └── interfaces/
│       ├── cli.py                   # analyze / ingest / ask / serve
│       └── api.py                   # FastAPI /chat 与 /health
├── eval/                            # 样例采集与自定义 LLM Judge
├── judges/                          # DeepEval 指标与 pytest 用例
└── eval_report.html                 # HTML 评测报告
```

## 数据源与降级策略

| 市场/能力 | 主数据源 | 降级策略 |
| --- | --- | --- |
| A 股行情 | 新浪 | 东方财富 |
| A 股新闻 | 东方财富搜索 | AkShare `stock_news_em` |
| 美股行情 | Finnhub `quote` | 返回空行情及错误说明 |
| 美股新闻 | Finnhub `company-news` | 返回错误说明 |
| 中文向量 | BGE Small Zh v1.5 | 本地哈希向量 |
| 股票名称/代码搜索 | 东方财富搜索 | 返回可读错误，不中断 Agent |

外部接口可能因限流、休市、地区网络或上游变更而失败；项目会尽量将失败信息作为上下文交给 Agent，而不是直接中断全部流程。

## 当前限制

- 仅输出研究性评级，不含账户、持仓、回测、交易执行或风控系统。
- 行情与新闻依赖第三方公开接口，不保证实时性和长期兼容性。
- 会话仅保存在单个 Python 进程内；服务重启、多进程或多实例之间不会共享。
- 自动市场识别依赖模型理解和东方财富股票搜索结果；名称存在多个候选时，识别结果仍可能有歧义。
- 成本金额按代码中的固定 DeepSeek 单价估算，可能与实际账单不同。
- RAG 不会自动去重或清理旧文档，重复导入可能产生重复片段。
