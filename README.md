# ResearchAgent — 投研多智能体流水线

> 面向 A 股个股的端到端投资研究 Agent：给一个股票代码，由五个阶段 Agent
> 协作产出**一份结构完整的研报草稿**，经人工风控签字后发布，并回灌进 RAG 语料库，
> 供「问已发布研报」的问答（轨道 B）检索。
>
> 本工程是投研分支，**已从旧工程重构而来**：移除了旧版 QA / Exam /
> Resume / Interview 四个 Agent，也**去掉了 LangChain / LangGraph** —— 流水线改由
> 表驱动引擎（`runner.py`）+ 显式段执行器（`steps.py`）+ 契约闸门（`protocols.py`）
> 驱动。核心大模型：**Qwen（通义千问/百炼）—— 六条 Agent 链路的结构化输出全走它，
> 实测支持 json_schema(strict)**；DeepSeek 端点已登记但当前无 Agent 路由到它。

---

## 一、这个项目在做什么

| 阶段 | 目录 | 职责 | 产出 |
|------|------|------|------|
| ① 采集 collect | `backend/agents/collect/` | 拉取个股四路数据（公告 / 财报 / 行业 / 新闻） | `research_data_items` |
| ② 分析 analyze | `backend/agents/analyze/` | 基本面/行业/情绪/技术**四维并发**打分 | `dimension_analyses` |
| ③ 检索 retrieve | `backend/agents/retrieve/` | 混合检索 + 精排，提炼可比对要点 | `comparison_points` |
| ④ 估值 valuation | `backend/agents/valuation/` | DCF 与可比法并发估值，合成结论 | `valuation_results` |
| ⑤ 风控 risk | `backend/agents/risk/` | 合规预检 → 复核 → 落草稿 → **人工签字** | 草稿 / `pipeline_checkpoints` |
| ⑥ 发布 | `runner.py` 终态分支 | 把已签发草稿翻成 published，回灌语料库 | `research_reports` |
| ⑦ 问答（轨道 B） | `backend/agents/qa/` | 问已发布研报的 RAG 问答（会话记忆 + 画像） | `qa_sessions` / `qa_messages` |

- **表驱动引擎**：`runner.py` 的 `_drive` 用 while 循环沿阶段表线性推进，取代了
  LangGraph 的父图 + checkpointer。暂停点只记在业务表 `pipeline_checkpoints`
  （「停在哪、等什么」），正文都在业务表，不再靠 Graph 状态不回传。
- **人工在环（HITL）**：风控段是唯一「停或续」的段 —— 写草稿、写暂停点后挂起，
  等审阅人签字（`risk-decision` 接口）才继续或驳回；驳回后重跑 analyze/valuation。
- **容错取向**：采集/估值四路失败可部分交付（记 errors、能用的继续用）；唯独**风控失败必须阻断**（`failed`），宁可不发布 —— 这是监管强制的直接体现。

---

## 二、技术栈

| 层面 | 选型 |
|------|------|
| 语言 | Python 3.11（严格锁定）+ TypeScript |
| 后端 | FastAPI + uvicorn + sse-starlette（SSE 流式）+ asyncpg + SQLAlchemy(2, asyncio) |
| 流水线 | **无 LangGraph / LangChain** —— 表驱动 `runner._drive` + `steps.run_stage` |
| LLM | **Qwen/百炼（qwen-turbo）**——六链路结构化输出，实测支持 json_schema(strict)；DeepSeek 端点已登记但未路由 |
| LLM 运输 | openai 官方 SDK 打 Qwen / DeepSeek 的 OpenAI 兼容接口 |
| 向量库 | Milvus（单集合 `report_corpus`，多租户用 `tenant_id` 过滤隔离） |
| 嵌入/精排 | BGE-M3（dense+sparse 双输出，进程内单例）+ BGE-Reranker-large |
| 意图分类 | MiniLM-L6-v2（进程内三层分类） |
| 真实财报 | baostock（可选，数据源 mixed 模式） |
| 前端 | Vue 3 + TypeScript + Element Plus + Pinia + Vite |
| 协议 | FastMCP（`/mcp/kb` 知识库检索、`/mcp/web-search` 联网搜索） |

> BGE-M3 / BGE-Reranker / MiniLM **三个本地模型均为进程内调用**，无需单独起服务，
> 启动时随后端进程预热模型权重。权重在 `backend/models/`（不入库，`.gitignore`）。

---

## 三、目录结构

```
ResearchAgent-refactored/
├── backend/
│   ├── main.py                  # FastAPI 入口：lifespan 建表 + 模型预热 + MCP 子应用
│   ├── config.py                # 所有配置，从 .env.local 读取（无默认值=必填）
│   ├── api/v1/                  # auth / companies / research / qa 模块
│   ├── agents/
│   │   ├── research/            # 引擎层：runner / steps / protocols / routing / state
│   │   ├── collect|analyze|retrieve|valuation|risk   # 五阶段各一子目录
│   │   └── qa/                  # 轨道 B：问已发布研报
│   ├── core/                    # llm_factory / research_repo / report_ingest /
│   │                            #   embedding / reranker / query_classifier / memory ...
│   ├── db/migrations.py         # 启动自动跑幂等 DDL（补建 pipeline_checkpoints / qa_*）
│   └── mcp/                     # FastMCP 子应用 + 自定义 JSON-RPC client
├── backend/agents/*/fixtures/    # 采集四源 fixture(12公司) + 检索语料种子(report_corpus)
├── frontend/                    # Vue3 SPA（登录 / 工作台 / 研报明细 / 风控台 / 研报问答）
├── scripts/                     # 建库 / 重建 Milvus / 灌种子 / 导入研报 / 离线评测 / FAQ
├── samples/                     # 示例财报 PDF / Markdown（研报导入用）
├── requirements.txt
├── docker-compose.yml           # postgres / etcd / minio / milvus / attu
└── .env.local                   # 本地配置（模板见 §五 步骤1）
```

---

## 四、环境要求

- **Python 3.11**（严格锁定，不兼容其他版本）
- **Docker & Docker Compose**
- 能联网下载模型权重；或已有 `backend/models/` 权重
- **Qwen（百炼）API Key**（必填）：六条 Agent 链路的结构化输出全走 qwen 网关，实测支持
  json_schema(strict)；DeepSeek 网关不识别 json_schema，结构性输出会 400 挡掉
- **DeepSeek API Key**（可选）：`llm_factory` 已登记端点，但当前无任何 Agent 路由到它

---

## 五、快速开始（拉到代码后按顺序做）

### 1. 配置环境变量（★首次必做）

根目录**没有** `.env.example`（已 .gitignore），需手动新建 `.env.local`，参考最小模板：

```ini
# ===== 数据库（PostgreSQL）=====
DB_HOST=localhost
DB_PORT=5433
DB_NAME=researchagent
DB_USER=researchagent_user
DB_PASSWORD=你的PG密码

# ===== Milvus =====
MILVUS_HOST=localhost
MILVUS_PORT=19531

# ===== 通义千问（必填：全部结构化输出走它，实测支持 json_schema strict）=====
QWEN_API_KEY=sk-xxxxxxxx
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen-turbo

# ===== DeepSeek（可选：端点已登记，当前无 Agent 路由到它）=====
DEEPSEEK_API_KEY=sk-xxxxxxxx
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL_CHAT=deepseek-chat
DEEPSEEK_MODEL_CODER=deepseek-coder

# ===== JWT 认证 =====
JWT_SECRET_KEY=任意一段随机字符串
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=10080
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7

# ===== 本地模型路径（相对 backend/）=====
RERANKER_MODEL_PATH=models/reranker/bge-reranker-large
BGE_M3_MODEL_PATH=models/embedding/bge-m3

# ===== 应用 / 数据源 =====
APP_ENV=local
APP_DEBUG=true
APP_PORT=8000
LOG_LEVEL=INFO
DEFAULT_TENANT_ID=tenant_default
RESEARCH_DATA_SOURCE_MODE=mixed   # 新闻走 Tavily 真实搜索；全离线可改 fixture
```

启动必填只有 `db_user` / `db_password` / `jwt_secret_key`（`config.py` 无默认值，缺了启动报错）。
  LLM 的 key：**Qwen 必配**——六条链路结构化输出全走它，key 无效流水线跑不通；**DeepSeek
  已改为可选**（默认空串，端点虽登记但当前无 Agent 使用，不需要填）。前端在
  `frontend/.env.local` 配后端直连地址（SSE 需直连 8000，见 `frontend/.env.example`）。

### 2. 起基础设施（Postgres + Milvus 等）

```bash
docker-compose --env-file .env.local up -d
```

> 端口：Postgres **5433**、Milvus **19531**、Milvus 管理台 attu **30000**。
> docker 数据卷 `postgres_data` 持久化，首次启动自动执行 `scripts/init_db.sql` 建表。

### 3. 建 Python 环境

```bash
conda create -n research_agent python=3.11 -y
conda activate research_agent
pip install -r requirements.txt
```

### 4. 灌种子数据（首次 / 数据重建后）

```bash
python scripts/seed_data.py        # 三大角色测试账号 + 研究标的（公司表）
python scripts/init_milvus.py      # 建 report_corpus 集合并灌检索语料种子
                                   #   = 各公司样例 + 12 家公司×11 行业的已发布研报种子
```

> 重建数据库 + Milvus 的顺序：`psql -f scripts/reset_research_db.sql`（DROP 重建投研库）→ `python scripts/init_milvus.py`。已发布研报种子
> `published_reports.json` 会被 init_milvus 自动回灌，跨重建存活，但**它是手写的
> 构造演示文本**，不是流水线真实产出。

### 5. 启动后端

```bash
# 在项目根目录下
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

首次启动 lifespan 会自动执行幂等迁移补全业务表，并预热 BGE-M3 / Reranker / MiniLM
三个模型（约数秒~十几秒）。

### 6. 启动前端

```bash
cd frontend
npm install
# 先配 frontend/.env.local（VITE_API_BASE_URL=http://localhost:8000）
npm run dev        # http://localhost:3000
```

---

## 六、服务端口

| 服务 | 端口 | 说明 |
|------|------|------|
| FastAPI 后端 | **8000** | REST + SSE，`/docs` 看 Swagger，`/health` 探活 |
| Vue3 前端 | **3000** | 登录 / 工作台 / 研报明细 / 风控台 / 研报问答 |
| PostgreSQL | **5433** | 宿主机 5432 已被占，隔离到 5433 |
| Milvus | **19531** | Milvus 向量库（宿主机 19530 已被占） |
| Attu（Milvus 台） | **30000** | Milvus 管理界面 |
| MCP 子应用 | 挂载在 8000 | `/mcp/kb`、`/mcp/web-search` |

> 所有连接配置统一从 `.env.local` 读取，**禁止硬编码端口**。

---

## 七、跑单个任务的 API（核心链路）

```bash
# 提交一个研报研究任务（如 600519.SH 茅台）
POST /api/v1/research/tasks   body: {"company_code": "600519.SH"}

# 轮询任务状态 / 阶段事件
GET  /api/v1/research/tasks/{id}/events
GET  /api/v1/research/tasks/{id}

# 风控人工决策（继续 / 驳回）
POST /api/v1/research/tasks/{id}/risk-decision

# 读最终研报草稿 / 报告
GET  /api/v1/research/tasks/{id}/report

# 问已发布研报（轨道 B QA）
POST /api/v1/qa/chat
POST /api/v1/qa/chat/stream   # SSE 流式
```

---

## 八、常用脚本

| 脚本 | 作用 |
|------|------|
| `scripts/init_db.sql` | 建投研库全部业务表（Postgres 首启自动执行） |
| `scripts/reset_research_db.sql` | **一次性破坏性重建**投研库（DROP+CREATE） |
| `scripts/init_milvus.py` | 建 report_corpus 向量集合 + 灌检索引例句料与已发布研报种子；`--no-seed-reports` 跳过灌数据 |
| `scripts/seed_data.py` | 灌测试账号 + 研究标的公司 |
| `scripts/build_knowledge_base.py` | 研报语料导入（.md/.pdf 切块 → report_corpus），库模块，调 `build_report_pipeline(file_path, ...)` |
| `scripts/run_retrieval_eval.py` | 轨道 B 离线检索/回答质量评估（需真实 Milvus，不进 pytest） |
| `scripts/consolidate_pending_faq.py` | 待补料 → FAQ 闭环 |
| `scripts/sink_agentic_answers.py` | agentic 答案沉淀桶回灌 |

---

## 九、可能遇到的坑

- **启动报必填缺失**：`.env.local` 缺 `DB_USER / DB_PASSWORD / JWT_SECRET_KEY` 会 config 校验失败、启动报错（DeepSeek key 已设为可选默认空串，不再强制）。**Qwen key 无效或为空**则后端能起来但六链路结构化调用全部失败——跑研报前先确认 Qwen 可用。
- **`docker-compose --env-file` 找不到文件**：先建 `.env.local` 再 `up`。
- **MCP 版本勿动**：`requirements.txt` 锁定 mcp==1.30.0，先升 2.x 会顶 pydantic/starlette、fastapi 起不来。
- **pip 环境残留 LangGraph/LangChain 但不影响**：仓库已移除依赖与源码引用，残留只是以前安装没卸载。
- **前端 SSE 直连 8000**：不经过 Vite proxy，须在 `frontend/.env.local` 指向后端地址。

---

## 十、License

本工程仅供学习 / 演示使用，研报结论与估值均为演示构造，**不构成任何投资依据**。
未经授权不得用于商业用途。