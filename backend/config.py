# backend/config.py
# 全项目唯一的「配置中心」：从 .env.local 读取所有配置项，供任何模块取用。

from pydantic_settings import BaseSettings   # Pydantic 的「配置基类」，能自动从环境变量/.env 读取并做类型校验
from functools import lru_cache              # 标准库装饰器：缓存函数结果，让函数实际只执行一次
import os, sys
live_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

env_local_path = os.path.join(live_path, ".env.local")
# print(f"env_local_path: {env_local_path}")

class Settings(BaseSettings):
    """配置模型：每个类属性对应 .env.local 里的一项配置。
    继承 BaseSettings 后，Pydantic 会自动把同名（大小写不敏感）的配置读进来并转成对应类型。"""

    # ── 数据库（PostgreSQL）──
    db_host: str = "localhost"   # 主机；写了默认值 = 可选项
    db_port: int = 5433          # 端口；本机 5432 已占用，隔离到 5433
    db_name: str = "researchagent"  # 库名
    db_user: str                 # 用户名；没有默认值 = 必填，.env.local 缺了会启动报错
    db_password: str             # 密码；同样必填

    @property
    def database_url(self) -> str:
        """把上面几个散件拼成 SQLAlchemy 需要的连接串。
        用 @property 装饰后，可像访问属性一样 settings.database_url 取值，不用加括号调用。

        :return: str，PostgreSQL 异步连接串（postgresql+asyncpg://…）。
        """
        return (
            f"postgresql+asyncpg://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    @property
    def checkpoint_dsn(self) -> str:
        """LangGraph 的 checkpointer 用的连接串（psycopg，不是 asyncpg）。

        与 database_url 是**同一个库**，只差驱动：psycopg 不认 SQLAlchemy 的
        `+asyncpg` 方言后缀，照搬 database_url 会在建池时报
        "invalid connection option"。所以单独给一份，而不是让调用方各自去
        replace("+asyncpg", "") —— 那种改法散在各处，改一处漏一处。

        :return: str，PostgreSQL 连接串（postgresql://…，无 asyncpg 后缀）。
        """
        return (
            f"postgresql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    # ── Milvus 向量库 ──
    milvus_host: str = "localhost"
    milvus_port: int = 19531     # 本机 19530 已占用，隔离到 19531

    # ── 大模型（DeepSeek）──
    # 可选：端点已登记但当前无 Agent 路由到它（结构化输出全走 qwen）。默认空串，
    # 不填不影响启动与跑研报；仅当未来某链路切回 deepseek 时才需要有效 key。
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"  # DeepSeek 接口地址
    deepseek_model_chat: str = "deepseek-chat"              # 对话模型名
    deepseek_model_coder: str = "deepseek-coder"            # 代码模型名

    # ── 大模型（通义千问 / 百炼，OpenAI 兼容）──
    # 投研分支结构化输出切到 qwen 网关：百炼兼容接口实测同时支持
    # json_object 与 json_schema(strict)，而 DeepSeek 网关只支持 json_object。
    # 换供应商只改这里三行 + llm_factory._AGENT_MODEL_ROUTING/_VENDOR_ENDPOINTS，不动结构化代码。
    qwen_api_key: str = ""                                  # 百炼 API Key（sk-...）
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_model: str = "qwen-turbo"                          # 通用稳定款，不新不旧

    # ── 本地模型权重路径 ──
    reranker_model_path: str = "./models/reranker/bge-reranker-large"    # 精排模型
    bge_m3_model_path: str = "./models/embedding/bge-m3"                 # 嵌入模型

    # ── 本地 Query 分类模型权重路径（QA 三层分类的 L2）──
    # 约定同 reranker：os.path.join(backend_path, ...) 落到 backend/models/...
    classifier_model_path: str = "./models/classifier/all-MiniLM-L6-v2"            # 基座
    finetuned_classifier_path: str = "./models/classifier/query-classifier-finetuned"  # 微调

    # ── JWT 认证 ──
    jwt_secret_key: str                           # 必填：签发登录令牌用的密钥
    jwt_algorithm: str = "HS256"                  # 签名算法
    jwt_access_token_expire_minutes: int = 10080  # 令牌有效期（分钟）

    # ── 投研域业务规则（设计文档第 12 章的「待标定」项集中在此，便于调参）──
    # 这些是经验值，不是实测值。面试时讲「我设计了可调的权重与标定方法」比讲一个
    # 拍脑袋的数字更可信 —— 调参只改配置，不动代码。
    research_max_redo: int = 3                  # 驳回重做的次数上限，防死循环
    research_token_budget: int = 100000         # 单研报任务累计 token 上限；超限中止任务(failed)。0=不限
    research_evidence_min_count: int = 3        # 单个维度的最少证据条数
    research_timeliness_full_days: int = 30     # 多少天以内视为「最新」→ 权重 1.0
    research_timeliness_decay_days: int = 365   # 超过阈值后的线性衰减跨度
    research_timeliness_floor: float = 0.3      # 时效权重下限
    research_timeliness_valid: float = 0.5      # 「有效证据」的时效阈值
    research_rating_buy_threshold: float = 70.0   # 综合分 ≥ 此值 → 买入
    research_rating_sell_threshold: float = 45.0  # 综合分 < 此值 → 卖出

    # ── 轨道B 问答引擎：proverty 确定性管线 / agentic 多轮自搜 ──
    # 默认 pipeline（现有确定性单次检索）；agentic 是完整的 LLM 自搜循环（可选，显式打开才走）。
    # 用同一个 env 前缀，.env.local 设 QA_ENGINE_MODE=agentic 即切换。
    qa_engine_mode: str = "pipeline"     # pipeline(现在这套) / agentic(多轮自搜)
    qa_agent_max_steps: int = 3          # agentic 最多搜几轮，硬上限防死循环
    qa_agent_result_limit: int = 6       # 累计证据条数上限，攒满就不再搜

    # 默认 fixture（全离线），不是 mixed —— 演示必须可复现，不能取决于当天外网和
    # 第三方接口的可用性：mixed 下「四源全败」这条路径根本触发不了（新闻/财报永远有数据），
    # 而且同一份输入会给出不同结果。要看真实财报/新闻源时显式打开：
    # RESEARCH_DATA_SOURCE_MODE=mixed，并确保 tavily_api_key、baostock 已配。
    research_data_source_mode: str = "fixture"  # mixed（财报 baostock + 新闻 Tavily）/ fixture（全离线）
    research_fixture_dir: str = "./backend/agents/collect/fixtures"

    # ── 研报检索（P5）──
    # 子图 State（retrieve/state.py）要求精排后 <= 5 条、正文按 report_chunk_chars 截断：
    # 子图内部字段仍会被 checkpoint，量大一样会撑。
    report_collection_name: str = "report_corpus"
    report_recall_top_k: int = 10
    report_rerank_top_k: int = 5
    report_chunk_chars: int = 400
    report_chunk_overlap: int = 60

    # ── 估值建模（P5）──
    # ⚠️ 以下五个是【经验值，未经实测标定】。总设计 §12 把它们列为待标定项。
    # 在被真实标定之前，它们不得在任何地方被当作事实陈述 —— 调参只改这里，不动代码。
    # 这是「不编造数字」这条规则在配置层的表现：经验系数必须自己认领身份。
    valuation_forecast_years: int = 5
    valuation_terminal_growth: float = 0.025
    valuation_discount_rate: float = 0.095
    valuation_comparable_pe_low: float = 20.0   # 可比法 PE 占位，未经可从可比公司市值/净利标定 —— 待标定后替换
    valuation_comparable_pe_high: float = 28.0

    # ── MCP Server 地址（第五章用）──
    # 知识库 MCP 检索研报语料 report_corpus（/mcp/kb）；联网搜索 MCP
    kb_mcp_server_url: str = "http://localhost:8000/mcp/kb"
    web_search_mcp_url: str = "http://localhost:8000/mcp/web-search"
    # 采集 MCP：四源取数（/mcp/data-source）
    data_source_mcp_server_url: str = "http://localhost:8000/mcp/data-source"

    # ── Web 搜索（Tavily 可选；留空则自动用免费的 DuckDuckGo）──
    tavily_api_key: str = ""

    # ── 应用基础配置 ──
    app_env: str = "local"                     # 运行环境标识
    app_debug: bool = False                    # 是否调试模式
    app_host: str = "0.0.0.0"                  # 监听地址
    app_port: int = 8000                       # 监听端口
    log_level: str = "INFO"                    # 日志级别

    class Config:
        """Pydantic 的元配置：告诉 BaseSettings 该怎么读取配置。"""
        env_file = env_local_path          # 从这个文件读取配置
        env_file_encoding = "utf-8"      # 文件编码
        case_sensitive = False           # 大小写不敏感：环境变量 DB_HOST 能对应字段 db_host
        extra = "ignore"                 # .env.local 里多出来的、模型没定义的字段一律忽略（不报错）


@lru_cache()                             # 缓存：保证 get_settings() 只创建一次 Settings、只读一次文件
def get_settings() -> Settings:
    """获取全局唯一的配置对象。任何模块要用配置，都调用这个函数。

    :return: Settings 实例；经 lru_cache 缓存，多次调用返回同一对象。
    """
    return Settings()                    # 首次调用时创建实例；之后每次都返回同一个缓存对象


if __name__ == '__main__':
    settings = get_settings()
    print(settings.database_url)
    print(settings.db_user)
    ...
