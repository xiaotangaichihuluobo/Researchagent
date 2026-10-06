# backend/db/migrations.py
#
# 启动时自动执行的 Schema 补丁（全部幂等，可重复运行）。
# 规则：
#   - 只写 ADD COLUMN IF NOT EXISTS / CREATE INDEX IF NOT EXISTS 等幂等 DDL
#   - 禁止写 DROP / TRUNCATE 等破坏性变更
#   - 每次 init_db.sql 新增字段，同步在 _MIGRATIONS 里追加一条

from sqlalchemy import text
from backend.dependencies import AsyncSessionLocal
from backend.core.logger import get_logger

logger = get_logger(__name__)

# ── 所有需要补丁的 DDL，按时间顺序追加。SQL 必须幂等（IF NOT EXISTS）──
# 初始为空是【正确】的：建表由 scripts/init_db.sql 负责，这里只放「上线后新增的字段/索引」。
# 破坏性变更（DROP / 改 CHECK 约束）一律不写在这里 —— 迁移会在每次启动时自动跑，
# 把破坏性 DDL 放进来等于让服务重启就能删数据。那类变更走 scripts/reset_research_db.sql，人工执行。
_MIGRATIONS: list[tuple[str, str]] = [
    # P5：估值新增「每股价值」两列。纯加法、幂等，正落在本文件允许的范围内。
    # 同一张表的两条 CHECK 改动【不在】这里 —— 那需要 DROP + ADD，
    # 属破坏性变更，走 scripts/manual_alter_p5_valuation.sql（人工执行一次）。
    # 同一张表的两种变更走两条路，正是本文件头注想要的分工：
    # 加列幂等可重跑，改 CHECK 不可逆，而这个函数每次启动都跑。
    (
        "valuation_results.per_share_low/per_share_high",
        "ALTER TABLE valuation_results"
        " ADD COLUMN IF NOT EXISTS per_share_low  NUMERIC(20,4),"
        " ADD COLUMN IF NOT EXISTS per_share_high NUMERIC(20,4)",
    ),
    # P5：valuation_results.task_id 的唯一索引。
    #
    # 为什么必须有：仓储的 upsert 用 ON CONFLICT (task_id) DO UPDATE，而 ON CONFLICT
    # 需要一个唯一约束/唯一索引来推断冲突目标。只有普通索引时 Postgres 直接报
    # 「no unique or exclusion constraint matching the ON CONFLICT specification」——
    # **每一次估值写入都会失败**。research_reports 上有 UNIQUE(task_id)，
    # valuation_results 上从来没有，这个差别照抄过来就踩坑了。
    #
    # 建索引是幂等且非破坏性的，正落在本文件允许的范围内，所以走自动迁移；
    # 那条多余的普通索引（idx_valuation_results_task）由人工脚本清掉 ——
    # DROP 不在这里。
    (
        "valuation_results.uq_valuation_results_task",
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_valuation_results_task"
        " ON valuation_results (task_id)",
    ),
    # P5/13：research_reports 补「历史参照要点」列。纯加法、幂等。
    # 这两个值（has_reference / comparison_points）原本只在检索子图 State 里，
    # 没落业务表 —— 研报详情接口要给结构化字段，就得有这一列可读，
    # 而不是让前端去解析正文文本里的要点。
    (
        "research_reports.comparison_points",
        "ALTER TABLE research_reports"
        " ADD COLUMN IF NOT EXISTS comparison_points JSONB NOT NULL DEFAULT '[]'::jsonb",
    ),
    # 去 LangGraph：前进式续跑用「轻量暂停点」取代 PostgresSaver 的全量快照。
    # 只记「暂停在哪个段、等什么」，不记流程状态的任何内容 —— 暂停态的正文内容
    # 都在业务表（草稿 / 待审 / 预检清单）。纯建表、幂等。
    (
        "pipeline_checkpoints 轻量暂停点",
        "CREATE TABLE IF NOT EXISTS pipeline_checkpoints ("
        " id BIGSERIAL PRIMARY KEY,"
        " tenant_id VARCHAR(36) NOT NULL,"
        " task_id UUID NOT NULL UNIQUE,"
        " stage VARCHAR(32) NOT NULL,"
        " step VARCHAR(32) NOT NULL,"
        " payload JSONB NOT NULL DEFAULT '{}'::jsonb,"
        " created_at TIMESTAMPTZ NOT NULL DEFAULT now(),"
        " updated_at TIMESTAMPTZ NOT NULL DEFAULT now()"
        ")",
    ),
    # 轨道 B：RAG 独立问答（真多轮）建表。三张都是纯建表、幂等。
    # user_id 是普通 UUID、不强外键 —— 仿 pipeline_checkpoints 无 FK 先例，
    # 避免 QA 会话被 users 表的删除/Cleanup 级联误伤。多轮历史以 qa_messages
    # 为唯一真相来源（重启可读 /sessions/{id}/history）；qa_sessions 只存滚动摘要。
    (
        "qa_sessions 问答会话（摘要）",
        "CREATE TABLE IF NOT EXISTS qa_sessions ("
        " id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),"
        " tenant_id VARCHAR(64) NOT NULL DEFAULT 'tenant_default',"
        " user_id UUID,"
        " thread_id VARCHAR(128) NOT NULL UNIQUE,"
        " summary TEXT,"
        " segments JSONB NOT NULL DEFAULT '[]'::jsonb,"
        " summary_version INT NOT NULL DEFAULT 0,"
        " created_at TIMESTAMPTZ NOT NULL DEFAULT now(),"
        " updated_at TIMESTAMPTZ NOT NULL DEFAULT now()"
        ")",
    ),
    (
        "qa_sessions 中程分段列（对已存在表补列，幂等）",
        "ALTER TABLE qa_sessions"
        " ADD COLUMN IF NOT EXISTS segments JSONB NOT NULL DEFAULT '[]'::jsonb",
    ),
    (
        "qa_messages 问答消息流",
        "CREATE TABLE IF NOT EXISTS qa_messages ("
        " id BIGSERIAL PRIMARY KEY,"
        " thread_id VARCHAR(128) NOT NULL,"
        " role VARCHAR(16) NOT NULL,"
        " content TEXT NOT NULL,"
        " seq INT NOT NULL,"
        " created_at TIMESTAMPTZ NOT NULL DEFAULT now()"
        ")",
    ),
    (
        "qa_messages 会话内按序索引",
        "CREATE INDEX IF NOT EXISTS idx_qa_messages_thread_seq"
        " ON qa_messages (thread_id, seq)",
    ),
    (
        "qa_messages 助手消息参考来源(JSON 数组)",
        "ALTER TABLE qa_messages ADD COLUMN IF NOT EXISTS sources TEXT",
    ),
    (
        "knowledge_pending_queue 低置信度待补充问题",
        "CREATE TABLE IF NOT EXISTS knowledge_pending_queue ("
        " id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),"
        " tenant_id VARCHAR(64) NOT NULL DEFAULT 'tenant_default',"
        " question TEXT NOT NULL,"
        " user_id UUID,"
        " confidence DOUBLE PRECISION NOT NULL DEFAULT 0.0,"
        " status VARCHAR(16) NOT NULL DEFAULT 'pending',"
        " created_at TIMESTAMPTZ NOT NULL DEFAULT now()"
        ")",
    ),
    (
        "knowledge_pending_queue 同问题去重索引",
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_pending_tenant_question"
        " ON knowledge_pending_queue (tenant_id, question)",
    ),
    (
        "qa_faq 待补料问题闭环后落成的 FAQ 问答表",
        "CREATE TABLE IF NOT EXISTS qa_faq ("
        " id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),"
        " tenant_id VARCHAR(64) NOT NULL DEFAULT 'tenant_default',"
        " question TEXT NOT NULL,"
        " answer TEXT NOT NULL,"
        " source_count INT NOT NULL DEFAULT 1,"
        " status VARCHAR(16) NOT NULL DEFAULT 'active',"
        " created_at TIMESTAMPTZ NOT NULL DEFAULT now()"
        ")",
    ),
    (
        "qa_faq 同租户同问题去重索引",
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_faq_tenant_question"
        " ON qa_faq (tenant_id, question)",
    ),
    (
        "qa_dead_letter 问答落库/入队失败的载荷（死信,供人工复核）",
        "CREATE TABLE IF NOT EXISTS qa_dead_letter ("
        " id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),"
        " tenant_id VARCHAR(64) NOT NULL DEFAULT 'tenant_default',"
        " channel VARCHAR(32) NOT NULL,"
        " thread_id VARCHAR(128) DEFAULT '',"
        " payload JSONB,"
        " error TEXT,"
        " created_at TIMESTAMPTZ NOT NULL DEFAULT now()"
        ")",
    ),
    # agentic 高质量答案沉淀桶：存「已有现成答案」的回答（question+answer+sources），
    # 由 scripts/sink_agentic_answers.py 离线双落 qa_faq + report_corpus。与 knowledge_pending_queue
    # 的区别：那边只存 question、靠联动脚本 LLM 聚类现生成答案；这里是现成答案直接落，故独立成表。
    (
        "qa_sediment_queue agentic 高质量答案待沉淀桶",
        "CREATE TABLE IF NOT EXISTS qa_sediment_queue ("
        " id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),"
        " tenant_id VARCHAR(64) NOT NULL DEFAULT 'tenant_default',"
        " question TEXT NOT NULL,"
        " answer TEXT NOT NULL,"
        " sources TEXT NOT NULL DEFAULT '',"
        " user_id UUID,"
        " status VARCHAR(16) NOT NULL DEFAULT 'pending',"
        " created_at TIMESTAMPTZ NOT NULL DEFAULT now()"
        ")",
    ),
    (
        "qa_sediment_queue 同问题去重索引",
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_sediment_tenant_question"
        " ON qa_sediment_queue (tenant_id, question)",
    ),
    # 轨道 B：student_id → user_id 列改名。只对「已按旧名建表」的环境生效；
    # 全新库上面 CREATE TABLE 已写成 user_id，此块自动跳过。用 information_schema
    # 判存在再改名，保证幂等可重跑（迁移每次启动都跑，RENAME 需防二次报错）。
    (
        "qa_sessions.student_id → user_id（列改名，幂等）",
        "DO $$ BEGIN"
        " IF EXISTS (SELECT 1 FROM information_schema.columns"
        "            WHERE table_name='qa_sessions' AND column_name='student_id') THEN"
        "   ALTER TABLE qa_sessions RENAME COLUMN student_id TO user_id;"
        " END IF;"
        " END $$;",
    ),
    (
        "knowledge_pending_queue.student_id → user_id（列改名，幂等）",
        "DO $$ BEGIN"
        " IF EXISTS (SELECT 1 FROM information_schema.columns"
        "            WHERE table_name='knowledge_pending_queue' AND column_name='student_id') THEN"
        "   ALTER TABLE knowledge_pending_queue RENAME COLUMN student_id TO user_id;"
        " END IF;"
        " END $$;",
    ),
    # 跨会话用户画像（设计见 docs/跨会话记忆设计.md §3.1）。以 (tenant, user) 为键、
    # 存稳定偏好；结构化字段不改写，写入走显式接口（PUT /api/v1/qa/profile）。
    # 纯建表、幂等，落本文件允许的范围内。
    (
        "user_profiles 跨会话用户画像：风格/interests/watchlist/facts",
        "CREATE TABLE IF NOT EXISTS user_profiles ("
        " id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),"
        " tenant_id VARCHAR(64) NOT NULL DEFAULT 'tenant_default',"
        " user_id VARCHAR(64) NOT NULL,"
        " preferred_style VARCHAR(32),"
        " interests JSONB NOT NULL DEFAULT '[]'::jsonb,"
        " watchlist JSONB NOT NULL DEFAULT '[]'::jsonb,"
        " facts JSONB NOT NULL DEFAULT '[]'::jsonb,"
        " updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),"
        " created_at TIMESTAMPTZ NOT NULL DEFAULT now(),"
        " UNIQUE (tenant_id, user_id)"
        ")",
    ),
    # LLM token usage 观测：每次外部 LLM 调用落一行（按任务/会话聚合成本）。
    # 与 qa_dead_letter 同一条宽松先例：task_id / thread_id 都可空、无外键。
    # 轨道 A（研报任务）带 task_id；轨道 B（问答）带 thread_id；
    # 两条轨道并存于同一张表，靠可空关联键区分（仿 pipeline_checkpoints 无 FK）。
    # 纯粹是观测，落库失败只记日志、绝不反向污染 LLM 调用（见 usage_ctx.py）。
    (
        "llm_usage LLM token 用量观测表",
        "CREATE TABLE IF NOT EXISTS llm_usage ("
        " id BIGSERIAL PRIMARY KEY,"
        " tenant_id VARCHAR(64) NOT NULL DEFAULT 'tenant_default',"
        " task_id UUID,"
        " thread_id VARCHAR(128),"
        " agent_type VARCHAR(32) NOT NULL,"
        " vendor VARCHAR(16) NOT NULL,"
        " model VARCHAR(64) NOT NULL,"
        " prompt_tokens INT,"
        " completion_tokens INT,"
        " total_tokens INT,"
        " latency_ms INT,"
        " streaming BOOLEAN NOT NULL DEFAULT FALSE,"
        " created_at TIMESTAMPTZ NOT NULL DEFAULT now()"
        ")",
    ),
    (
        "llm_usage 按任务聚合索引",
        "CREATE INDEX IF NOT EXISTS idx_llm_usage_task"
        " ON llm_usage (tenant_id, task_id)",
    ),
]


async def run_migrations() -> None:
    """
    在应用启动时执行所有 Schema 补丁。
    单条失败只记录警告，不阻断启动流程。

    :return: 无返回值。
    """
    async with AsyncSessionLocal() as session:
        for desc, sql in _MIGRATIONS:
            try:
                await session.execute(text(sql))
                await session.commit()
                logger.debug("db.migration_applied", column=desc)
            except Exception as e:
                await session.rollback()
                err = str(e)
                if "already exists" not in err:
                    logger.warning("db.migration_failed", column=desc, error=err)

    logger.info("db.migrations_done", count=len(_MIGRATIONS))
