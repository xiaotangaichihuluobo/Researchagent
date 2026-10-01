-- ============================================================
-- ResearchAgent PostgreSQL 数据库初始化脚本（投研域 11 张表）
--
-- 库名：researchagent
-- 两条执行路径共用本文件，保证两边 schema 完全一致、不重复维护：
--   ① Docker 首次启动：docker-compose.yml 把它挂到
--      /docker-entrypoint-initdb.d/01_init_db.sql，容器初始化时自动执行
--   ② 本机手工重建：scripts/reset_research_db.sql 里 \i init_db.sql
--
-- 规矩：全部 IF NOT EXISTS，可重复执行
--
-- ⚠️ 本文件**不包含** LangGraph 的四张 checkpoint 表
-- （checkpoints / checkpoint_blobs / checkpoint_writes / checkpoint_migrations）。
-- 它们由 checkpointer 自己在启动时建：backend/main.py 的 lifespan 里
-- `await ensure_checkpointer()` → PostgresSaver.setup()（幂等）。
-- 所以「全新环境一键初始化」的实际依赖是：本文件 + 启动一次后端。
-- 只跑本文件、不启动后端的话，P6 的风控闸门会因为没有 checkpointer 表而起不来
-- （这是**故意**的：见 memory.py 里「宁可起不来」那段）。

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================
-- 1. 用户与权限：角色换域为 研究员 / 风控 / 管理员
-- ============================================================
CREATE TABLE IF NOT EXISTS users (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id       VARCHAR(64) NOT NULL DEFAULT 'tenant_default',
    username        VARCHAR(64) NOT NULL,
    email           VARCHAR(128) NOT NULL,
    password_hash   VARCHAR(256) NOT NULL,
    role            VARCHAR(16) NOT NULL CHECK (role IN ('researcher', 'risk_control', 'admin')),
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (tenant_id, email)
);
CREATE INDEX IF NOT EXISTS idx_users_tenant_id ON users (tenant_id);
CREATE INDEX IF NOT EXISTS idx_users_role      ON users (role);

-- ============================================================
-- 2. 研究标的
-- ============================================================
CREATE TABLE IF NOT EXISTS companies (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id       VARCHAR(64) NOT NULL DEFAULT 'tenant_default',
    code            VARCHAR(32)  NOT NULL,            -- 标的代码，如 '600519.SH'
    name            VARCHAR(128) NOT NULL,
    industry        VARCHAR(64)  NOT NULL,            -- 所属行业，RAG 横向检索用
    market          VARCHAR(16)  NOT NULL DEFAULT 'A股',
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (tenant_id, code)
);
CREATE INDEX IF NOT EXISTS idx_companies_industry ON companies (industry);

-- ============================================================
-- 3. 研究任务（一次研究 = 一个标的 = 一张这张表 = 一条流程实例）
-- ============================================================
CREATE TABLE IF NOT EXISTS research_tasks (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id       VARCHAR(64) NOT NULL DEFAULT 'tenant_default',
    company_id      UUID NOT NULL REFERENCES companies(id),
    created_by      UUID NOT NULL REFERENCES users(id),
    thread_id       VARCHAR(128) NOT NULL UNIQUE,     -- 关联 LangGraph checkpoint
    status          VARCHAR(24) NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending','running','awaiting_risk_review',
                                      'approved','rejected','published','failed')),
    current_stage   VARCHAR(24) NOT NULL DEFAULT 'collect'
                    CHECK (current_stage IN ('collect','analyze','retrieve',
                                             'valuation','risk_review','publish')),
    redo_count      INT NOT NULL DEFAULT 0,           -- 驳回重做次数，防死循环
    last_error      TEXT,
    started_at      TIMESTAMPTZ,
    finished_at     TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_research_tasks_tenant_created
    ON research_tasks (tenant_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_research_tasks_status ON research_tasks (status);
CREATE INDEX IF NOT EXISTS idx_research_tasks_company ON research_tasks (company_id);

-- ============================================================
-- 4. 采集到的每条数据：来源标注 + 时效降权 的落点
-- ============================================================
CREATE TABLE IF NOT EXISTS research_data_items (
    id                UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id         VARCHAR(64) NOT NULL DEFAULT 'tenant_default',
    task_id           UUID NOT NULL REFERENCES research_tasks(id) ON DELETE CASCADE,
    company_id        UUID NOT NULL REFERENCES companies(id),
    source_type       VARCHAR(24) NOT NULL
                      CHECK (source_type IN ('financial_report','announcement','news','industry')),
    source_name       VARCHAR(128) NOT NULL,          -- 'tavily' / 'fixture:annual_report'
    url               VARCHAR(1024),
    title             VARCHAR(512) NOT NULL,
    content           TEXT NOT NULL,
    raw               JSONB,                          -- 原始返回，便于回溯
    published_at      TIMESTAMPTZ,                    -- 数据发布时间：时效降权的基准
    fetched_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- 采集时刻算好落库，不在查询时重算：
    -- 它记录的是「采集时这份数据有多新」，属审计信息；数据会变旧而权重不变是刻意的。
    timeliness_weight NUMERIC(3,2) NOT NULL
                      CHECK (timeliness_weight >= 0 AND timeliness_weight <= 1),
    reliability       NUMERIC(3,2) NOT NULL
                      CHECK (reliability >= 0 AND reliability <= 1),
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_research_data_items_task
    ON research_data_items (task_id, source_type);
CREATE INDEX IF NOT EXISTS idx_research_data_items_company
    ON research_data_items (company_id, published_at DESC);

-- ============================================================
-- 5. 四维分析结果（fundamental/technical/sentiment/industry）
-- ============================================================
CREATE TABLE IF NOT EXISTS dimension_analyses (
    id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id        VARCHAR(64) NOT NULL DEFAULT 'tenant_default',
    task_id          UUID NOT NULL REFERENCES research_tasks(id) ON DELETE CASCADE,
    dimension        VARCHAR(24) NOT NULL
                     CHECK (dimension IN ('fundamental','technical','sentiment','industry')),
    version          INT NOT NULL DEFAULT 1,          -- 驳回重做时 +1，保留历史不覆盖
    -- score 允许 NULL 是刻意的：业务规则要求「任一维数据缺失则标数据不足而非补零」。
    -- 补零会让综合评级出现虚假低分；NULL + data_sufficiency 让聚合能区分「得 0 分」与「没数据」。
    score            NUMERIC(5,2) CHECK (score IS NULL OR (score >= 0 AND score <= 100)),
    conclusion       TEXT,
    evidence         JSONB NOT NULL DEFAULT '[]'::jsonb,   -- 引用的 research_data_items.id 数组
    data_sufficiency VARCHAR(16) NOT NULL
                     CHECK (data_sufficiency IN ('sufficient','partial','insufficient')),
    is_current       BOOLEAN NOT NULL DEFAULT TRUE,   -- 只有当前版本参与聚合
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (task_id, dimension, version)
);
CREATE INDEX IF NOT EXISTS idx_dimension_analyses_current
    ON dimension_analyses (task_id, is_current);

-- ============================================================
-- 6. 估值结果
-- ============================================================
CREATE TABLE IF NOT EXISTS valuation_results (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id     VARCHAR(64) NOT NULL DEFAULT 'tenant_default',
    task_id       UUID NOT NULL REFERENCES research_tasks(id) ON DELETE CASCADE,
    method        VARCHAR(24) NOT NULL CHECK (method IN ('dcf','comparable','blended','unavailable')),
    value_low     NUMERIC(20,4),
    value_high    NUMERIC(20,4),
    -- P5：每股价值（元）= value_low/high(亿元) / 总股本(万股) × 1e4。
    -- 单独两列而不是塞进 assumptions：详情页要按结构化字段渲染，不解析 JSONB。
    per_share_low  NUMERIC(20,4),
    per_share_high NUMERIC(20,4),
    currency      VARCHAR(8) NOT NULL DEFAULT 'CNY',
    assumptions   JSONB NOT NULL DEFAULT '{}'::jsonb,
    rationale     TEXT,
    is_available  BOOLEAN NOT NULL DEFAULT TRUE,      -- FALSE 时 value_*/per_share_* 必须为 NULL
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- 不可用就必须是 NULL，不允许留一个「默认区间」蒙混过关。
    -- P5 把覆盖范围从两列扩到四列：同样的病对 per_share_* 也成立。
    -- 'unavailable' 进 method 的允许集合，是因为 method 是 NOT NULL，
    -- 降级时若写 'blended' 之类就是谎称做过合成。
    CONSTRAINT chk_valuation_nullable CHECK (
        is_available = TRUE OR (value_low IS NULL AND value_high IS NULL
                                AND per_share_low IS NULL AND per_share_high IS NULL)
    )
);
-- task_id 必须【唯一】：仓储的 upsert 用 ON CONFLICT (task_id) DO UPDATE，
-- 而 ON CONFLICT 需要唯一约束/唯一索引作推断依据 —— 只有普通索引时 Postgres
-- 会直接报 no unique or exclusion constraint matching the ON CONFLICT
-- specification，每一次估值写入都会失败。
-- （research_reports 上一直有 UNIQUE(task_id)，这张表从来没有 —— 别照抄错的那张。）
CREATE UNIQUE INDEX IF NOT EXISTS uq_valuation_results_task ON valuation_results (task_id);

-- ============================================================
-- 7. 风控签字记录（监管留痕核心）
-- ============================================================
CREATE TABLE IF NOT EXISTS risk_reviews (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id     VARCHAR(64) NOT NULL DEFAULT 'tenant_default',
    task_id       UUID NOT NULL REFERENCES research_tasks(id) ON DELETE CASCADE,
    report_id     UUID,                               -- 待审研报（草稿），由应用层写入后回填
    reviewer_id   UUID REFERENCES users(id),          -- 占位行为 NULL，签字时必须落真实签字人
    decision      VARCHAR(16) CHECK (decision IN ('approve','modify','reject')),
    redo_targets  JSONB,                              -- {"stage":"analyze","dimensions":["sentiment"]}
    comments      TEXT,
    checklist     JSONB NOT NULL DEFAULT '{}'::jsonb, -- 自动预检结果快照
    signed_at     TIMESTAMPTZ,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- 「签字必须带意见」写进 DB 约束而非应用层：合规要求谁签字谁负责，
    -- 空意见的签字在合规上无效。约束写在数据层，任何人绕不过去。
    -- 未签字时三列全 NULL；一旦签字，三列必须同时满足且意见不短于 10 字。
    CONSTRAINT chk_signed_complete CHECK (
        (reviewer_id IS NULL     AND decision IS NULL     AND comments IS NULL)
     OR (reviewer_id IS NOT NULL AND decision IS NOT NULL AND comments IS NOT NULL
         AND length(comments) >= 10)
    )
);
CREATE INDEX IF NOT EXISTS idx_risk_reviews_task ON risk_reviews (task_id);

-- ============================================================
-- 8. 研报产出（草稿 / 已发布）
-- ============================================================
CREATE TABLE IF NOT EXISTS research_reports (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id       VARCHAR(64) NOT NULL DEFAULT 'tenant_default',
    task_id         UUID NOT NULL REFERENCES research_tasks(id) ON DELETE CASCADE,
    company_id      UUID NOT NULL REFERENCES companies(id),
    title           VARCHAR(256) NOT NULL,
    content         TEXT NOT NULL,
    rating          VARCHAR(16),
    status          VARCHAR(16) NOT NULL DEFAULT 'draft'
                    CHECK (status IN ('draft','published','rejected')),
    risk_disclosure TEXT,                             -- 风险揭示段落
    comparison_points JSONB NOT NULL DEFAULT '[]'::jsonb,  -- P5：检索到的可比对要点
    published_at    TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- 一次任务只留一份研报，驳回重做走 UPSERT 覆盖草稿，不产生多份
    UNIQUE (task_id)
);
CREATE INDEX IF NOT EXISTS idx_research_reports_status
    ON research_reports (status, published_at DESC);

-- ============================================================
-- 9. 历史研报语料元数据（RAG 用，正文切分在 Milvus）
-- ============================================================
CREATE TABLE IF NOT EXISTS report_corpus (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id     VARCHAR(64) NOT NULL DEFAULT 'tenant_default',
    company_id    UUID REFERENCES companies(id),
    industry      VARCHAR(64),
    title         VARCHAR(256) NOT NULL,
    report_type   VARCHAR(16) NOT NULL CHECK (report_type IN ('internal','broker','company')),
    published_at  TIMESTAMPTZ NOT NULL,               -- 纵向检索「观点变化轨迹」的依据
    milvus_ids    JSONB NOT NULL DEFAULT '[]'::jsonb, -- 该研报切分后在 Milvus 中的主键
    is_active     BOOLEAN NOT NULL DEFAULT TRUE,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_report_corpus_company
    ON report_corpus (company_id, published_at DESC);
CREATE INDEX IF NOT EXISTS idx_report_corpus_industry
    ON report_corpus (industry, published_at DESC);

-- ============================================================
-- 10. 阶段进度事件（供前端轮询）
-- ============================================================
CREATE TABLE IF NOT EXISTS task_stage_events (
    id           UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id    VARCHAR(64) NOT NULL DEFAULT 'tenant_default',
    task_id      UUID NOT NULL REFERENCES research_tasks(id) ON DELETE CASCADE,
    stage        VARCHAR(24) NOT NULL,
    status       VARCHAR(16) NOT NULL CHECK (status IN ('started','success','failed','skipped')),
    detail       JSONB NOT NULL DEFAULT '{}'::jsonb,
    occurred_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_task_stage_events_task
    ON task_stage_events (task_id, occurred_at);

-- ============================================================
-- 11. 审计流水
-- ============================================================
CREATE TABLE IF NOT EXISTS audit_logs (
    id           BIGSERIAL PRIMARY KEY,
    tenant_id    VARCHAR(64) NOT NULL DEFAULT 'tenant_default',
    actor_id     UUID REFERENCES users(id),
    action       VARCHAR(64) NOT NULL,
    target_type  VARCHAR(32) NOT NULL,
    target_id    UUID,
    detail       JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_audit_logs_target ON audit_logs (target_type, target_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_actor  ON audit_logs (actor_id, created_at DESC);

-- ============================================================
-- 自动更新 updated_at 触发器
-- ============================================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 本文件要有「可重复执行」的承诺，就必须先 DROP 再建：
-- PG 15 的 CREATE TRIGGER 没有 IF NOT EXISTS，直接 CREATE 第二次执行会报
-- "trigger ... already exists"，与文件头的承诺相矛盾。
DO $$
DECLARE
    t TEXT;
BEGIN
    FOREACH t IN ARRAY ARRAY['users', 'companies', 'research_tasks', 'research_reports']
    LOOP
        EXECUTE format('DROP TRIGGER IF EXISTS trg_%s_updated_at ON %s', t, t);
        EXECUTE format('
            CREATE TRIGGER trg_%s_updated_at
            BEFORE UPDATE ON %s
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
        ', t, t);
    END LOOP;
END;
$$;
