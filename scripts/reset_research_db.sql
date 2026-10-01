-- ============================================================
-- 一次性破坏性重建：只重建【投研库 researchagent】
--
-- ⚠️ 本脚本【绝不】触碰非投研库 —— 其余库原封不动地留着。
--    全程只对 researchagent 做 DROP DATABASE + CREATE DATABASE。
-- ⚠️ DROP DATABASE 要求该库没有任何活动连接。执行前先停掉后端服务，
--    否则会报 "database ... is being accessed by other users"。
-- ⚠️ 本文件【不】进 backend/db/migrations.py —— 迁移机制只跑幂等 DDL，
--
-- ⚠️ 重建完先跑 `python scripts/init_milvus.py` 重建 report_corpus 集合并灌样例，
--     它会顺带把 backend/agents/retrieve/fixtures/published_reports.json（手写的
--     12 家公司×11 行业种子语料）一起回灌（同目录所有 fixture 自动 glob）。
--     破坏性变更必须由人显式执行一次，绝不能在应用启动时自动跑。
--
-- 执行方式（psql 不在本机 PATH 上，走容器内）：
--   docker cp scripts/init_db.sql           research_agent_postgres:/tmp/init_db.sql
--   docker cp scripts/reset_research_db.sql research_agent_postgres:/tmp/reset_research_db.sql
--   docker exec -e PGPASSWORD=<密码> research_agent_postgres \
--     psql -U researchagent_user -d postgres -v ON_ERROR_STOP=1 -f /tmp/reset_research_db.sql
--
-- 若本机装了 psql，也可直连执行（\ir 会按脚本所在目录找 init_db.sql）：
--   psql -h localhost -p 5433 -U researchagent_user -d postgres -f scripts/reset_research_db.sql
-- ============================================================

-- DROP / CREATE DATABASE 不能放在事务块里（Postgres 不允许），
-- 所以本文件没有 BEGIN/COMMIT，靠 psql 的自动提交逐条执行。
DROP DATABASE IF EXISTS researchagent;

-- 不写 OWNER：owner 默认是执行本脚本的连接角色（就是 DB_USER），
-- 免得把用户名硬编码进 SQL、换台机器就找不到这个角色。
CREATE DATABASE researchagent;

-- 切到新库里，再用初始化脚本建那 11 张表。
-- 用 \ir 而不是 \i：\i 相对 psql 进程的工作目录解析，\ir 相对【本脚本所在目录】解析。
-- 本脚本要和 init_db.sql 放在同一目录下，两者才算一套。
\connect researchagent
\ir init_db.sql

-- 校验：应恰好 11 张业务表
-- SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY 1;
