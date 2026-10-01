-- ============================================================
-- P5 人工执行一次的破坏性 DDL：valuation_results 的两条 CHECK 约束。
--
-- ⚠️ 本文件【不】进 backend/db/migrations.py —— 那个函数每次启动都跑，
--    而这里含 DROP CONSTRAINT。把破坏性 DDL 放进去等于让服务重启就能改约束。
--
-- 执行方式见 docs/superpowers/plans/2026-09-18-investment-research-p5-p7.md 开头的
-- 「前置人工步骤」。执行前该表为 0 行且无任何读写方，无数据风险。
--
-- 幂等性：DROP CONSTRAINT IF EXISTS + ADD CONSTRAINT，可重复执行。
-- ============================================================

-- ① 不可用 ⇒ 四个数值列必须全为 NULL（原来只覆盖 value_low/high 两列）
ALTER TABLE valuation_results DROP CONSTRAINT IF EXISTS chk_valuation_nullable;
ALTER TABLE valuation_results ADD CONSTRAINT chk_valuation_nullable CHECK (
    is_available = TRUE OR (value_low IS NULL AND value_high IS NULL
                            AND per_share_low IS NULL AND per_share_high IS NULL)
);

-- ② method 允许 'unavailable'：降级时 method 是 NOT NULL，没有合法值可写
ALTER TABLE valuation_results DROP CONSTRAINT IF EXISTS valuation_results_method_check;
ALTER TABLE valuation_results ADD CONSTRAINT valuation_results_method_check
    CHECK (method IN ('dcf','comparable','blended','unavailable'));

-- ③ 清掉被唯一索引取代的那条普通索引。
--    唯一索引本身走 migrations.py（幂等、非破坏性）；退役旧索引要 DROP，
--    所以放这里人工执行 —— DROP 不进自动迁移是 migrations.py 头注的硬规则。
--    先建后删的顺序是刻意的：唯一索引建失败时旧索引还在，查询不受影响。
DROP INDEX IF EXISTS idx_valuation_results_task;
