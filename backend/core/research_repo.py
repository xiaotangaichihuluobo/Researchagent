# backend/core/research_repo.py
# 投研域仓储层：所有对业务表的读写都收口在这里。
#
# 为什么要有这一层：业务表是唯一可信来源（设计文档 §4.3），
# 如果每个 Agent 节点各写各的 SQL，改一张表的字段就要满仓库找。
# 收口之后，节点只调函数、不写 SQL；表的形状变化只影响这一个文件。

import json
import uuid
from datetime import datetime, timezone

from backend.core.timeutil import cn_now
from typing import Any, Optional

from sqlalchemy import text

from backend.dependencies import AsyncSessionLocal
from backend.core.logger import get_logger

logger = get_logger(__name__)


def _row_to_dict(row) -> dict:
    """把 SQLAlchemy Row 转成普通 dict（JSON 便于后续序列化）。

    :param row: SQLAlchemy 查询返回的单行 Row 对象
    :return: 列名到列值的 dict
    """
    return dict(row._mapping)


def _rows_to_dicts(rows) -> list[dict]:
    """把多行结果批量转成 dict 列表。

    :param rows: SQLAlchemy 查询返回的多行结果集
    :return: dict 列表，每项对应一行
    """
    return [_row_to_dict(r) for r in rows]


# ── 研究标的 ────────────────────────────────────────────────────

async def create_company(tenant_id: str, code: str, name: str,
                         industry: str, market: str = "A股") -> str:
    """新增研究标的，返回其 id。同一租户下 code 已存在则返回已有 id。

    :param tenant_id: 租户隔离键
    :param code: 证券代码，如 600519.SH
    :param name: 公司名称
    :param industry: 所属行业（RAG 横向检索维度）
    :param market: 市场板块，默认 A股
    :return: 标的记录的 id
    """
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("""
                INSERT INTO companies (tenant_id, code, name, industry, market)
                VALUES (:tenant_id, :code, :name, :industry, :market)
                ON CONFLICT (tenant_id, code) DO UPDATE SET name = EXCLUDED.name
                RETURNING id
            """),
            {"tenant_id": tenant_id, "code": code, "name": name,
             "industry": industry, "market": market},
        )
        company_id = str(result.scalar_one())
        await session.commit()
    logger.info("repo.company_created", code=code, company_id=company_id)
    return company_id


async def get_company_by_code(tenant_id: str, code: str) -> Optional[dict]:
    """按标的代码查标的；查不到返回 None（调用方负责判定 → 阶段失败）。

    :param tenant_id: 租户隔离键
    :param code: 证券代码，如 600519.SH
    :return: 标的行 dict；查不到返回 None
    """
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("SELECT * FROM companies WHERE tenant_id = :t AND code = :code LIMIT 1"),
            {"t": tenant_id, "code": code},
        )
        row = result.fetchone()
    return _row_to_dict(row) if row else None


async def get_company_by_id(tenant_id: str, company_id: str) -> Optional[dict]:
    """按标的 id 查标的。运行续跑在重建 state 时需要从 id 反查 code/industry。

    :param tenant_id: 租户隔离键
    :param company_id: 标的记录 id
    :return: 标的行 dict；查不到返回 None
    """
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("SELECT * FROM companies WHERE tenant_id = :t AND id = :id LIMIT 1"),
            {"t": tenant_id, "id": company_id},
        )
        row = result.fetchone()
    return _row_to_dict(row) if row else None


async def list_companies(tenant_id: str, limit: int = 50, offset: int = 0) -> list[dict]:
    """列出租户下启用的研究标的，按 code 升序分页。

    :param tenant_id: 租户隔离键
    :param limit: 返回条数上限，默认 50
    :param offset: 分页偏移量，默认 0
    :return: 标的行 dict 列表
    """
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("SELECT * FROM companies WHERE tenant_id = :t AND is_active = TRUE"
                 " ORDER BY code LIMIT :limit OFFSET :offset"),
            {"t": tenant_id, "limit": limit, "offset": offset},
        )
        return _rows_to_dicts(result.fetchall())


# ── 研究任务 ────────────────────────────────────────────────────

async def create_task(tenant_id: str, company_id: str,
                      created_by: str, thread_id: str) -> str:
    """创建一次研究任务（流程实例），返回其 id。

    :param tenant_id: 租户隔离键
    :param company_id: 研究标的 id
    :param created_by: 发起用户 id
    :param thread_id: 关联历史 checkpoint 的线程标识（唯一）
    :return: 任务记录 id
    """
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("""
                INSERT INTO research_tasks (tenant_id, company_id, created_by, thread_id)
                VALUES (:tenant_id, :company_id, :created_by, :thread_id)
                RETURNING id
            """),
            {"tenant_id": tenant_id, "company_id": company_id,
             "created_by": created_by, "thread_id": thread_id},
        )
        task_id = str(result.scalar_one())
        await session.commit()
    logger.info("repo.task_created", task_id=task_id, company_id=company_id)
    return task_id


async def get_task(tenant_id: str, task_id: str) -> Optional[dict]:
    """按 id 取单个研究任务。

    :param tenant_id: 租户隔离键
    :param task_id: 任务 id
    :return: 任务行 dict；查不到返回 None
    """
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("SELECT * FROM research_tasks WHERE tenant_id = :t AND id = :id"),
            {"t": tenant_id, "id": task_id},
        )
        row = result.fetchone()
    return _row_to_dict(row) if row else None


async def list_tasks(tenant_id: str, created_by: Optional[str] = None,
                     status: Optional[str] = None,
                     limit: int = 20, offset: int = 0) -> list[dict]:
    """列出任务。created_by=None 时列出该租户全部（风控/管理员视角）。

    status 过滤是 P6 加的：风控待审列表只需要 awaiting_risk_review 那几条，
    在应用层过滤会把整租户的任务都拉回来再筛 —— 任务只会越积越多，
    而待审列表是审核人每天要打开的第一个页面。

    :param tenant_id: 租户隔离键
    :param created_by: 发起用户 id；传 None 时列出该租户全部任务（风控/管理员视角）
    :param status: 任务状态过滤，如 awaiting_risk_review；传 None 表示不过滤
    :param limit: 返回条数上限，默认 20
    :param offset: 分页偏移量，默认 0
    :return: 任务行 dict 列表（含联表出来的 company_code / company_name），按创建时间倒序
    """
    sql = ("SELECT t.*, c.code AS company_code, c.name AS company_name"
           " FROM research_tasks t JOIN companies c ON c.id = t.company_id"
           " WHERE t.tenant_id = :t")
    params: dict[str, Any] = {"t": tenant_id, "limit": limit, "offset": offset}
    if created_by:
        sql += " AND t.created_by = :created_by"
        params["created_by"] = created_by
    if status:
        sql += " AND t.status = :status"
        params["status"] = status
    sql += " ORDER BY t.created_at DESC LIMIT :limit OFFSET :offset"

    async with AsyncSessionLocal() as session:
        result = await session.execute(text(sql), params)
        return _rows_to_dicts(result.fetchall())


async def update_task_status(tenant_id: str, task_id: str,
                             status: Optional[str] = None,
                             current_stage: Optional[str] = None,
                             last_error: Optional[str] = None,
                             mark_started: bool = False,
                             mark_finished: bool = False) -> None:
    """局部更新任务状态。

    用 COALESCE 做局部更新：只传 status 时不会把 current_stage 冲成 NULL。
    分两段执行（一次 UPDATE），避免读-改-写的竞态。

    :param tenant_id: 租户隔离键
    :param task_id: 任务 id
    :param status: 新状态；传 None 则保持原值（不冲掉）
    :param current_stage: 新阶段；传 None 则保持原值
    :param last_error: 失败原因；传 None 则保持原值
    :param mark_started: 为 True 时把 started_at 置为首次开始时间（已有时不变）
    :param mark_finished: 为 True 时把 finished_at 置为当前时间
    :return: 无返回值
    """
    async with AsyncSessionLocal() as session:
        await session.execute(
            text("""
                UPDATE research_tasks
                SET status        = COALESCE(:status, status),
                    current_stage = COALESCE(:current_stage, current_stage),
                    last_error    = COALESCE(:last_error, last_error),
                    started_at    = CASE WHEN :mark_started  THEN COALESCE(started_at, NOW())
                                         ELSE started_at END,
                    finished_at   = CASE WHEN :mark_finished THEN NOW()
                                         ELSE finished_at END
                WHERE tenant_id = :t AND id = :id
            """),
            {"t": tenant_id, "id": task_id, "status": status,
             "current_stage": current_stage, "last_error": last_error,
             "mark_started": mark_started, "mark_finished": mark_finished},
        )
        await session.commit()


async def mark_redo_started(tenant_id: str, task_id: str) -> int:
    """驳回重做：把任务转出待审状态，并把重做次数 +1。返回新的计数。

    计数落 DB 而不是父 State：子图写不了父 State 的字段；
    落 DB 后由父图回填，计数同时获得持久性（重启不丢）与可审计性。

    **两个字段必须在同一条 UPDATE 里。** 拆成两次写（先 +1、再改状态）会留一个间隙：
    那段时间里任务仍算「待风控审核」，于是它还挂在风控的待审列表上，
    但它的审核记录已经签过字了 —— 审核人点进去只能拿到 409。
    窗口是毫秒级，正常操作撞不上，但「列表里有、点进去说不在了」正是最难解释的一类故障。

    :param tenant_id: 租户隔离键
    :param task_id: 任务 id
    :return: 新的重做次数（此 UPDATE 后 redo_count+1 的返回值）；任务不存在返回 0
    """
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("UPDATE research_tasks"
                 " SET redo_count = redo_count + 1, status = 'running'"
                 " WHERE tenant_id = :t AND id = :id RETURNING redo_count"),
            {"t": tenant_id, "id": task_id},
        )
        row = result.fetchone()
        await session.commit()
    return int(row[0]) if row else 0


# ── 阶段事件（供前端轮询）──────────────────────────────────────

async def record_stage_event(tenant_id: str, task_id: str, stage: str,
                             status: str, detail: Optional[dict] = None) -> None:
    """记一条阶段事件。status ∈ started / success / failed / skipped。

    这是跨阶段的审计入口：errors 在 State 里只反映当前阶段、会被下一阶段覆盖，
    而阶段事件是只追加的，可以回答「这一轮到底哪一步出了问题」。

    :param tenant_id: 租户隔离键
    :param task_id: 任务 id
    :param stage: 阶段名，如 collect/analyze/retrieve/valuation/risk/publish/pipeline
    :param status: 阶段结果，∈ started/success/failed/skipped
    :param detail: 事件载荷（JSON 可序列化），默认空 dict
    :return: 无返回值
    """
    async with AsyncSessionLocal() as session:
        await session.execute(
            text("INSERT INTO task_stage_events (tenant_id, task_id, stage, status, detail)"
                 " VALUES (:t, :task, :stage, :status, CAST(:detail AS JSONB))"),
            {"t": tenant_id, "task": task_id, "stage": stage, "status": status,
             "detail": json.dumps(detail or {}, ensure_ascii=False)},
        )
        await session.commit()


async def list_stage_events(tenant_id: str, task_id: str) -> list[dict]:
    """按时间顺序列出某任务的全部阶段事件。

    :param tenant_id: 租户隔离键
    :param task_id: 任务 id
    :return: 阶段事件行 dict 列表，按 occurred_at、id 升序
    """
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("SELECT * FROM task_stage_events WHERE tenant_id = :t AND task_id = :task"
                 " ORDER BY occurred_at, id"),
            {"t": tenant_id, "task": task_id},
        )
        return _rows_to_dicts(result.fetchall())


# ── 采集数据 ────────────────────────────────────────────────────

async def insert_data_items(tenant_id: str, task_id: str, company_id: str,
                            items: list[dict]) -> list[str]:
    """批量写入采集到的数据，返回按入参顺序排列的 id 列表。

    权重（timeliness_weight / reliability）由调用方在采集时算好传进来：
    它记录的是「采集时刻这份数据有多新」，是审计快照，不在查询时重算。

    :param tenant_id: 租户隔离键
    :param task_id: 任务 id
    :param company_id: 研究标的 id
    :param items: 采集数据字典列表；每项必须含 source_type/source_name/title/content/timeliness_weight/reliability，可选 url/raw/published_at
    :return: 按入参顺序排列的新写入行 id 列表；items 为空返回空列表
    """
    if not items:
        return []

    ids: list[str] = []
    async with AsyncSessionLocal() as session:
        for item in items:
            result = await session.execute(
                text("""
                    INSERT INTO research_data_items
                        (tenant_id, task_id, company_id, source_type, source_name, url,
                         title, content, raw, published_at, timeliness_weight, reliability)
                    VALUES
                        (:tenant_id, :task_id, :company_id, :source_type, :source_name, :url,
                         :title, :content, CAST(:raw AS JSONB), :published_at,
                         :timeliness_weight, :reliability)
                    RETURNING id
                """),
                {
                    "tenant_id": tenant_id, "task_id": task_id, "company_id": company_id,
                    "source_type": item["source_type"], "source_name": item["source_name"],
                    "url": item.get("url"), "title": item["title"], "content": item["content"],
                    "raw": json.dumps(item.get("raw") or {}, ensure_ascii=False),
                    "published_at": item.get("published_at"),
                    "timeliness_weight": item["timeliness_weight"],
                    "reliability": item["reliability"],
                },
            )
            ids.append(str(result.scalar_one()))
        await session.commit()

    logger.info("repo.data_items_inserted", task_id=task_id, count=len(ids))
    return ids


async def fetch_data_items(tenant_id: str, task_id: str,
                           source_type: Optional[str] = None) -> list[dict]:
    """取本次任务采集到的数据。按「时效权重 × 可信度」倒序 —— 越新越可信的排前面。

    :param tenant_id: 租户隔离键
    :param task_id: 任务 id
    :param source_type: 按来源类型过滤（financial_report/announcement/news/industry）；传 None 查全部
    :return: 数据行 dict 列表；其中 timeliness_weight / reliability 已转成 float 供浮点运算
    """
    sql = ("SELECT * FROM research_data_items WHERE tenant_id = :t AND task_id = :task")
    params: dict[str, Any] = {"t": tenant_id, "task": task_id}
    if source_type:
        sql += " AND source_type = :source_type"
        params["source_type"] = source_type
    sql += " ORDER BY (timeliness_weight * reliability) DESC, published_at DESC NULLS LAST"

    async with AsyncSessionLocal() as session:
        result = await session.execute(text(sql), params)
        rows = _rows_to_dicts(result.fetchall())

    # NUMERIC 列经 asyncpg 回来是 Decimal，而这两个权重是给调用方做浮点运算用的；
    # 不转的话调用方拿 float 与 Decimal 直接运算会抛 TypeError。
    # 与 fetch_current_dimension_results 的 float(r["score"]) 同一处理方式。
    for r in rows:
        r["timeliness_weight"] = float(r["timeliness_weight"])
        r["reliability"] = float(r["reliability"])
    return rows


# ── 维度分析结果 ────────────────────────────────────────────────

async def upsert_dimension_analysis(tenant_id: str, task_id: str, dimension: str,
                                    score: Optional[float], conclusion: str,
                                    evidence: list[str], data_sufficiency: str) -> dict:
    """写入一版维度分析结果，返回新记录。

    语义是「追加新版本」而不是「覆盖」：version = 该维度当前最大版本 + 1，
    旧记录的 is_current 置 FALSE。历史版本保留，研报才能回答
    「上一版为什么改了」—— 监管留痕的要求。

    score 允许 None：数据不足时不补零，由 data_sufficiency 说明原因。

    :param tenant_id: 租户隔离键
    :param task_id: 任务 id
    :param dimension: 维度名，∈ fundamental/technical/sentiment/industry
    :param score: 维度分 0-100；数据不足传 None
    :param conclusion: 维度结论文本
    :param evidence: 引用的 research_data_items.id 数组
    :param data_sufficiency: 数据充分度，∈ sufficient/partial/insufficient
    :return: 新写入版本的行 dict
    """
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("SELECT COALESCE(MAX(version), 0) FROM dimension_analyses"
                 " WHERE task_id = :task AND dimension = :dim"),
            {"task": task_id, "dim": dimension},
        )
        next_version = int(result.scalar_one()) + 1

        # 先把旧的降级，再插新的：两条语句同一事务，中间态不会被外部看到
        await session.execute(
            text("UPDATE dimension_analyses SET is_current = FALSE"
                 " WHERE task_id = :task AND dimension = :dim"),
            {"task": task_id, "dim": dimension},
        )
        result = await session.execute(
            text("""
                INSERT INTO dimension_analyses
                    (tenant_id, task_id, dimension, version, score, conclusion,
                     evidence, data_sufficiency, is_current)
                VALUES
                    (:tenant_id, :task_id, :dimension, :version, :score, :conclusion,
                     CAST(:evidence AS JSONB), :data_sufficiency, TRUE)
                RETURNING *
            """),
            {"tenant_id": tenant_id, "task_id": task_id, "dimension": dimension,
             "version": next_version, "score": score, "conclusion": conclusion,
             "evidence": json.dumps(evidence, ensure_ascii=False),
             "data_sufficiency": data_sufficiency},
        )
        row = _row_to_dict(result.fetchone())
        await session.commit()

    logger.info("repo.dimension_upserted", task_id=task_id, dimension=dimension,
                version=next_version, sufficiency=data_sufficiency)
    return row


async def fetch_current_dimension_results(tenant_id: str, task_id: str) -> dict:
    """取当前有效的四维结果，形如 {dimension: {"score":..., "data_sufficiency":..., ...}}。

    只取 is_current = TRUE 的记录 —— 历史版本留给审计，不参与聚合。

    :param tenant_id: 租户隔离键
    :param task_id: 任务 id
    :return: 形如 {dimension: {"score":...|None, "conclusion":..., "evidence":..., "data_sufficiency":..., "version":...}} 的 dict
    """
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("SELECT * FROM dimension_analyses"
                 " WHERE tenant_id = :t AND task_id = :task AND is_current = TRUE"),
            {"t": tenant_id, "task": task_id},
        )
        rows = _rows_to_dicts(result.fetchall())

    out: dict[str, dict] = {}
    for r in rows:
        out[r["dimension"]] = {
            "score": float(r["score"]) if r["score"] is not None else None,
            "conclusion": r["conclusion"],
            "evidence": r["evidence"],
            "data_sufficiency": r["data_sufficiency"],
            "version": r["version"],
        }
    return out


# ── 研报 ────────────────────────────────────────────────────────

async def upsert_report(tenant_id: str, task_id: str, company_id: str, title: str,
                        content: str, rating: Optional[str], status: str,
                        risk_disclosure: Optional[str] = None,
                        comparison_points: Optional[list] = None) -> str:
    """写入/覆盖本次任务的研报，返回其 id。

    一次任务只留一份研报（表上有 UNIQUE(task_id)）：驳回重做时覆盖草稿，
    不产生多份。published 时补上 published_at。

    comparison_points 是检索子图扒出的可比对要点，随草稿一起落库 ——
    研报详情接口要它的结构化字段，不靠前端解析正文。

    :param tenant_id: 租户隔离键
    :param task_id: 任务 id
    :param company_id: 研究标的 id
    :param title: 研报标题
    :param content: 研报正文
    :param rating: 评级，可空
    :param status: 状态，∈ draft/published/rejected（published 时自动补 published_at）
    :param risk_disclosure: 风险揭示段落，可空
    :param comparison_points: 可比对要点数组，可空，默认参与 UPSERT
    :return: 研报记录 id
    """
    async with AsyncSessionLocal() as session:
        # :status 在同一语句里出现两次（赋值 + 与 'published' 比较），SQLAlchemy 把两处
        # 渲染成同一个占位符，而这两处推不出同一个类型，asyncpg 会报
        # AmbiguousParameterError —— 显式 CAST 消歧，与同文件 JSONB 参数同一手法。
        result = await session.execute(
            text("""
                INSERT INTO research_reports
                    (tenant_id, task_id, company_id, title, content, rating,
                     status, risk_disclosure, comparison_points, published_at)
                VALUES
                    (:tenant_id, :task_id, :company_id, :title, :content, :rating,
                     :status, :risk_disclosure, CAST(:comparison_points AS JSONB),
                     CASE WHEN CAST(:status AS VARCHAR(16)) = 'published'
                          THEN NOW() ELSE NULL END)
                ON CONFLICT (task_id) DO UPDATE SET
                    title             = EXCLUDED.title,
                    content           = EXCLUDED.content,
                    rating            = EXCLUDED.rating,
                    status            = EXCLUDED.status,
                    risk_disclosure   = EXCLUDED.risk_disclosure,
                    comparison_points = EXCLUDED.comparison_points,
                    published_at      = CASE WHEN EXCLUDED.status = 'published'
                                             THEN NOW() ELSE research_reports.published_at END
                RETURNING id
            """),
            {"tenant_id": tenant_id, "task_id": task_id, "company_id": company_id,
             "title": title, "content": content, "rating": rating,
             "status": status, "risk_disclosure": risk_disclosure,
             "comparison_points": json.dumps(comparison_points or [], ensure_ascii=False)},
        )
        report_id = str(result.scalar_one())
        await session.commit()

    logger.info("repo.report_upserted", task_id=task_id, status=status)
    return report_id


async def get_report(tenant_id: str, task_id: str) -> Optional[dict]:
    """取某任务当前唯一的研报记录。

    :param tenant_id: 租户隔离键
    :param task_id: 任务 id
    :return: 研报行 dict；查不到返回 None
    """
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("SELECT * FROM research_reports WHERE tenant_id = :t AND task_id = :task"),
            {"t": tenant_id, "task": task_id},
        )
        row = result.fetchone()
    return _row_to_dict(row) if row else None


# ── 估值结果（P5）──────────────────────────────────────────────

async def upsert_valuation_result(
    tenant_id: str, task_id: str, method: str,
    value_low: Optional[float], value_high: Optional[float],
    per_share_low: Optional[float], per_share_high: Optional[float],
    currency: str, assumptions: list, rationale: Optional[str],
    is_available: bool,
) -> str:
    """写入/覆盖本次任务的估值结果，返回其 id。

    一次任务只留一份：驳回重做覆盖上一次，不保留轨迹 —— valuation_results 没有
    version / is_current 列（dimension_analyses 有），这是本轮如实记下的缺口，
    不是遗漏。

    ⚠️ 下面那句 ON CONFLICT (task_id) 依赖的 UNIQUE 索引是本轮【新加的】
    （uq_valuation_results_task，见 Task 1）：这张表此前只有普通索引，
    research_reports 上的 UNIQUE(task_id) 不能照搬过来。索引不在时这条 SQL
    不是偶尔失败，而是必然报
    「no unique or exclusion constraint matching the ON CONFLICT specification」。

    value_low/high 的语义是【股权价值（亿元）】，per_share_* 是【每股价值（元）】。
    不可用时四个数值列必须全为 NULL —— 不给 0、不给默认区间，
    chk_valuation_nullable 会在库层再拦一道。

    :param tenant_id: 租户隔离键
    :param task_id: 任务 id
    :param method: 估值方法，∈ dcf/comparable/blended/unavailable
    :param value_low: 股权价值下限（亿元），不可用时为 None
    :param value_high: 股权价值上限（亿元），不可用时为 None
    :param per_share_low: 每股价值下限（元），不可用时为 None
    :param per_share_high: 每股价值上限（元），不可用时为 None
    :param currency: 货币，默认 CNY
    :param assumptions: 假设清单数组（derived:/config: 溯源）
    :param rationale: 选法理由（LLM 写），可空
    :param is_available: 是否可交付；为 False 时四个数值列必须全为 None
    :return: 估值结果记录 id
    """
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("""
                INSERT INTO valuation_results
                    (tenant_id, task_id, method, value_low, value_high,
                     per_share_low, per_share_high, currency, assumptions,
                     rationale, is_available)
                VALUES
                    (:tenant_id, :task_id, :method, :value_low, :value_high,
                     :per_share_low, :per_share_high, :currency,
                     CAST(:assumptions AS JSONB), :rationale, :is_available)
                ON CONFLICT (task_id) DO UPDATE SET
                    method         = EXCLUDED.method,
                    value_low      = EXCLUDED.value_low,
                    value_high     = EXCLUDED.value_high,
                    per_share_low  = EXCLUDED.per_share_low,
                    per_share_high = EXCLUDED.per_share_high,
                    currency       = EXCLUDED.currency,
                    assumptions    = EXCLUDED.assumptions,
                    rationale      = EXCLUDED.rationale,
                    is_available   = EXCLUDED.is_available
                RETURNING id
            """),
            {"tenant_id": tenant_id, "task_id": task_id, "method": method,
             "value_low": value_low, "value_high": value_high,
             "per_share_low": per_share_low, "per_share_high": per_share_high,
             "currency": currency,
             "assumptions": json.dumps(assumptions or [], ensure_ascii=False),
             "rationale": rationale, "is_available": is_available},
        )
        row_id = str(result.scalar_one())
        await session.commit()

    logger.info("repo.valuation_upserted", task_id=task_id, method=method,
                is_available=is_available)
    return row_id


async def fetch_valuation_result(tenant_id: str, task_id: str) -> Optional[dict]:
    """取本次任务的估值结果。

    NUMERIC 列经 asyncpg 回来是 Decimal，这里统一转 float —— 与 fetch_data_items
    同一处理：调用方拿 Decimal 与 float 直接运算会抛 TypeError。

    :param tenant_id: 租户隔离键
    :param task_id: 任务 id
    :return: 估值结果行 dict（四个数值列已转 float）；无记录返回 None
    """
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("SELECT * FROM valuation_results WHERE tenant_id = :t AND task_id = :task"),
            {"t": tenant_id, "task": task_id},
        )
        row = result.fetchone()
    if row is None:
        return None

    data = _row_to_dict(row)
    for key in ("value_low", "value_high", "per_share_low", "per_share_high"):
        data[key] = None if data[key] is None else float(data[key])
    return data


# ── 研报语料（P5 回灌）─────────────────────────────────────────

async def upsert_report_corpus(tenant_id: str, report_id: Optional[str],
                               company_id: Optional[str], industry: Optional[str],
                               title: str, report_type: str,
                               published_at: Optional[datetime],
                               milvus_ids: list[str]) -> str:
    """登记一份已进入语料库的研报，返回其 id。

    report_id 传 None 表示这是**样例冷启动语料**（不是本系统产出的研报）；
    company_id 传 None 表示这是**可比公司**的样例（report_corpus.company_id 本就可空，
    横向检索只按 industry 过滤并排除自己，不依赖 companies 表）。

    milvus_ids 这一列此前无任何写入方 —— 它是「PG 里的登记行」与
    「Milvus 里的向量」之间的唯一联系。

    幂等：同一份研报(tenant_id, company_id, title 相同)重复登记时，不算插不进——
    既然调用它是为了「回灌一份已存在的研报」，就复用已有行并刷新 milvus_ids，
    让 init 脚本/回灌可重跑而不把登记行越堆越多。company_id 用 IS NOT DISTINCT
    FROM 判空：可比公司样例的 company_id 是 NULL，NULL = NULL 也得匹配。
    published_at 传 None 时沿用既有值，不再覆盖成 NOW()（否则两次回灌会因时戳
    不同而反复走 INSERT，幂等破功）。

    report_id 【不进 SQL】：report_corpus 表上没有这一列（DDL 实测：
    id/tenant_id/company_id/industry/title/report_type/published_at/milvus_ids/
    is_active/created_at）。它只写进日志，好让「这行语料是哪份研报灌进来的」
    在事后可查。

    :param tenant_id: 租户隔离键
    :param report_id: 本系统产出的研报 id；传 None 表示样例冷启动语料（不进 SQL，仅写日志）
    :param company_id: 标的 id；传 None 表示可比公司样例语料
    :param industry: 所属行业（横向过滤），可空
    :param title: 研报标题（幂等判重的键之一）
    :param report_type: 来源类型，∈ internal/broker/company
    :param published_at: 发布时点；传 None 时沿用既有值或取 NOW()
    :param milvus_ids: 该研报在 Milvus 中的切片主键数组，与 PG 登记行对齐
    :return: 语料登记行 id（重复登记时复用已有行）
    """
    async with AsyncSessionLocal() as session:
        existing = await session.execute(
            text("SELECT id FROM report_corpus"
                 " WHERE tenant_id = :tenant_id"
                 "   AND company_id IS NOT DISTINCT FROM :company_id"
                 "   AND title = :title"),
            {"tenant_id": tenant_id, "company_id": company_id, "title": title},
        )
        row = existing.fetchone()

        if row is not None:
            await session.execute(
                text("""
                    UPDATE report_corpus SET
                        industry       = COALESCE(:industry,      industry),
                        report_type    = :report_type,
                        published_at   = COALESCE(:published_at,  published_at),
                        milvus_ids     = CAST(:milvus_ids AS JSONB),
                        is_active      = TRUE
                    WHERE id = :id
                """),
                {"id": row[0], "industry": industry, "report_type": report_type,
                 "published_at": published_at,
                 "milvus_ids": json.dumps(milvus_ids, ensure_ascii=False)},
            )
            row_id = str(row[0])
        else:
            result = await session.execute(
                text("""
                    INSERT INTO report_corpus
                        (tenant_id, company_id, industry, title, report_type,
                         published_at, milvus_ids)
                    VALUES
                        (:tenant_id, :company_id, :industry, :title, :report_type,
                         COALESCE(:published_at, NOW()), CAST(:milvus_ids AS JSONB))
                    RETURNING id
                """),
                {"tenant_id": tenant_id, "company_id": company_id, "industry": industry,
                 "title": title, "report_type": report_type, "published_at": published_at,
                 "milvus_ids": json.dumps(milvus_ids, ensure_ascii=False)},
            )
            row_id = str(result.scalar_one())
        await session.commit()

    logger.info("repo.report_corpus_upserted", tenant_id=tenant_id,
                report_id=report_id, chunks=len(milvus_ids))
    return row_id


def _escape_like_token(s: str) -> str:
    """把 report_key 安全嵌进 LIKE 前缀，转义掉 `\\ % _` 三个通配/转义符。

    report_key 不是纯受控键（样例是 `sample-{stem}-{hash}`），必须当不可信输入，
    否则一个含 `%` 的 report_key 会把 LIKE 前缀匹配放大成全表匹配。
    Postgres LIKE 的默认转义符就是反斜杠，故无需显式 ESCAPE 子句。

    :param s: 原始 report_key
    :return: 转义后的字符串，可安全用于 LIKE 前缀匹配
    """
    return (s.replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_"))


async def list_report_milvus_ids_by_key(tenant_id: str, report_key: str) -> list[str]:
    """返回落在该 report_key 下的全部切块主键（供 Milvus 精确删除用）。

    report_key 不独立落列，只以 {report_key}:{index} 的前缀隐式存在于登记行的
    milvus_ids 里 —— 所以先用 LIKE 前缀匹配定位登记行，再解出精确的 id 清单。
    去重后返回扁平列表；无命中返回空列表（删除「不存在的东西」是合法的）。

    :param tenant_id: 租户隔离键
    :param report_key: 研报唯一键（与登记行 milvus_ids 前缀匹配）
    :return: 该 report_key 下全部去重后的切片主键列表
    """
    prefix = _escape_like_token(report_key) + ":%"
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("""
                SELECT DISTINCT jsonb_array_elements_text(milvus_ids) AS mid
                FROM report_corpus
                WHERE tenant_id = :t
                  AND EXISTS (
                      SELECT 1
                      FROM jsonb_array_elements_text(milvus_ids) AS e
                      WHERE e LIKE :prefix
                  )
            """),
            {"t": tenant_id, "prefix": prefix},
        )
        return [row[0] for row in result.fetchall()]


async def delete_report_corpus(tenant_id: str, report_key: str) -> int:
    """按 report_key 前缀删除对应登记行，返回删掉的行数。

    与 list_report_milvus_ids_by_key 用同一套前缀定位，保证「拿到了哪些 id 去删
    向量」和「删掉哪些登记行」始终对齐同一批文本。没有命中返回 0。

    :param tenant_id: 租户隔离键
    :param report_key: 研报唯一键（前缀定位删除）
    :return: 被删除的登记行数；无命中返回 0
    """
    prefix = _escape_like_token(report_key) + ":%"
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("""
                DELETE FROM report_corpus
                WHERE tenant_id = :t
                  AND EXISTS (
                      SELECT 1
                      FROM jsonb_array_elements_text(milvus_ids) AS e
                      WHERE e LIKE :prefix
                  )
            """),
            {"t": tenant_id, "prefix": prefix},
        )
        await session.commit()
        return result.rowcount or 0


async def list_report_corpus(tenant_id: str, limit: int = 50) -> list[dict]:
    """列出语料登记行，按发布时点倒序。

    :param tenant_id: 租户隔离键
    :param limit: 返回条数上限，默认 50
    :return: 语料登记行 dict 列表（含 milvus_ids 对齐信息）
    """
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("SELECT * FROM report_corpus WHERE tenant_id = :t"
                 " ORDER BY published_at DESC LIMIT :limit"),
            {"t": tenant_id, "limit": limit},
        )
        return _rows_to_dicts(result.fetchall())


# ── 风控审核（人工签字）─────────────────────────────────────────

async def insert_risk_review(tenant_id: str, task_id: str, report_id: str,
                             checklist: dict) -> str:
    """插一行未签字的审核记录，返回其 id。

    签字三列（reviewer_id / decision / comments）一律留 NULL —— 表上的
    chk_signed_complete 约束只允许「三列全 NULL」或「三列全有值且意见 ≥10 字」，
    所以这一行在人工签字之前【不可能】被写成半个签字。

    :param tenant_id: 租户隔离键
    :param task_id: 任务 id
    :param report_id: 待审研报（草稿）id
    :param checklist: 自动预检结果快照
    :return: 审核记录 id
    """
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("""
                INSERT INTO risk_reviews (tenant_id, task_id, report_id, checklist)
                VALUES (:t, :task, :report, CAST(:checklist AS JSONB))
                RETURNING id
            """),
            {"t": tenant_id, "task": task_id, "report": report_id,
             "checklist": json.dumps(checklist, ensure_ascii=False)},
        )
        review_id = str(result.scalar_one())
        await session.commit()
    logger.info("repo.risk_review_inserted", task_id=task_id, review_id=review_id)
    return review_id


async def get_latest_risk_review(tenant_id: str, task_id: str) -> Optional[dict]:
    """取最近一行审核记录 —— 驳回重做会产生多行，只有最后一行是待签的。

    :param tenant_id: 租户隔离键
    :param task_id: 任务 id
    :return: 最新审核行 dict；无记录返回 None
    """
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("SELECT * FROM risk_reviews WHERE tenant_id = :t AND task_id = :task"
                 " ORDER BY created_at DESC, id DESC LIMIT 1"),
            {"t": tenant_id, "task": task_id},
        )
        row = result.fetchone()
    return _row_to_dict(row) if row else None


async def list_risk_reviews(tenant_id: str, task_id: str) -> list[dict]:
    """按创建顺序列出某任务的审核记录（含历史重做产生的多行）。

    :param tenant_id: 租户隔离键
    :param task_id: 任务 id
    :return: 审核行 dict 列表
    """
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("SELECT * FROM risk_reviews WHERE tenant_id = :t AND task_id = :task"
                 " ORDER BY created_at, id"),
            {"t": tenant_id, "task": task_id},
        )
        return _rows_to_dicts(result.fetchall())


async def sign_risk_review(tenant_id: str, review_id: str, reviewer_id: str,
                           decision: str, comments: str,
                           redo_targets: Optional[dict] = None) -> None:
    """把签字写进待审核的那一行。

    只更新 reviewer_id IS NULL 的行：重复签字不会覆盖掉上一个人的留痕
    （那既是审计要求，也是「谁签的字」这个问题的唯一答案）。
    意见长度由 chk_signed_complete 在数据层把住 —— 应用层也校验一次是为了给出
    可读的 400，而不是让用户拿到一个 IntegrityError 的 500。

    :param tenant_id: 租户隔离键
    :param review_id: 待签字的审核记录 id
    :param reviewer_id: 真实签字人用户 id
    :param decision: 裁决，∈ approve/reject (modify 走 redo)
    :param comments: 签字意见，长度须 ≥10 字
    :param redo_targets: 驳回时指定的重做目标，可空
    :return: 无返回值；重复签字或记录不存在时抛 RuntimeError
    """
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("""
                UPDATE risk_reviews
                SET reviewer_id  = :reviewer,
                    decision     = :decision,
                    comments     = :comments,
                    redo_targets = CAST(:redo_targets AS JSONB),
                    signed_at    = NOW()
                WHERE tenant_id = :t AND id = :id AND reviewer_id IS NULL
            """),
            {"t": tenant_id, "id": review_id, "reviewer": reviewer_id,
             "decision": decision, "comments": comments,
             "redo_targets": json.dumps(redo_targets, ensure_ascii=False)
                             if redo_targets else None},
        )
        if result.rowcount == 0:
            # 已经被签过（或行不存在）。不抛异常吞掉：静默失败会让调用方以为签上了。
            await session.rollback()
            raise RuntimeError(f"审核记录 {review_id} 不存在或已被签字")
        await session.commit()


# ── 审计 ────────────────────────────────────────────────────────

async def write_audit_log(tenant_id: str, actor_id: Optional[str], action: str,
                          target_type: str, target_id: Optional[str],
                          detail: Optional[dict] = None) -> None:
    """写一条审计流水。审计失败【不能】影响主流程，因此异常只记日志。

    :param tenant_id: 租户隔离键
    :param actor_id: 操作者用户 id；系统动作可传 None
    :param action: 动作名，如 task.publish、risk.sign
    :param target_type: 目标对象类型
    :param target_id: 目标对象 id，可空
    :param detail: 明细载荷，可空
    :return: 无返回值
    """
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(
                text("""
                    INSERT INTO audit_logs (tenant_id, actor_id, action, target_type, target_id, detail)
                    VALUES (:t, :actor, :action, :target_type, :target_id, CAST(:detail AS JSONB))
                """),
                {"t": tenant_id, "actor": actor_id, "action": action,
                 "target_type": target_type, "target_id": target_id,
                 "detail": json.dumps(detail or {}, ensure_ascii=False)},
            )
            await session.commit()
    except Exception as e:                      # noqa: BLE001 —— 审计不能拖垮业务
        logger.warning("repo.audit_write_failed", action=action, error=str(e))


# ── 流水线暂停点（CheckPoint） ───────────────────────────────────
# 取代 langgraph 的 PostgresSaver 全量快照：这里只存「暂停在哪个段、等什么」。
# 「暂停态的全部内容」在业务表（草稿、待审、预检清单），checkpoint 只负责回答
# 「续跑从哪里前进」。status=awaiting_risk_review 是业务侧暂停信号，
# 这里多一张表是为了给前进式续跑一个明确的「原点是这里」的落点。

async def upsert_checkpoint(tenant_id: str, task_id: str, stage: str, step: str,
                            payload: Optional[dict] = None) -> None:
    """写/覆盖一个暂停点的 checkpoint。同一任务只保留最新一条。

    :param tenant_id: 租户隔离键
    :param task_id: 任务 id
    :param stage: 暂停所在阶段
    :param step: 暂停所在步骤
    :param payload: 续跑载荷（如签字决策注入），可空
    :return: 无返回值
    """
    now = cn_now()
    async with AsyncSessionLocal() as session:
        await session.execute(
            text("""
                INSERT INTO pipeline_checkpoints (tenant_id, task_id, stage, step, payload, updated_at)
                VALUES (:t, :task, :stage, :step, CAST(:payload AS JSONB), :now)
                ON CONFLICT (task_id) DO UPDATE
                    SET stage = EXCLUDED.stage, step = EXCLUDED.step,
                        payload = EXCLUDED.payload, updated_at = EXCLUDED.updated_at
            """),
            {"t": tenant_id, "task": task_id, "stage": stage, "step": step,
             "payload": json.dumps(payload or {}, ensure_ascii=False), "now": now},
        )
        await session.commit()


async def get_checkpoint(tenant_id: str, task_id: str) -> Optional[dict]:
    """读暂停点；不存在返回 None（调用方据此判定：无暂停 / 进程重启后不一致）。

    ⚠️ 不能用 result.rowcount 判断有没有行：SQLAlchemy 的 text().execute 对 SELECT
    的 rowcount 恒为 -1（truthy），查无行时也是 -1，于是 'no row' 与 '有行' 都被
    rowcount 判成「有」—— 只能基于 fetchone() 是否取到行来判定。

    :param tenant_id: 租户隔离键
    :param task_id: 任务 id
    :return: 暂停点行 dict；不存在返回 None
    """
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("SELECT * FROM pipeline_checkpoints WHERE tenant_id = :t AND task_id = :task LIMIT 1"),
            {"t": tenant_id, "task": task_id},
        )
        row = result.fetchone()
        return _row_to_dict(row) if row is not None else None


async def clear_checkpoint(tenant_id: str, task_id: str) -> None:
    """续跑收口后清除暂停点。失败不能拖垮主流程，因此只记日志。

    :param tenant_id: 租户隔离键
    :param task_id: 任务 id
    :return: 无返回值
    """
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(
                text("DELETE FROM pipeline_checkpoints WHERE tenant_id = :t AND task_id = :task"),
                {"t": tenant_id, "task": task_id},
            )
            await session.commit()
    except Exception as e:                      # noqa: BLE001
        logger.warning("repo.checkpoint_clear_failed", task_id=task_id, error=str(e))
