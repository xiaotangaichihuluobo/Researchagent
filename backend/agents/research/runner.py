# backend/agents/research/runner.py
# 投研流水线引擎 —— 取代 LangGraph 的父图与 checkpointer。
#
# 核心是一个表驱动的 while 循环（_drive）：沿着 NEXT_STAGE 线性推进五个段，每段进入前
# validate_requires、出段后 verify_produces（handoff 契约闸门）；走到 risk 段时停下等签字。
# 「停 / 完成 / 失败」三种收场全在这层收口。
#
# P6 起的暂停语义落在两处：
#   · 业务侧：任务 status 翻成 awaiting_risk_review（前端据此展示待审列表）；
#   · 引擎侧：run_risk_stage 首跑写一条轻量 checkpoint（pipeline_checkpoints）
#            并返回 PENDING_SIGN_OFF 哨兵，_drive 据此回 "paused"，任务停在等待。
#   resume 时用业务表重建父 State → 从 risk 段前进式续跑（注入签字，不重跑闸门之前）。
#   checkpoint 只回答「原点是这里」，暂停态的全部内容都在业务表 —— 见 risk/nodes.py 头注。

import asyncio

from backend.core import research_repo as repo
from backend.core import retry as core_retry
from backend.core import usage_ctx
from backend.core.logger import get_logger
from backend.agents.research import events, steps
from backend.agents.research.protocols import validate_requires, verify_produces
from backend.agents.research.routing import route_after_risk_review
from backend.agents.research.state import NEXT_STAGE, PENDING_SIGN_OFF

logger = get_logger(__name__)

# ── stage 级重试（F）────────────────────────────────────────────
# 流水线的线性段偶发打外部（LLM/Milvus）会掉进网络抖动 —— 但每条失败都整链翻车太贵。
# 这里只对「瞬时/可重试」异常（复用 retry.py 的分类）做指数退避重试，重试用尽仍失败
# 才上抛让 _run_pipeline 走 _fail_task。风控段是 HITL 决策、且本就 NON_DEGRADABLE，
# 不纳入重试。
STAGE_MAX_RETRIES  = 3
STAGE_RETRY_DELAYS = [1.0, 2.0, 4.0]


async def _run_stage_with_retry(stage: str, state: dict) -> dict:
    """包住 steps.run_stage 做有限次指数退避重试。返回运行后的 state。

    只重试 core_retry.RETRYABLE_ERRORS 内的异常；阻断/不可重试错误立即上抛，
    重试耗尽也上抛 —— 收场仍统一归 _fail_task，这里不吞错。

    :param stage: 要跑的线性段名（collect/analyze/retrieve/valuation）。
    :param state: 当前父 State；只读，真正的推进在 steps.run_stage 内部。
    :return: 该段跑完后更新过的完整 state（dict）。
    """
    for attempt in range(STAGE_MAX_RETRIES + 1):        # 首次 + STAGE_MAX_RETRIES 次重试
        if attempt > 0:
            delay = STAGE_RETRY_DELAYS[min(attempt - 1, len(STAGE_RETRY_DELAYS) - 1)]
            await asyncio.sleep(delay)
        try:
            return await steps.run_stage(stage, state)
        except (*core_retry.BLOCKING_ERRORS, *core_retry.NON_RETRYABLE_ERRORS) as e:
            logger.warning("runner.stage_no_retry", stage=stage, error=str(e)[:200])
            raise
        except core_retry.RETRYABLE_ERRORS as e:
            if attempt < STAGE_MAX_RETRIES:
                logger.warning("runner.stage_retry", stage=stage,
                               attempt=attempt + 1,
                               delay=STAGE_RETRY_DELAYS[min(attempt, len(STAGE_RETRY_DELAYS) - 1)],
                               error=str(e)[:200])
                continue
            logger.error("runner.stage_retries_exhausted", stage=stage, error=str(e)[:200])
            raise
        except Exception as e:                          # 其它不可重试异常，原样上抛
            raise

# 持有正在运行的任务引用（start 与 resume 共用）。
# 必须有这个 set：asyncio.create_task 返回的 Task 若不保存引用，可能被 GC 掉，
# 任务会在跑到一半时静默消失（官方文档明确要求保存引用）。
# 用 set 而非 dict：set 化只需要「登记 + 结束移除」，并能让结束回调里
# `_running.discard` 正好以回调传入的 Task 对象为参 —— 幂等、O(1)。
# 存 Task 对象而非 task_id：start 与 resume 是同 id 的两个不同 Task，
# 用 id 做 dict 键会互相顶掉引用，set 两边各自保住。
_running: set[asyncio.Task] = set()


def _spawn(coro) -> None:
    """把协程挂到事件循环上并保住引用。任务级日志由 caller 带 task_id。

    :param coro: 待调度的协程对象（如 _run_pipeline(task)）。
    :return: 无返回值。
    """
    bg = asyncio.create_task(coro)
    _running.add(bg)
    # 任务结束就从登记表移除：done_callback 传入的就是 bg 本身。
    bg.add_done_callback(_running.discard)


async def _fail_task(tenant_id: str, task_id: str, error: Exception) -> None:
    """把一次异常收成任务失败状态。

    任何未捕获异常都必须落到 research_tasks.status='failed'：停在 running 的前端会永远
    轮询。补偿写入自身也必须闭合 —— 这里一旦二次抛错，协程无人 await，任务从此卡在
    running。抽成独立函数，就是为了让 start 与 resume 两处共用同一份不会走样的补偿。

    :param tenant_id: 租户，数据隔离键。
    :param task_id: 任务唯一标识。
    :param error: 触发的异常，用于记日志与 last_error。
    :return: 无返回值。
    """
    # 空串 error（如无参异常/超时 str() 为 ""）只靠 error= 会丢全部上下文。
    # 这里补打完整 traceback 并在阶段事件里留栈，定位「到底哪一步崩」不再靠猜。
    import traceback
    tb = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    logger.error("runner.pipeline_failed", task_id=task_id, error=str(error) or repr(error),
                 trace=tb[-2000:])
    try:
        await repo.update_task_status(
            tenant_id, task_id, status="failed", last_error=str(error) or repr(error),
            mark_finished=True,
        )
        await repo.record_stage_event(tenant_id, task_id, "pipeline", "failed",
                                     {"error": str(error)[:500], "trace": tb[-2000:]})
    except Exception as ce:                         # noqa: BLE001 —— 补偿失败也只能记日志
        logger.error("runner.compensation_failed", task_id=task_id, error=str(ce)[:500])
    # 失败也是终态，主动推给订阅者，别再让前端干等下一次轮询。
    events.publish(tenant_id, events.outcome_event(task_id, "failed"))


async def _drive(state: dict, start_stage: str, resume_payload: dict | None = None):
    """表驱动推进流水线，直到【停】或【终】。

    start_stage 允许从 collect 之外进入（驳回回边重跑 analyze/valuation 时用）。

    :param state: 父 State；首跑由 _run_pipeline 构造，续跑由 _rebuild_state 重建。
    :param start_stage: 起始段名；首跑为 "collect"，续跑为 "risk"。
    :param resume_payload: 续跑时的签字载荷（decision/comments 等）；首跑为 None。默认 None。
    :return: (outcome, state)，outcome ∈ {"published", "rejected", "failed", "paused"}。
    """
    stage = start_stage
    while True:
        if stage == "risk":
            validate_requires("risk", state)
            out = await steps.run_risk_stage(state, resume_payload=resume_payload)
            resume_payload = None                    # 只有首次进 risk 用 resume 载荷
            state = out
            if state.get("paused"):
                return "paused", state
            verify_produces("risk", state)

            # 已裁决 → 路由。分支名来自 routing.py（原 graph.py 的条件边表）：
            target = route_after_risk_review(state)
            if target == "publish_report":
                await steps.publish_report_final(state)
                return "published", state
            if target == "mark_rejected":
                await steps.mark_rejected_final(state)
                return "rejected", state
            if target == "mark_failed":
                await steps.mark_failed_final(state)
                return "failed", state
            # 驳回回边：从契约节点重新进 analyze / valuation（重跑的路也走同一道闸门）。
            stage = "analyze" if target == "analyze_dimensions" else "valuation"
            continue

        # 线性段：进入前契约校验 → 跑段（带指数退避重试）→ 出段契约校验 → 推向下一段。
        validate_requires(stage, state)
        state = await _run_stage_with_retry(stage, state)
        verify_produces(stage, state)
        stage = NEXT_STAGE[stage]


async def _run_pipeline(task: dict) -> None:
    """第一次跑：从 START 开始。

    :param task: 来自业务表的任务行 dict，读取 tenant_id/task_id/company_id/
        company_code/industry/redo_count 等键。
    :return: 无返回值。
    """
    tenant_id = task["tenant_id"]
    task_id = str(task["task_id"])
    await repo.update_task_status(tenant_id, task_id, status="running", mark_started=True)

    state = {
        "task_id": task_id,
        "tenant_id": tenant_id,
        "company_id": str(task["company_id"]),
        "company_code": task["company_code"],
        "industry": task.get("industry") or "",
        "redo_count": task.get("redo_count", 0),
    }
    try:
        # 包一层 usage 上下文：整条链（含 _drive 内每次 LLM 调用、with_retry 每次重试）
        # 都会带上 tenant_id/task_id 落 llm_usage，按任务聚合成本。
        async with usage_ctx.usage_ctx(tenant_id=tenant_id, task_id=task_id):
            outcome, _ = await _drive(state, start_stage="collect")
        events.publish(tenant_id, events.outcome_event(task_id, outcome))
        if outcome == "paused":
            logger.info("runner.pipeline_paused", task_id=task_id)
        else:
            logger.info("runner.pipeline_done", task_id=task_id, outcome=outcome)
    except Exception as e:                          # noqa: BLE001 —— 兜住一切，别卡在 running
        await _fail_task(tenant_id, task_id, e)


async def _resume_pipeline(task: dict, decision: dict) -> None:
    """从风控闸门接着跑。

    ⚠️ 调用方必须先确认这个 thread 上【确实有】暂停记录（API 层守卫会查 checkpoint）。
    这里再做一次同款校验作为保底：没有 checkpoint 却跑来 resume，说明「进程重启+状态
    不一致」，前进式续跑的结果不可信 —— 宁可失败也不凭空放行。

    :param task: 来自业务表的任务行 dict，用于读 tenant_id/task_id。
    :param decision: 审核签字载荷 dict，含 decision/comments/reviewer_id/redo_targets 等键。
    :return: 无返回值。
    """
    tenant_id = task["tenant_id"]
    task_id = str(task["task_id"])

    cp = await repo.get_checkpoint(tenant_id, task_id)
    if cp is None:
        await _fail_task(tenant_id, task_id,
                         RuntimeError("没有可续跑的暂停点：任务状态与执行器不一致"))
        return

    try:
        # 续跑同样包 usage 上下文（含 _drive 内每次 LLM 调用）。
        async with usage_ctx.usage_ctx(tenant_id=tenant_id, task_id=task_id):
            state = await _rebuild_state(tenant_id, task_id)
            outcome, _ = await _drive(state, start_stage="risk", resume_payload=decision)
        events.publish(tenant_id, events.outcome_event(task_id, outcome))
        logger.info("runner.pipeline_resumed", task_id=task_id,
                    decision=decision.get("decision"), outcome=outcome)
    except Exception as e:                          # noqa: BLE001 —— 与 start 同一条理由
        await _fail_task(tenant_id, task_id, e)


async def _rebuild_state(tenant_id: str, task_id: str) -> dict:
    """用业务表重建父 State（前进式续跑的重做路径要重新消费这些键）。

    checkpoint 只记「原点是 risk 段」，不背正文；这里从真相来源（业务表）把父 State
    里下游要读的键捡回来。重建缺失时宁可少带键、也别塞默认值 —— 少键会被 handoff
    契约闸门在边界上抓成响亮失败。

    :param tenant_id: 租户，数据隔离键。
    :param task_id: 任务唯一标识。
    :return: 重建出的父 State dict，键与 risk 段之后的下游契约对齐
        （task_id/tenant_id/company_code/industry/redo_count/collected_ids/
        dimension_results/rating/comparison_points/valuation_available 等）。
    """
    t = await repo.get_task(tenant_id, task_id)
    if not t:
        raise RuntimeError("续跑找不到任务行，任务已被删除？")

    company = await repo.get_company_by_id(tenant_id, str(t["company_id"]))
    state: dict = {
        "task_id": task_id,
        "tenant_id": tenant_id,
        "company_id": str(t["company_id"]),
        "company_code": company["code"] if company else str(t["company_id"]),
        "industry": (company or {}).get("industry") or "",
        "redo_count": t.get("redo_count", 0),
        # 驳回重做 analyze 需要 collected_ids 非空；用表里的采集行 id 重建。
        "collected_ids": [str(r["id"]) for r in
                          await repo.fetch_data_items(tenant_id, task_id)],
    }

    dimension_results = await repo.fetch_current_dimension_results(tenant_id, task_id)
    if dimension_results:
        state["dimension_results"] = dimension_results

    report = await repo.get_report(tenant_id, task_id)
    if report:
        state["rating"] = report.get("rating") or "未评级"
        state["has_reference"] = bool(report.get("comparison_points"))
        state["comparison_points"] = report.get("comparison_points") or []

    valuation = await repo.fetch_valuation_result(tenant_id, task_id)
    state["valuation_available"] = bool(valuation and valuation.get("is_available"))

    review = await repo.get_latest_risk_review(tenant_id, task_id)
    if review:
        state["compliance_checklist"] = review.get("checklist") or {}

    return state


def start_pipeline(task: dict) -> None:
    """把流水线挂到事件循环上，立即返回。投研任务是分钟级的，HTTP 不能同步等它。

    :param task: 来自业务表的任务行 dict（含 tenant_id/task_id/company_id 等键）。
    :return: 无返回值。
    """
    task_id = str(task["task_id"])
    _spawn(_run_pipeline(task))
    logger.info("runner.pipeline_started", task_id=task_id)


def resume_pipeline(task: dict, decision: dict) -> None:
    """签字之后接着跑，立即返回。同样不能同步等 —— 重跑一次分析是分钟级的。

    :param task: 来自业务表的任务行 dict（含 tenant_id/task_id 键）。
    :param decision: 审核签字载荷 dict，含 decision/comments/reviewer_id 等键。
    :return: 无返回值。
    """
    task_id = str(task["task_id"])
    _spawn(_resume_pipeline(task, decision))
    logger.info("runner.pipeline_resuming", task_id=task_id,
                decision=decision.get("decision"))