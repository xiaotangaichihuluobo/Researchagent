# backend/core/usage_ctx.py
# LLM token usage 观测的上下文 + 写库（contextvar 打通任务/会话维度）。
#
# 为什么需要 contextvar：LLMFactory 是缓存单例（get_llm 按 (模型,温度,流式) 缓存），
# __init__ 不接收、也不持有任何 request scope —— _complete/_stream 内部拿不到
# task_id / thread_id。要落出带关联键的 usage 行，就给每次任务/问答包一层
# ContextVar：研报任务在 start_pipeline / resume_pipeline 包、问答在 run_qa 包，
# 下游所有 LLM 调用（含 with_retry 每次重试）都落在 context 内、读得到。
#
# 铁规矩：观测绝不能被当成第二份故障源。落库失败只记日志、不抛——
# 否则观测会把一次好好的 LLM 调用打成一个"可重试错误"，被 with_retry 带偏。
# 这与 retry.py 的"LLM 层不重试不降级"是同一条分层边界：cost 账不属于业务路径。

import contextlib
from contextvars import ContextVar
from typing import Optional

from sqlalchemy import text

from backend.config import get_settings
from backend.dependencies import AsyncSessionLocal
from backend.core.logger import get_logger
from backend.core.exceptions import TaskBudgetExceeded

logger = get_logger(__name__)

# 当前调用上下文：{tenant_id, task_id, thread_id}。键存在时写库，读不到就只打日志。
# 默认空 dict —— 离线 / 无上下文场景（测试直调 _complete）不炸，也不落库。
_CTX: ContextVar[dict] = ContextVar("llm_usage_ctx", default={})


def get_usage_context() -> dict:
    """读当前线程/协程的 usage 上下文（不 set 过则为空 dict）。

    :return: dict，含可空的 tenant_id / task_id / thread_id 键。
    """
    return _CTX.get() or {}


def usage_ctx(*, tenant_id: Optional[str] = None,
              task_id: Optional[str] = None,
              thread_id: Optional[str] = None) -> contextlib.AbstractAsyncContextManager:
    """上下文管理器：在入口（研报 / 问答）把一个请求的 usage 关联键包住。

    with 内部发起的 LLM 调用（含 create_task 派生的后台协程）都能读到这些键。
    contextvar 由 asyncio 自动复制进 create_task 的 context，故后台跑研报任务时
    在 _spawn 之外 set 即可穿透整条链。

    :param tenant_id: 租户隔离键（轨道 A / B 都有）。
    :param task_id:   研报任务 id（轨道 A 有、B 无 → 写 NULL）。
    :param thread_id: 问答线程 id（轨道 B 有、A 无 → 写 NULL）。
    :return: 异步上下文管理器，退出时自动 reset。
    """
    @contextlib.asynccontextmanager
    async def _manager():
        token = _CTX.set({"tenant_id": tenant_id, "task_id": task_id,
                          "thread_id": thread_id})
        try:
            yield
        finally:
            _CTX.reset(token)
    return _manager()


async def record_llm_usage(*, agent_type: str, vendor: str, model: str,
                           prompt_tokens: Optional[int],
                           completion_tokens: Optional[int],
                           total_tokens: Optional[int],
                           latency_ms: Optional[int],
                           streaming: bool = False) -> None:
    """把一次 LLM 调用写进 llm_usage 表。观测失败只记日志、绝不抛。

    流式拿不到 usage 时 tokens 全传 None → 落 NULL，不降级成 0（0 会污染聚合均值）。

    :param agent_type: 业务 agent 类型（analyze/retrieve/…）。
    :param vendor: 供应商键（qwen/deepseek）。
    :param model: 模型名（qwen-turbo 等）。
    :param prompt_tokens: prompt token 数；流式拿不到为 None。
    :param completion_tokens: 生成 token 数；流式拿不到为 None。
    :param total_tokens: 总 token 数；流式拿不到为 None。
    :param latency_ms: 本次调用耗时（毫秒），方便聚合。
    :param streaming: 是否流式调用。
    :return: 无返回值。
    """
    ctx = get_usage_context()
    # 无上下文（离线/测试直调）→ 打 debug 日志跳过，不让观测拖累纯净调用。
    if not ctx:
        logger.debug("llm_usage.skipped_no_context", agent_type=agent_type)
        return

    try:
        async with AsyncSessionLocal() as session:
            await session.execute(
                text(
                    "INSERT INTO llm_usage"
                    " (tenant_id, task_id, thread_id, agent_type, vendor, model,"
                    "  prompt_tokens, completion_tokens, total_tokens, latency_ms, streaming)"
                    " VALUES (:tenant_id, :task_id, :thread_id, :agent_type, :vendor, :model,"
                    "         :prompt, :completion, :total, :latency, :streaming)"
                ),
                {
                    "tenant_id": ctx.get("tenant_id") or "tenant_default",
                    "task_id": ctx.get("task_id"),
                    "thread_id": ctx.get("thread_id"),
                    "agent_type": agent_type,
                    "vendor": vendor,
                    "model": model,
                    "prompt": prompt_tokens,
                    "completion": completion_tokens,
                    "total": total_tokens,
                    "latency": latency_ms,
                    "streaming": streaming,
                },
            )
            await session.commit()
        logger.info("llm_usage.recorded", agent_type=agent_type,
                    total_tokens=total_tokens, latency_ms=latency_ms)
    except Exception:                                   # noqa: BLE001 —— 观测永不污染 LLM 路径
        logger.warning("llm_usage.record_failed", agent_type=agent_type,
                       exc_info=True)


async def enforce_budget() -> None:
    """研报任务 token 预算闸门：超限抛 TaskBudgetExceeded 中止任务。

    在每次外部 LLM 调用前（llm_factory._complete/_stream 开头）调用。读数复用
    llm_usage 表的累计 SUM(total_tokens)，不另建计数 —— 观测与控制同源共一个收口。

    作用范围只限研报任务（当前 context 带 task_id 的轨道 A）；问答（thread_id）
    与离线调用不受限。配置 research_token_budget=0 视为不限。

    超限最多超过一个调用：用的是「已落库的历史累计」，某次调用把累计推过上限
    会先落库，下一次调用才被拦 —— 预算类闸门无法预知单次调用成本。

    读库失败【放行】并只记日志：成本控制不该因读库抖动误杀一个正常任务，
    与 record_llm_usage 同一条「辅助机制不被当成第二份故障源」的分层边界。
    """
    ctx = get_usage_context()
    task_id = ctx.get("task_id")
    if not task_id:                                # 非研报任务（问答/离线）不受限
        return
    budget = get_settings().research_token_budget
    if not budget:                                 # 0 = 不限
        return

    try:
        async with AsyncSessionLocal() as session:
            used = (await session.execute(
                text("SELECT COALESCE(SUM(total_tokens), 0) FROM llm_usage"
                     " WHERE tenant_id = :t AND task_id = :i"),
                {"t": ctx.get("tenant_id") or "tenant_default", "i": task_id},
            )).scalar_one()
        if used >= budget:
            raise TaskBudgetExceeded(
                f"任务 token 预算超限: 已用 {used} / 上限 {budget}")
    except TaskBudgetExceeded:
        raise                                      # 预算超限是唯一允许上抛的错误
    except Exception:                              # noqa: BLE001 —— 读不到累计值就放行
        logger.warning("llm_usage.budget_check_failed", task_id=task_id,
                       exc_info=True)