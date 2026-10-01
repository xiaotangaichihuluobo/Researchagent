# backend/core/retry.py
# 三层兜底机制：自动重试 → Agent 级降级 → 系统级兜底

import asyncio                                   # 异步：用于超时控制和等待
from dataclasses import dataclass                # 哨兵类型：见 FallbackResult
from functools import wraps                      # @wraps：装饰器里保留原函数的名字/文档
from typing import Callable, Any, Optional       # 类型注解：可调用对象 / 任意 / 可选

from backend.core.exceptions import (            # 引入 3.3 定义的异常（已去掉 Judge0 的 Sandbox 异常）
    LLMAPIError,
    MilvusConnectionError,
    InvalidInputError,
    AuthenticationError,
    NonDegradableError,
    TaskBudgetExceeded,
)
from backend.core.logger import get_logger          # configure_logging 曾只被末尾的演示块用到，随它一起删了
logger = get_logger(__name__)

# ── 异常分类 ───────────────────────────────────────────────
# 可重试：多半是短暂故障（网络抖动、超时），重试一下可能就好
RETRYABLE_ERRORS = (
    LLMAPIError,
    MilvusConnectionError,
    TimeoutError,
    ConnectionError,
)
# 不可重试：重试也没用（输入非法、认证失败），应立即抛出
NON_RETRYABLE_ERRORS = (
    InvalidInputError,
    AuthenticationError,
)
# 必须阻断：既不重试、也不降级，原样上抛让任务停下来。
# 与上面那组走同一个 except 分支（那个分支的 raise 会直接穿出 wrapper），
# 分开成一个元组只为在日志里能区分「输入不合法」与「任务必须阻断」——
# 事后排查这两种情况要看的地方完全不同。
BLOCKING_ERRORS = (
    NonDegradableError,
    TaskBudgetExceeded,   # 研报任务 token 预算超限：不该重试(重试只会烧更多)、不该降级(降级=冒充成功)
)

# 这几类 Agent 【永不降级】：重试用尽后仍失败就必须阻断，宁可不发布/不报假答案。
#   · risk —— 风控失败必须阻断任务，宁可不发布。
#   · qa_reports —— 轨道 B 问答宁可对用户报错，也不拿「降级哨兵」冒充答案奉上。
# 为什么不只靠异常类型判定：这两处节点抛的都是通用 LLMAPIError（可重试），
# 从异常上看不出「它必须阻断」，只有按 agent_type 才拦得住。
# 这一条把「不包 with_retry / 不降级」从约定变成了强制 —— 无论谁怎么接线，
# 他们都拿不到一个「降级成功」的哨兵。
NON_DEGRADABLE_AGENTS = frozenset({"risk", "qa_reports"})

MAX_RETRIES = 2                  # 最多重试 2 次（加上首次 = 共 3 次尝试）
RETRY_DELAYS = [1.0, 3.0]        # 第 1 次重试前等 1 秒，第 2 次前等 3 秒
TIMEOUT_PER_ATTEMPT = 120.0       # 单次调用最多等 120 秒，超时算失败


@dataclass(frozen=True)
class FallbackResult:
    """降级结果 —— 【故意不是 dict】。

    装饰器的用法是包住 graph.ainvoke(...)，所以它的返回值顶替的是【整图返回值】。
    以前各降级方法返回 dict，那些键又都不在对应 State schema 里（_collect_fallback
    给 items，CollectState 里叫 raw_items；_retrieve_fallback 给 chunks，schema 里
    叫 retrieved_chunks；_analyze_fallback 给四个平铺键，而它们实际住在
    dimension_results 里）。谁把它当节点输出返回、或并进 state，就会写入 schema
    里不存在的键 —— LangGraph 会静默丢弃或直接报状态校验错。
    而且能冒充结果的形状还带来第二个后果：风控那层可以拿它冒充「一次成功放行」。

    做成哨兵类型后，这两件事结构性地不可能发生：它不是 dict，没有键可并。
    消费方必须显式判别（isinstance），就像 analyze 节点已经在做的那样。
    """

    agent_type: str
    layer: str      # "agent"（第二层，该 Agent 的降级策略生效）/ "system"（第三层，连降级都没兜住）
    note: str       # 给人看的说明。消费方直接拿去用，不必自己再编一句文案



def with_retry(agent_type: str = ""):
    """三层兜底装饰器工厂。给异步函数套上「重试 → 降级 → 系统兜底」三层保护。

    用法：
        @with_retry(agent_type="retrieve")
        async def _invoke():
            return await graph.ainvoke(state, config=config)

    :param agent_type: Agent 类型，用于决定重试/降级/阻断策略；默认空串
    :return: 已被包装的装饰器（接收异步函数并返回带保护的 wrapper）
    """
    def decorator(func: Callable) -> Callable:       # 中间层：接收被装饰的函数
        """装饰器内层：接收被装饰函数并返回包装后的异步函数。

        :param func: 被装饰的异步函数（通常是对 graph.ainvoke 的封装）
        :return: 包装后的异步 wrapper 函数
        """
        @wraps(func)                                 # 保留原函数的元信息（名字、docstring）
        async def wrapper(*args, **kwargs) -> Any:   # 最内层：真正的执行逻辑

            """最内层执行体：承载「重试 → 降级 → 系统兜底」三层保护。

            :param args: 透传给原函数的位置参数
            :param kwargs: 透传给原函数的关键字参数
            :return: 原函数结果；重试用尽后按 agent_type 可能返回降级哨兵 FallbackResult
            """
            # ── 第一层：自动重试 ──────────────────────────
            last_error: Optional[Exception] = None   # 记录最后一次的错误，留给后面降级用
            for attempt in range(MAX_RETRIES + 1):   # 循环 3 次：attempt = 0, 1, 2
                try:
                    # 给单次调用套一个超时；超过 120 秒就抛 TimeoutError
                    result = await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=TIMEOUT_PER_ATTEMPT,
                    )
                    if attempt > 0:                  # 如果是重试后成功的，记一条日志
                        logger.info("retry.succeeded", agent_type=agent_type, attempt=attempt + 1)
                    return result                    # 成功，直接返回，结束

                except BLOCKING_ERRORS as e:         # 必须阻断：不重试，也【不降级】
                    # 注意这个 raise 是穿出整个 wrapper 的，不会落到下面的
                    # 第二层 —— 这正是想要的：连重试都不必（重试完也还是无据可依）。
                    logger.error("retry.blocking_error", agent_type=agent_type, error=str(e))
                    raise

                except NON_RETRYABLE_ERRORS as e:    # 不可重试异常：立即抛出，不再重试
                    logger.warning("retry.non_retryable_error", agent_type=agent_type, error=str(e))
                    raise                            # 原样抛出，交给上层处理

                except Exception as e:               # 其它（可重试）异常
                    last_error = e                   # 记下来
                    if attempt < MAX_RETRIES:        # 还没到上限：等待后重试
                        delay = RETRY_DELAYS[attempt]
                        logger.warning(
                            "retry.attempt_failed", agent_type=agent_type,
                            attempt=attempt + 1, max_retries=MAX_RETRIES, delay=delay, error=str(e),
                        )
                        await asyncio.sleep(delay)   # 等 1s 或 3s 再重试
                    else:                            # 到上限了：记录失败，跳出循环去降级
                        logger.error("retry.all_attempts_failed", agent_type=agent_type, error=str(e))

            # ── 不可降级的 Agent：到此为止，原样上抛 ──────
            # 放在第二层【之前】：风控这类 Agent 根本没有「降级交付」这个选项。
            # 重试已经用尽，再往下走就只能拿一个哨兵冒充「风控已通过」，那正是
            # 这条规则以前失效的方式：文案说「任务已阻断」，实际没有任何东西阻断。
            if agent_type in NON_DEGRADABLE_AGENTS:
                logger.error("retry.degradation_refused", agent_type=agent_type,
                             error=str(last_error))
                # last_error 理论上必非空（走到这里说明循环跑满了），兜一下是
                # 为了不让「raise None」变成 TypeError 掩盖掉真正的错误。
                raise last_error or RuntimeError(
                    f"{agent_type} 不可降级，但重试循环没有留下异常")

            # ── 第二层：Agent 级降级 ──────────────────────
            try:
                fallback_result = await AgentFallbackHandler.handle(  # 按 agent_type 找降级策略
                    agent_type=agent_type, original_error=last_error,
                )
                logger.info("retry.fallback_succeeded", agent_type=agent_type)
                return fallback_result               # 降级结果（哨兵，不是 dict）
            except Exception as fallback_error:      # 连降级都失败
                logger.error("retry.fallback_failed", agent_type=agent_type, error=str(fallback_error))

            # ── 第三层：系统级兜底 ────────────────────────
            logger.error("retry.system_fallback", agent_type=agent_type, original_error=str(last_error))
            return _system_fallback_response(agent_type)  # 最后的保底，永远不会再失败
        return wrapper
    return decorator



class AgentFallbackHandler:
    """第二层降级：各 Agent 的专项降级策略（尽量保留核心功能，退化为更简单的实现）。"""

    @classmethod
    async def handle(cls, agent_type: str, original_error: Exception) -> Any:
        """根据 agent_type 选择对应的降级策略。

        :param agent_type: Agent 类型，用于查降级策略表
        :param original_error: 原始异常，作为兜底抛出使用
        :return: 降级结果 FallbackResult；若无对应策略则原样抛出 original_error
        """
        fallback_map = cls.describe_registry()        # 类型 → 降级方法 的映射表
        handler = fallback_map.get(agent_type)        # 查表
        if handler:
            return await handler()
        raise original_error                          # 没有对应降级策略，原样抛出（交给系统兜底）

    @classmethod
    def describe_registry(cls) -> dict:
        """返回当前注册的降级策略表。**测试用来钉住它的形状** ——
        注册表里只该有可达的策略，而「可达」这件事需要一条断言来守，
        不能只写在注释里。

        :return: {agent_type: 降级方法} 的注册表字典，供 handle 与测试断言使用
        """

        # 只注册【可达】的降级策略。
        # "collect" 已于 P5 摘除：采集子图零个 LLM 调用点，没有任何地方调
        #   with_retry("collect")，那条 _collect_fallback 是不可达的死代码。
        #   采集的失败模式是**源故障**（已由 gather(return_exceptions=True) 逐源处理
        #   + CollectExhaustedError 阻断），不存在 LLM 层降级。
        #   留着一条永远不可达的注册，就是又一条「写在文档里的机制」。
        # "risk" 故意不注册降级处理器：风控失败必须阻断任务，宁可不发布。
        #   查不到 handler 时 handle() 会原样抛出，由调用方决定任务是 failed 还是继续。
        return {
            "analyze":   cls._analyze_fallback,
            "retrieve":  cls._retrieve_fallback,
            "valuation": cls._valuation_fallback,
        }

    @classmethod
    async def _analyze_fallback(cls) -> FallbackResult:
        """分析降级：该维度判定为数据不足 —— 绝不补零。

        :return: agent 层降级哨兵 FallbackResult，note 注明「数据不足」
        """
        logger.info("fallback.analyze_dimension_insufficient")
        return FallbackResult(
            agent_type="analyze", layer="agent",
            note="该维度分析服务暂时不可用，已标记为数据不足。",
        )

    @classmethod
    async def _retrieve_fallback(cls) -> FallbackResult:
        """RAG 降级：无历史参照，不阻断流程。

        :return: agent 层降级哨兵 FallbackResult，note 注明「无历史参照」
        """
        logger.info("fallback.retrieve_no_reference")
        return FallbackResult(
            agent_type="retrieve", layer="agent",
            note="研报检索服务暂时不可用，本次未提供历史参照。",
        )

    @classmethod
    async def _valuation_fallback(cls) -> FallbackResult:
        """估值降级：标注不可用 —— 金融场景绝不编造估值数字。

        :return: agent 层降级哨兵 FallbackResult，note 注明「不提供估值区间」
        """
        logger.info("fallback.valuation_unavailable")
        return FallbackResult(
            agent_type="valuation", layer="agent",
            note="估值服务暂时不可用，本次不提供估值区间。",
        )


def _system_fallback_response(agent_type: str) -> FallbackResult:
    """第三层：系统级兜底。所有降级都失败后返回它。

    这里【没有 risk 这一条】：风控属于 NON_DEGRADABLE_AGENTS，在第二层之前就
    上抛了，永远到不了第三层。以前这里有一句「风控服务不可用，任务已阻断」，
    是句空头支票 —— 它被返回给调用方之后，没有任何东西真的阻断。
    第三层现在只剩 analyze / retrieve / valuation 三条（同 describe_registry）：
    collect 的死降级策略已于 P5 摘除，源头都不再注册，第三层也不该有它。

    :param agent_type: Agent 类型，用于从 notes 表取对应的兜底文案
    :return: system 层降级哨兵 FallbackResult（永远不失败、直接可返回）
    """
    notes = {
        "analyze":   "多维分析服务暂时不可用，各维度已标记为数据不足。",
        "retrieve":  "研报检索服务暂时不可用，本次未提供历史参照。",
        "valuation": "估值服务暂时不可用，本次不提供估值区间。",
    }
    return FallbackResult(
        agent_type=agent_type, layer="system",
        note=notes.get(agent_type, "服务暂时不可用，请稍后再试。"),
    )
# 这里原来有一个 if __name__ == '__main__' 演示块：五个用例里四个被注释掉，
# 实际只跑得通一条路径，而且它重绑模块常量 RETRY_DELAYS —— 对着它跑出来的结论
# 与 import 这个模块时并不是同一套配置。现在上面那三层每一条分支都由
# tests/test_retry.py 直接钉住（含「不可降级」「不可重试」「重试后成功」），
# 演示块留着只会是第二份真相，故删除。