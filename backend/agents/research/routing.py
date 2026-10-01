# backend/agents/research/routing.py
# 风控签字之后的条件路由 —— 整张图唯一的回边出口，也是死循环防护的落点。

from backend.agents.research.state import ResearchState
from backend.config import get_settings
from backend.core.logger import get_logger

logger = get_logger(__name__)


def route_after_risk_review(state: ResearchState) -> str:
    """风控决策分发：放行、驳回重做、或阻断。

    这个函数决定图能走多远，因此它的每一条分支都必须有测试：
    一个漏掉的 return 就是一条永远不会被走到的死边，
    而一个多出来的兜底 return 可能把「未签字」悄悄放行成「已发布」。

    :param state: 续跑后的父 State，读 risk_decision/redo_targets/redo_count/task_id。
    :return: 父图中的节点名（publish_report / analyze_dimensions / build_valuation /
        mark_rejected / mark_failed），必须与 graph.py 里 add_node 注册的名字一致。
    """
    settings = get_settings()
    decision = state.get("risk_decision")
    targets = state.get("redo_targets") or {}
    redo_count = state.get("redo_count", 0)

    # ── 放行 ───────────────────────────────────────────────────
    if decision in ("approve", "modify"):
        return "publish_report"

    # ── 驳回：回边到阶段边界，或超限转人工 ─────────────────────
    if decision == "reject":
        if redo_count >= settings.research_max_redo:
            # 死循环防护：反复驳回说明这不是「再跑一遍能解决」的问题，
            # 继续回边只会烧钱，转人工介入
            logger.warning("routing.redo_limit_reached",
                           task_id=state.get("task_id"), redo_count=redo_count)
            return "mark_rejected"

        stage = targets.get("stage")
        if stage == "analyze":
            return "analyze_dimensions"
        if stage == "valuation":
            return "build_valuation"

        # 驳回但目标阶段不可识别 = 状态已损坏。
        # 这里【不】猜一个阶段去重跑：猜错会掩盖问题，还会白跑一轮。
        logger.error("routing.reject_with_unknown_stage",
                     task_id=state.get("task_id"), targets=targets)
        return "mark_failed"

    # ── decision 缺失或不可识别 ────────────────────────────────
    # 绝不能落到「放行」：未签字的研报不得发布，这是监管底线。
    logger.error("routing.risk_decision_missing",
                 task_id=state.get("task_id"), decision=decision)
    return "mark_failed"
