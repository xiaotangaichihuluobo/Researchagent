# backend/agents/risk/nodes.py
# 风控子图：预检 → LLM 合规复核 → 落草稿 → 人工签字闸门 → 应用决策。
#
# 五个节点的顺序是设计文档 §5.5 的硬要求，尤其是 persist_draft 必须在闸门【之前】：
# 暂停期间审核人从业务表就能读到全部待审内容，不必去读 checkpoint ——
# checkpoint 是流程状态，业务表才是内容来源（§4.3 的分工）。
#
# 这个子图是整个投研域与 ResearchAgent 最大的行为差别：「未签字不得发布」不是一句注释，
# 而是流程上有一个人必须跨过去的闸门。
#
# 重构说明：闸门不再用 LangGraph 的 interrupt。拆成两步：
#   · persist_draft_node —— 写草稿 + 翻状态（awaiting_risk_review），【没有任何暂停语义】；
#   · human_review_gate_node —— 只回一个哨兵，真正「停住等签字」发生在
#     steps.run_risk_stage（首跑写 checkpoint 后返回 PENDING_SIGN_OFF，引擎据此收口为暂停）；
#     resume 时引擎把签字注入 state["review_payload"]，交给 apply_decision_node 前进式续跑
#     （不重跑闸门之前的任何节点）。

from typing import Optional

from pydantic import BaseModel, Field

from backend.agents.research.report_draft import (
    DIMENSION_LABELS, build_content, build_risk_disclosure, build_title,
)
from backend.agents.risk.prompts import COMPLIANCE_SYSTEM, build_compliance_prompt
from backend.agents.risk.state import RiskState
from backend.core import research_repo as repo
from backend.core.llm_factory import get_structured_llm
from backend.core.logger import get_logger
from backend.core.retry import FallbackResult, with_retry

logger = get_logger(__name__)

# decision 的合法取值。modify 的语义在本轮被正式定义为「有条件通过」：
# 审核意见随研报披露，但内容不改、不重跑 —— 因此它与 approve 一样放行。
# 这消掉了 routing.py:27 那条「若 modify 表示改完再签则是旁路」的含糊记录。
VALID_DECISIONS = ("approve", "modify", "reject")

# 驳回时可回退到的阶段。analyze 之外只留 valuation：
# collect 重跑代价最高且驳回理由极少是「采集错了」，retrieve 目前是占位。
REDO_STAGES = ("analyze", "valuation")

# 签字意见的最短长度，与 risk_reviews.chk_signed_complete 保持一致。
# 应用层也判一次不是为了「双保险」，而是为了把 IntegrityError 变成可读的 400。
MIN_COMMENTS_LEN = 10


async def precheck_compliance_node(state: RiskState) -> dict:
    """四项自动预检：只提供信息，不做裁决。

    裁决权在人 —— 预检不过不等于驳回，预检全过也不等于放行。它唯一的作用是
    把「审核人必须看到的局限」摆到台面上，省得他去逐字读草稿才发现估值是空的。

    结果写进 state 的 compliance_checklist，由 persist_draft 随草稿一起落库。

    :param state: RiskState，读 tenant_id/task_id/valuation_available。
    :return: dict，含 current_stage="risk_review" 与 compliance_checklist（含
        items/blocking 键）。
    """
    tenant_id, task_id = state["tenant_id"], state["task_id"]
    dimensions = await repo.fetch_current_dimension_results(tenant_id, task_id)

    missing = [d for d in ("fundamental", "technical", "sentiment", "industry")
               if d not in dimensions]
    unevidenced = [d for d, item in dimensions.items()
                   if not (item or {}).get("evidence")]
    insufficient = [d for d, item in dimensions.items()
                    if (item or {}).get("data_sufficiency") != "sufficient"]

    checklist = {
        "items": [
            {"key": "dimensions_complete", "label": "四维结论齐全",
             "passed": not missing,
             "note": "本次未产出：" + "、".join(missing) if missing else "四个维度均有结论"},
            {"key": "evidence_backed", "label": "结论有证据支撑",
             "passed": bool(dimensions) and not unevidenced,
             "note": ("以下维度缺证据：" + "、".join(unevidenced)) if unevidenced
                     else "各维度均带证据"},
            {"key": "data_sufficient", "label": "数据充分性达标",
             "passed": not insufficient,
             "note": ("数据不足：" + "、".join(insufficient)) if insufficient
                     else "各维度数据充分性均为 sufficient"},
            {"key": "valuation_available", "label": "估值可用",
             "passed": bool(state.get("valuation_available")),
             "note": "估值已产出" if state.get("valuation_available")
                     else "估值不可用 —— 研报不含目标价（如实记录，不阻断）"},
            # 这一项预检【判不了】：利益冲突只能由签字人自己声明。
            # passed=None 表示「待人工确认」，不用 False 冒充一个自动结论。
            {"key": "conflict_declared", "label": "签字人无利益冲突",
             "passed": None, "note": "由签字人在签字时确认，自动预检无法判定"},
        ],
    }
    checklist["blocking"] = [i["key"] for i in checklist["items"] if i["passed"] is False]

    await repo.record_stage_event(tenant_id, task_id, "risk_review", "started",
                                 {"checklist_blocking": checklist["blocking"]})
    logger.info("risk.precheck_done", task_id=task_id,
                blocking=checklist["blocking"])
    return {"current_stage": "risk_review", "compliance_checklist": checklist}


class WordingCompliance(BaseModel):
    """合规复核的输出契约。

    【刻意不含任何数值字段】：这个节点只做「是/否」两个判断 + 一句说明。
    给它数值字段等于给模型一个可以「打分量」的位置，而打分量必然要人来解释。
    """
    rating_backed: bool = Field(..., description="评级是否有四维结论作为证据支撑")
    investment_advice_risk: bool = Field(..., description="措辞是否构成投资建议")
    note: str = Field(..., max_length=600, description="不超过 150 字的中文说明")


async def llm_compliance_review_node(state: RiskState) -> dict:
    """LLM 合规复核：把「措辞合规」作为 checklist 的第 6 项。

    **必须区分两件事**（tests/test_risk_hitl.py 里各有一条）：
      · LLM 服务故障（重试耗尽）⇒ 上抛 ⇒ 父图 mark_failed，任务 failed、不产出研报。
        依据是「风控失败必须阻断」—— 风控在 NON_DEGRADABLE_AGENTS 里，
        重试耗尽后在第二层之前就上抛（retry.py），本节点拿不到哨兵。
      · LLM 成功但判定措辞有风险 ⇒ 只写进 blocking 列表，**不自动阻断**。
        裁决权在人。

    把这两件事混为一谈，就会得到一个「模型说措辞不好就把任务毙掉」的系统 ——
    那是把裁决权交给了模型。

    with_retry 的挂载形态与检索/估值子图一致：节点内部的闭包，只包 LLM 调用，
    重试因此不重复写阶段事件。

    :param state: RiskState，读 tenant_id/task_id/company_code/rating/rating_note/
        compliance_checklist。
    :return: dict，含 current_stage="risk_review" 与更新版的 compliance_checklist。
    """
    tenant_id, task_id = state["tenant_id"], state["task_id"]

    dimensions = await repo.fetch_current_dimension_results(tenant_id, task_id)
    lines = []
    for dim, label in DIMENSION_LABELS.items():
        item = dimensions.get(dim)
        if item:
            lines.append(f"{label}（评分 {item.get('score')}）：{item.get('conclusion')}")
        else:
            lines.append(f"{label}：本次未产出结论")

    prompt = build_compliance_prompt(state.get("company_code", ""),
                                     state.get("rating") or "未评级",
                                     state.get("rating_note") or "", lines)
    structured = get_structured_llm("risk", WordingCompliance)

    @with_retry(agent_type="risk")
    async def _invoke():
        """带闭包 prompt 调合规复核 LLM；无参数。返回 WordingCompliance。"""
        return await structured.ainvoke(prompt)

    parsed = await _invoke()
    # 走到这里说明 LLM 成功了。拿不到合法结构就不是「成功」—— 上抛，
    # 与「服务故障」同一条处置路径：风控宁可阻断，也不要让一份没复核过的研报发出去。
    if not isinstance(parsed, WordingCompliance):
        note = (parsed.note if isinstance(parsed, FallbackResult)
                else "合规复核未产出可用结果")
        raise RuntimeError(f"合规复核失败：{note}")

    passed = bool(parsed.rating_backed) and not bool(parsed.investment_advice_risk)
    checklist = dict(state.get("compliance_checklist") or {"items": [], "blocking": []})
    items = list(checklist.get("items") or [])
    # 幂等：节点在恢复时可能重跑，不能把同一项追加两次
    items = [i for i in items if i.get("key") != "wording_compliance"]
    items.append({
        "key": "wording_compliance",
        "label": "措辞合规（LLM 复核）",
        "passed": passed,
        "note": parsed.note,
    })
    checklist["items"] = items
    checklist["blocking"] = [i["key"] for i in items if i["passed"] is False]

    await repo.record_stage_event(tenant_id, task_id, "risk_review", "skipped",
                                 {"phase": "llm_compliance_review",
                                  "passed": passed,
                                  "blocking": checklist["blocking"]})
    logger.info("risk.compliance_reviewed", task_id=task_id, passed=passed)
    return {"current_stage": "risk_review", "compliance_checklist": checklist}


async def persist_draft_node(state: RiskState) -> dict:
    """把待审草稿落进业务表，并把任务翻成「待风控审核」。

    顺序不可调换：先写草稿、再翻状态、最后才让闸门暂停。
    反过来（先翻状态再写草稿）会让前端在这两步之间轮询到一个「待审核但无内容」的
    空窗 —— 那正是审核人点进来刷新的时候。

    状态必须在闸门【之前】落库：暂停期间流程不再往前走，任何「等恢复后再写」的东西
    在这段时间里都是不存在的。

    :param state: RiskState，读 tenant_id/task_id/company_code/company_id/rating/
        has_reference/valuation_available/industry/rating_note/comparison_points/
        valuation/compliance_checklist。
    :return: dict，含 current_stage="risk_review" 与 draft_report_id（落库后的研报 id）。
    """
    tenant_id, task_id = state["tenant_id"], state["task_id"]

    dimensions = await repo.fetch_current_dimension_results(tenant_id, task_id)
    company = await repo.get_company_by_code(tenant_id, state["company_code"])
    company_name = company["name"] if company else None

    rating = state.get("rating") or "未评级"
    has_reference = bool(state.get("has_reference"))
    valuation_available = bool(state.get("valuation_available"))

    title = build_title(company_name, state["company_code"], rating)
    content = build_content(company_name, state["company_code"],
                            state.get("industry") or "", rating,
                            state.get("rating_note") or "", dimensions,
                            has_reference, valuation_available,
                            comparison_points=state.get("comparison_points") or [],
                            valuation=state.get("valuation"))
    disclosure = build_risk_disclosure(dimensions, has_reference, valuation_available)

    report_id = await repo.upsert_report(
        tenant_id=tenant_id, task_id=task_id, company_id=state["company_id"],
        title=title, content=content, rating=rating, status="draft",
        risk_disclosure=disclosure,
        comparison_points=state.get("comparison_points") or [],
    )
    review_id = await repo.insert_risk_review(
        tenant_id, task_id, report_id, state.get("compliance_checklist") or {},
    )
    await repo.update_task_status(tenant_id, task_id, status="awaiting_risk_review",
                                 current_stage="risk_review")

    logger.info("risk.draft_persisted", task_id=task_id,
                report_id=report_id, review_id=review_id)
    return {"current_stage": "risk_review", "draft_report_id": report_id}


async def human_review_gate_node(state: RiskState) -> dict:
    """人工签字闸门（哨兵版）。

    【不再暂停任何框架】。这里只产出「审核人要看到的小数据」放进 State：
    steps.run_risk_stage 在 persist_draft 之后调用它、拿到 payload 后把 checkpoint
    与哨兵一并返回给引擎 —— 停住在编排层发生，而不是在某个图节点里。

    为什么本节点不写库、只拼 payload：写库（预检落库、草稿落库、状态翻待审）全部
    在 precheck/llm_compliance/persist_draft 三个节点完成，保证「暂停态的全部信息
    都在业务表」，sign 时才可以从表里读全待审内容、不必读任何流程态。

    :param state: RiskState，读 task_id/draft_report_id/company_code/rating/redo_count/
        compliance_checklist。
    :return: {"review_payload": {...}}，payload 是审核人要看的核心小数据
        （type/task_id/report_id/company_code/rating/redo_count/checklist/instruction）。
    """
    payload = {
        "type": "risk_sign_off",
        "task_id": state["task_id"],
        "report_id": state.get("draft_report_id"),
        "company_code": state.get("company_code"),
        "rating": state.get("rating"),
        "redo_count": state.get("redo_count", 0),
        "checklist": (state.get("compliance_checklist") or {}).get("items", []),
        "instruction": "请签字：approve / modify（有条件通过）/ reject（驳回重做）",
    }
    return {"review_payload": payload}


async def apply_decision_node(state: RiskState) -> dict:
    """把恢复值写进业务表：签字落库、任务翻状态。

    载荷校验放在这里而不是 API 层：**图是唯一的入口**。API 的校验是给用户看的 400，
    这里的校验才是「跑不到就别写库」的保底 —— 直接 ainvoke 图（测试、脚本、日后的
    批量重放）时 API 那层根本不在。校验不通过就抛，任务落到 mark_failed 那一侧。

    :param state: RiskState，读 tenant_id/task_id/review_payload（在 steps.run_risk_stage
        里由 resume_payload 注入）。
    :return: dict，含 risk_decision/risk_comments/current_stage="risk_review"；
        驳回时还带 redo_targets/redo_count，放行时带 reviewer 已确认。
    """
    tenant_id, task_id = state["tenant_id"], state["task_id"]
    payload = state.get("review_payload") or {}

    decision = payload.get("decision")
    comments = (payload.get("comments") or "").strip()
    reviewer_id = payload.get("reviewer_id")
    redo_targets: Optional[dict] = payload.get("redo_targets") or None

    if decision not in VALID_DECISIONS:
        raise ValueError(f"签字决策非法：{decision!r}")
    if not reviewer_id:
        raise ValueError("签字必须携带签字人")
    if len(comments) < MIN_COMMENTS_LEN:
        raise ValueError(f"签字意见不得短于 {MIN_COMMENTS_LEN} 字")
    if decision == "reject":
        stage = (redo_targets or {}).get("stage")
        if stage not in REDO_STAGES:
            raise ValueError(f"驳回必须指定重做阶段（{'/'.join(REDO_STAGES)}），实际 {stage!r}")

    review = await repo.get_latest_risk_review(tenant_id, task_id)
    if not review:
        raise RuntimeError("找不到待签字的审核记录，状态已损坏")
    await repo.sign_risk_review(tenant_id, str(review["id"]), reviewer_id,
                                decision, comments, redo_targets)

    if decision == "reject":
        # 计数 +1 与「转出待审」是一次原子写（见 repo.mark_redo_started 的注释）：
        # 状态必须退回 running，否则前端一直显示「待风控审核」、审核人以为还有一条等他签。
        # 计数再回填父图：父图的条件边读 redo_count 判上限。
        new_count = await repo.mark_redo_started(tenant_id, task_id)
        await repo.record_stage_event(tenant_id, task_id, "risk_review", "success",
                                     {"decision": decision, "redo_targets": redo_targets,
                                      "redo_count": new_count})
        logger.warning("risk.rejected", task_id=task_id, redo_count=new_count,
                       redo_targets=redo_targets)
        return {"risk_decision": "reject", "risk_comments": comments,
                "redo_targets": redo_targets, "redo_count": new_count,
                "current_stage": "risk_review"}

    # approve 与 modify 都放行（见本文件顶部对 modify 的定义）。
    # 这里写 status='approved' 而不是直接 published：发布是 publish_report 的事，
    # 风控只负责「签过了」。状态被拆成两格，图上的两个节点才各有一件事可做。
    await repo.update_task_status(tenant_id, task_id, status="approved",
                                 current_stage="risk_review")
    await repo.record_stage_event(tenant_id, task_id, "risk_review", "success",
                                 {"decision": decision, "reviewer_id": reviewer_id})
    logger.info("risk.signed", task_id=task_id, decision=decision)
    return {"risk_decision": decision, "risk_comments": comments,
            "current_stage": "risk_review"}
