# backend/agents/research/steps.py
# 阶段段的执行器：把每个子图的节点函数「显式串成一段」。
#
# 重构前这些由 LangGraph 的 add_node/add_edge/conditional 表达，节点之间的连接是图上
# 的边、并发是图引擎的 superstep 语义、合并是 reducer。重构后这里就是一段纯 Python 步骤
# 编排：线性的先后 await，并发用 asyncio.gather 显式写，合并手写并回填 state。
#
# 刻意只把这些塞进一个函数而非一堆 add_node 的动机：可以让引擎（runner.py）用一个极薄的
# 表驱动 while 循环推进 —— 段内如何排布是这里的细节，段间如何走是引擎的分工。

import asyncio

from backend.agents.analyze.nodes import (
    aggregate_rating_node,
    make_dimension_node,
    prepare_evidence_node,
)
from backend.agents.collect.nodes import (
    fetch_all_sources_node,
    normalize_and_score_node,
    persist_collected_node,
    resolve_company_node,
)
from backend.agents.retrieve.nodes import (
    extract_comparison_node,
    hybrid_retrieve_node,
    prepare_query_node,
    rerank_node,
)
from backend.agents.risk.nodes import (
    apply_decision_node,
    human_review_gate_node,
    llm_compliance_review_node,
    persist_draft_node,
    precheck_compliance_node,
)
from backend.agents.valuation.nodes import (
    extract_facts_node,
    prepare_inputs_node,
    run_comparable_node,
    run_dcf_node,
    synthesize_node,
)
from backend.core import research_repo as repo
from backend.core.research_rules import DIMENSIONS
from backend.core.logger import get_logger

logger = get_logger(__name__)

# 检索段里子图自用的中间大对象/内部键 —— 段内用它会再 pop 掉，不让父 State 背着它们。
_RETRIEVE_INTERNAL_KEYS = ("_query", "retrieved_chunks", "search_failed", "_search_note")


async def run_stage(stage: str, state: dict) -> dict:
    """跑一个线性段的执行器（collect/analyze/retrieve/valuation）。

    risk 不在这里：它有「停或续」两种语义，由引擎直接调 run_risk_stage。
    返回「跑完该段后的完整 state」。

    :param stage: 线性段名（collect/analyze/retrieve/valuation）；未知值会抛 ValueError。
    :param state: 进入该段前的父 State。
    :return: 该段跑完后更新过的完整 state（dict）。
    """
    if stage == "collect":
        return await collect_stage(state)
    if stage == "analyze":
        return await analyze_stage(state)
    if stage == "retrieve":
        return await retrieve_stage(state)
    if stage == "valuation":
        return await valuation_stage(state)
    raise ValueError(f"无该线性段的执行器：{stage!r}")


async def collect_stage(state: dict) -> dict:
    """collect 段：定位标的 → 拉四路数据 → 归一打分 → 落库。原 collect 子图主链路。

    :param state: 进入 collect 前的父 State（含 company_id/company_code 等）。
    :return: collect 跑完后的完整 state，新增 collected_ids/source_stats 等键。
    """
    state = dict(state)
    state.update(await resolve_company_node(state))
    state.update(await fetch_all_sources_node(state))
    state.update(await normalize_and_score_node(state))
    state.update(await persist_collected_node(state))
    return state


async def analyze_stage(state: dict) -> dict:
    """分析段：准备证据 → 四维并发打分析 → 汇总评级。

    【不用 return_exceptions=True】：采集段的四路失败允许部分交付（失败记录进
    errors、能用的继续用），但分析的四个维度是上下游消费的完整输入 —— 一个维度失败
    就该整体失败，绝不能把「缺了一个维度的四维结论」当作完整末态交给下游。
    这跟 collect 内部那处 gather 的取舍正好相反，是刻意为之。

    :param state: 进入 analyze 前的父 State（含 collected_ids 等）。
    :return: analyze 跑完后的完整 state，新增 dimension_results/rating/rating_note/
        has_buy_sell_advice 等键。
    """
    state = dict(state)
    state.update(await prepare_evidence_node(state))

    # 四维并发打分。把一行压缩的生成器拆开，每步一个概念：
    #   1) dimension_nodes：为 DIMENSIONS 里每个维度造一个节点函数
    #   2) tasks：每个节点各带上 state 调一次 → 得到四个待执行的协程
    #   3) asyncio.gather(*tasks)：并发跑完，结果按 DIMENSIONS 顺序收进 outputs
    dimension_nodes = [make_dimension_node(d) for d in DIMENSIONS]
    tasks = [node(state) for node in dimension_nodes]
    outputs = await asyncio.gather(*tasks)

    # 每个节点返回 {"dimension_results": {<维度>: …}}（没参与重做的返回 {}）。
    # 维度 key 互不冲突，各自 update 进同一张表即可；{} 的节点不并进任何东西。
    merged: dict = {}
    for out in outputs:
        parts = (out or {}).get("dimension_results") or {}
        merged.update(parts)
    state["dimension_results"] = merged

    state.update(await aggregate_rating_node(state))
    return state


async def retrieve_stage(state: dict) -> dict:
    """检索段：拼查询 → 混合检索 → 重排 → 提炼可比对要点。原 retrieve 子图主链路。

    段末 pop 掉子图自用的中间大对象（RAG 片段等），父 State 只保留
    has_reference / comparison_points 两个下游要读的产物 —— 「大对象不进父 State」
    的分工在这里显式收编，不再依赖 LangGraph 子图合并时的静默丢弃。

    :param state: 进入 retrieve 前的父 State（含 dimension_results 等）。
    :return: retrieve 跑完后的完整 state，新增 has_reference/comparison_points 键，
        并已 pop 掉段内中间大对象（_query/retrieved_chunks 等）。
    """
    state = dict(state)
    state.update(await prepare_query_node(state))
    state.update(await hybrid_retrieve_node(state))
    state.update(await rerank_node(state))
    state.update(await extract_comparison_node(state))
    for k in _RETRIEVE_INTERNAL_KEYS:
        state.pop(k, None)
    return state


async def valuation_stage(state: dict) -> dict:
    """估值段：提数 → 装配输入 → DCF 与可比法并发 → 合成。原 valuation 子图主链路。

    extract_facts_node 先跑：LLM 提数，写 financial_facts。prepare_inputs 只读产物，
    空则短路（inputs_ok=False）。两法任一失败都不该拖垮另一法（估值本就是「扣分制、
    可降级」的段）：这里手动用 asyncio.gather 同时起两路，各节点内部对自己的不可用
    负责（inputs_ok 判定短路 / 异常降级），与 collect 的容错取向一致。

    :param state: 进入 valuation 前的父 State（含 company_code/industry 等）。
    :return: valuation 跑完后的完整 state，新增 valuation/valuation_available/
        dcf_result/comparable_result 等键。
    """
    state = dict(state)
    state.update(await extract_facts_node(state))
    state.update(await prepare_inputs_node(state))

    dcf_res, comp_res = await asyncio.gather(
        run_dcf_node(state), run_comparable_node(state),
    )
    state["dcf_result"] = dcf_res.get("dcf_result")
    state["comparable_result"] = comp_res.get("comparable_result")

    state.update(await synthesize_node(state))
    return state


async def run_risk_stage(state: dict, resume_payload: dict | None = None) -> dict:
    """风控段：预检 → 合规复核 → 落草稿，然后【停或续】。

    state 里没有 resume_payload 时是首跑：落草稿 + 写暂停点 checkpoint + 挂哨兵，
    引擎据此收口为「暂停」，绝不发布。
    给了 resume_payload 则是续跑：把签字注入 state，前进式续跑 apply_decision_node
    （【不重跑闸门前的任何节点】），产出 risk_decision 供引擎路由。

    停/续的分界在第一步：声称要 resume、却没有任何载荷，就把「半途而废」当成故障
    （宁可因缺少载荷而失败，也不要凭空把一份没签过字的研报放行）。

    :param state: 进入 risk 前的父 State；首跑由驱动带入，续跑由 _rebuild_state 重建。
    :param resume_payload: 续跑时的签字载荷 dict；为 None（默认）表示首跑 —— 首跑
        会落草稿 + 写 checkpoint + 返回暂停态，续跑则只应用决策、清暂停点。
    :return: risk 段后的 state。首跑返回带 paused=True 与 draft_report_id 的暂停态，
        续跑返回带 risk_decision/risk_comments 的 state。
    """
    state = dict(state)
    state["review_payload"] = resume_payload

    if resume_payload is None:
        # ── 首跑：预检 → 合规复核 → 落草稿 → 停 ────────────────
        # 这三个节点是「停之前」的产物，只在首跑这里跑。审核人签的就是
        # persist_draft 之后留在表里的那份草稿。
        state.update(await precheck_compliance_node(state))
        state.update(await llm_compliance_review_node(state))
        state.update(await persist_draft_node(state))

        # 暂停点写进业务表（草稿本身已在 persist_draft 落库）。
        # checkpoint 只记「停在哪、等什么」，不背正文 —— 正文都在表里。
        await repo.upsert_checkpoint(
            tenant_id=state["tenant_id"], task_id=state["task_id"],
            stage="risk", step="pending_decision",
            payload={"report_id": state.get("draft_report_id")},
        )
        # 哨兵版闸门把「审核人要看的核心小数据」挂进 state，供审阅/留痕；
        # 真正「停住等人」发生在引擎看到 state["paused"] 之后。
        state.update(await human_review_gate_node(state))
        state["paused"] = True
        return state

    # ── 续跑：注入签字，前进式续跑 ─────────────────────────────
    # 【不重跑闸门前的任何节点】：precheck/compliance/persist 是停之前的产物，
    # compliance_checklist 等消费键已由 runner._rebuild_state 从业务表重建 /
    # 或由上一轮首跑留在内存里。这里只把签字交给 apply_decision，清掉暂停点收口。
    state.update(await apply_decision_node(state))
    await repo.clear_checkpoint(state["tenant_id"], state["task_id"])
    state.pop("paused", None)
    return state


# ── 终态收口（原 graph.py 的三个终态节点）────────────────────────
# 它们不是「段」，不会被 run_stage 表驱动；由引擎在读到风控裁决后按路由分支调用一次，
# 收口成终态（published / rejected / failed）后任务即终结。

async def publish_report_final(state: dict) -> dict:
    """发布：把**已签字的草稿**翻成 published，并把任务标为终态。

    正文不在这里拼装：风控的 persist_draft 在人工闸门【之前】就把它写好了
    （审核人要看的就是那一份）。这里只翻状态，保证「审核的就是发布的」。

    :param state: 已签字的父 State，读 tenant_id/task_id/company_code/industry 等键。
    :return: {"current_stage": "publish"}，并把草稿翻成 published、任务标终态。
    """
    tenant_id, task_id = state["tenant_id"], state["task_id"]

    report = await repo.get_report(tenant_id, task_id)
    if report is None:
        raise RuntimeError("发布阶段找不到待发布的研报草稿，状态已损坏")

    await repo.upsert_report(
        tenant_id=tenant_id, task_id=task_id, company_id=str(report["company_id"]),
        title=report["title"], content=report["content"], rating=report["rating"],
        status="published", risk_disclosure=report["risk_disclosure"],
        # 必须把草稿的 comparison_points 原样带过去：upsert_report 是整行覆盖，
        # 漏掉这个参数会把检索子图扒出的可比对要点写成空数组。
        comparison_points=report["comparison_points"],
    )
    await repo.update_task_status(
        tenant_id=tenant_id, task_id=task_id, status="published",
        current_stage="publish", mark_finished=True,
    )

    # ── 语料回灌（绝不抛出）────────────────────────────────────
    # 一份已发布的研报若因回灌失败把任务标成 failed，就是让一个内部索引问题
    # 否掉一个已生效的对外产物。失败只记录。宿主是既有 RAG 基建 report_ingest。
    corpus_ingested, corpus_reason = False, ""
    try:
        from datetime import datetime, timezone

        from backend.core.report_ingest import ingest_report

        chunks = await ingest_report(
            tenant_id=tenant_id,
            report_key=f"published-{report['id']}",
            title=report["title"], content=report["content"],
            company_code=state.get("company_code") or "",
            industry=state.get("industry") or "",
            report_type="internal",
            published_at=datetime.now(timezone.utc),
            company_id=str(report["company_id"]),
        )
        corpus_ingested, corpus_reason = True, f"已回灌 {chunks} 个切片"
    except Exception as e:                       # noqa: BLE001 —— 回灌失败绝不阻断发布
        corpus_reason = str(e)
        logger.warning("publish.corpus_ingest_failed", task_id=task_id, error=corpus_reason)

    await repo.record_stage_event(tenant_id, task_id, "publish", "success",
                                  {"report_id": str(report["id"]),
                                   "risk_decision": state.get("risk_decision"),
                                   "corpus_ingested": corpus_ingested,
                                   "corpus_reason": corpus_reason})
    return {"current_stage": "publish"}


async def mark_rejected_final(state: dict) -> dict:
    """驳回超限 → 转人工。任务不以 failed 结束：它需要人来看，而不是被丢掉。

    :param state: 父 State，读 tenant_id/task_id/redo_count。
    :return: {"current_stage": "risk_review"}，并把任务标为 rejected 终态。
    """
    await repo.update_task_status(
        tenant_id=state["tenant_id"], task_id=state["task_id"],
        status="rejected", current_stage="risk_review", mark_finished=True,
        last_error=f"驳回次数达到上限（{state.get('redo_count')}），已转人工处理",
    )
    await repo.record_stage_event(state["tenant_id"], state["task_id"], "risk_review", "failed",
                                 {"reason": "驳回次数达上限，已转人工处理"})
    return {"current_stage": "risk_review"}


async def mark_failed_final(state: dict) -> dict:
    """阻断：风控决策缺失或状态损坏。

    风控失败必须阻断 —— status='failed'，宁可不发布。其他四个阶段失败都可以降级交付，
    唯独风控不行。这是「监管强制」在容错设计上的直接体现。

    :param state: 父 State，读 tenant_id/task_id。
    :return: {"current_stage": "risk_review"}，并把任务标为 failed 终态。
    """
    await repo.update_task_status(
        tenant_id=state["tenant_id"], task_id=state["task_id"],
        status="failed", current_stage="risk_review", mark_finished=True,
        last_error="风控决策缺失或不可识别，任务已阻断",
    )
    await repo.record_stage_event(state["tenant_id"], state["task_id"], "risk_review", "failed",
                                 {"reason": "风控决策缺失或不可识别，已阻断"})
    return {"current_stage": "risk_review"}