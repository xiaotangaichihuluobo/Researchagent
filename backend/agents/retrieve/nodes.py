# backend/agents/retrieve/nodes.py
# 检索子图：拼查询 → 双方向检索 → 精排 → LLM 提炼对比点。
#
# 【两处 with_retry 挂载点】，都在节点内部的闭包里、只包无副作用的操作：
#   ① hybrid_retrieve 内包 search_reports（嵌入 + Milvus 查询 + 精排，全是只读）
#   ② extract_comparison 内包 LLM 调用
#
# 为什么不合并成「包整个节点体」：record_stage_event 是 append-only 的 INSERT
# （research_repo.py:186-201），包节点体会让每次重试都重复写一条审计流水。
# 为什么不装饰节点函数：with_retry 的返回值顶替被装饰函数的返回值，
# 而 FallbackResult 不是 dict，交给 LangGraph 会被拒绝合并（retry.py:52-66 的 docstring）。

from typing import Optional

from pydantic import BaseModel, Field

from backend.agents.retrieve.prompts import build_extract_prompt
from backend.agents.retrieve.state import RetrieveState
from backend.config import get_settings
# 研报检索沿用既有 reranker.py 里的检索流水（BGEModel / KnowledgeBaseClient /
# BGEReranker 的一套），不另起一份实现。取模块属性访问：conftest 的 autouse
# 假检索要能 monkeypatch reranker.search_reports 顶掉它 —— 直接 from-import 会
# 把绑定留在本模块，patch 就不生效了。
from backend.core import reranker as retriever
from backend.core import research_repo as repo
from backend.core.llm_factory import get_structured_llm
from backend.core.logger import get_logger
from backend.core.retry import FallbackResult, with_retry

logger = get_logger(__name__)


class ComparisonExtraction(BaseModel):
    """LLM 的输出契约。

    【刻意不含任何数值字段】—— 与 ValuationDecision 同一条理由：数字不经过模型。
    引用历史研报里的数字时，数字来自检索到的原文片段，不来自模型的计算。
    """
    points: list[str] = Field(default_factory=list,
                             description="2-4 条可比对要点，每条不超过 120 字")


async def prepare_query_node(state: RetrieveState) -> dict:
    """拼查询文本，并把两个方向所需的参数准备出来。

    查询文本取自四维结论的摘要 + 标的：检索要回答的是「本次结论相对历史有无变化」，
    而不是泛泛地找这家公司的资料。

    :param state: RetrieveState，读 tenant_id/task_id/company_code/industry。
    :return: dict，含 current_stage="retrieve" 与 _query（拼接的查询文本，截到 600 字）。
    """
    tenant_id, task_id = state["tenant_id"], state["task_id"]
    await repo.update_task_status(tenant_id, task_id, current_stage="retrieve")
    await repo.record_stage_event(tenant_id, task_id, "retrieve", "started")

    dimensions = await repo.fetch_current_dimension_results(tenant_id, task_id)
    parts = [f"{state.get('company_code', '')} {state.get('industry') or ''}".strip()]
    for dim, item in sorted((dimensions or {}).items()):
        conclusion = (item or {}).get("conclusion")
        if conclusion:
            parts.append(f"{dim}：{conclusion}")
    query = " ".join(parts)[:600]

    logger.info("retrieve.query_prepared", task_id=task_id, query_chars=len(query))
    return {"current_stage": "retrieve", "_query": query}


async def hybrid_retrieve_node(state: RetrieveState) -> dict:
    """双方向检索 + 精排。这是第一处 with_retry 挂载点。

    降级（拿不到候选）与「检索到但 LLM 提炼失败」在留痕上必须能区分：前者是
    语料/服务问题，后者是模型问题，排查方向完全不同。

    :param state: RetrieveState，读 tenant_id/task_id/_query/company_code/industry。
    :return: dict，含 retrieved_chunks、search_failed、_search_note。检索失败时
        retrieved_chunks 为空列表、search_failed=True。
    """
    tenant_id, task_id = state["tenant_id"], state["task_id"]
    settings = get_settings()

    @with_retry(agent_type="retrieve")
    async def _invoke():
        """带闭包 query 等参数调 search_reports；无参数。返回候档文档列表。"""
        return await retriever.search_reports(
            query=state.get("_query") or state.get("company_code", ""),
            tenant_id=tenant_id,
            company_code=state.get("company_code", ""),
            industry=state.get("industry") or "",
            recall_top_k=settings.report_recall_top_k,
            rerank_top_k=settings.report_rerank_top_k,
        )

    try:
        docs = await _invoke()
    except Exception as e:                       # noqa: BLE001 —— 降级也得留一条记录
        logger.error("retrieve.search_failed", task_id=task_id, error=str(e))
        docs = None

    if not isinstance(docs, list):
        # 哨兵（FallbackResult）或异常 —— 两者都表示「本次拿不到候选」
        note = (docs.note if isinstance(docs, FallbackResult)
                else "研报检索服务暂时不可用")
        await repo.record_stage_event(tenant_id, task_id, "retrieve", "failed",
                                     {"phase": "search", "reason": note})
        return {"retrieved_chunks": [], "search_failed": True,
                "_search_note": note}

    if not docs:
        await repo.record_stage_event(tenant_id, task_id, "retrieve", "failed",
                                     {"phase": "search", "reason": "语料库中没有匹配的研报"})
        return {"retrieved_chunks": [], "search_failed": True,
                "_search_note": "语料库中没有匹配的研报"}

    return {"retrieved_chunks": docs, "search_failed": False, "_search_note": None}


async def rerank_node(state: RetrieveState) -> dict:
    """把候选正文截短，只收进 State 该收的长度。

    条数上限已由 search_reports 内部的方向配额（apply_direction_quota）收口，
    这里只截 content：子图字段仍会被 checkpoint，正文不截短，量大一样会撑。

    :param state: RetrieveState，读 search_failed/retrieved_chunks。
    :return: dict，含 retrieved_chunks（正文已按 settings.report_chunk_chars 截短）；
        检索失败时返回空 dict。
    """
    if state.get("search_failed"):
        return {}

    settings = get_settings()
    retrieved = state.get("retrieved_chunks") or []   # 没检索到/缺省 → 空列表，别让切片炸
    chunks = []
    for doc in retrieved:                             # search_reports 已按 top_k 收口，不必再限条数
        item = dict(doc)                              # 拷一份原 doc
        item["content"] = doc["content"][:settings.report_chunk_chars]  # 正文截到最长期望长度
        chunks.append(item)

    return {"retrieved_chunks": chunks}


async def extract_comparison_node(state: RetrieveState) -> dict:
    """LLM 提炼对比点。这是第二处 with_retry 挂载点，也是本子图唯一的 LLM 调用。

    :param state: RetrieveState，读 tenant_id/task_id/retrieved_chunks/search_failed/
        _search_note/company_code/industry。
    :return: dict 含 current_stage="retrieve"、has_reference、comparison_points、
        retrieved_chunks。检索/提炼失败时 has_reference=False、points 为空。
    """
    tenant_id, task_id = state["tenant_id"], state["task_id"]
    chunks = state.get("retrieved_chunks") or []

    if state.get("search_failed") or not chunks:
        reason = state.get("_search_note") or "本次未检索到可对比的历史研报"
        await repo.record_stage_event(tenant_id, task_id, "retrieve", "skipped",
                                     {"reason": reason})
        return {"current_stage": "retrieve", "has_reference": False,
                "comparison_points": [], "retrieved_chunks": []}

    prompt = build_extract_prompt(state.get("company_code", ""),
                                 state.get("industry") or "", chunks)
    structured = get_structured_llm("retrieve", ComparisonExtraction)

    @with_retry(agent_type="retrieve")
    async def _invoke():
        """带闭包 prompt 调提炼 LLM；无参数。返回 ComparisonExtraction。"""
        return await structured.ainvoke(prompt)

    try:
        parsed = await _invoke()
    except Exception as e:                       # noqa: BLE001
        logger.error("retrieve.extract_failed", task_id=task_id, error=str(e))
        parsed = None

    if not isinstance(parsed, ComparisonExtraction) or not parsed.points:
        note = (parsed.note if isinstance(parsed, FallbackResult)
                else "对比点提炼未产出可用结果")
        await repo.record_stage_event(tenant_id, task_id, "retrieve", "failed",
                                     {"phase": "extract", "reason": note})
        # 检索到了片段但提炼失败：仍然标注「无历史参照」—— 没有可读的对比点，
        # 对下游与读者来说就等于没有参照。但留痕区分了是哪一段出的问题。
        return {"current_stage": "retrieve", "has_reference": False,
                "comparison_points": [], "retrieved_chunks": []}

    points = [p.strip() for p in parsed.points if p.strip()][:4]
    await repo.record_stage_event(tenant_id, task_id, "retrieve", "success",
                                 {"chunks": len(chunks), "points": len(points)})
    logger.info("retrieve.reference_ready", task_id=task_id, points=len(points))
    return {"current_stage": "retrieve", "has_reference": True,
            "comparison_points": points, "retrieved_chunks": chunks}