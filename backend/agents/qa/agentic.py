# backend/agents/qa/agentic.py
# 轨道B 的可选「多轮自搜」问答：agentic 完整 agent loop（config 开 qa_engine_mode=agentic 才走）。
#
# 和 service.run_qa 那套确定性管线的关系：本文件只是一个「循环」，套在现有检索/生成函数外面。
#   每一轮把 原问题 + 历史摘要 + 已搜到的材料 摊给模型，让它表达两个动作之一：
#     search  → 带上 1~2 个更聚焦的新关键词，再检索一轮，材料去重后累计
#     answer  → 现有材料够了，答
#   loop：说 answer 就收，说 search 就回到第 1 步；轮数上限 qa_agent_max_steps，防止死循环。
#
# 铁规矩：最终答案里的数字必须来自累计证据（研报原文/联网原文），模型只能「引用」不能「编」。
# 实现上：终局把 累计证据 全量摊进 AGENT_ANSWER_PROMPT，并命令只许用材料里的数。

import asyncio
from pydantic import BaseModel

from backend.agents.qa import nodes, prompts
from backend.config import get_settings
from backend.core.llm_factory import get_structured_llm
from backend.core.logger import get_logger

logger = get_logger(__name__)


class AgentStepDecision(BaseModel):
    """每轮模型的决断：search（再搜，带多个独立子查询）或 answer（够了，答）。"""
    action: str             # "search" / "answer"
    queries: list[str] = []  # action=search 时的独立子查询（覆盖缺失的几个侧面），可直接检索
    reason: str = ""         # 一句话说明为什么这么决断


# ── 检索工具：真走 MCP 调 search_knowledge_base，复用 nodes._search_kb ──
async def _search_report(state: dict, query: str) -> list[dict]:
    """按 query 搜已发布研报语料（真走 MCP 协议）。返回规整后的 doc 列表。

    :param state: 问答状态 dict，须含 tenant_id。
    :param query: 检索查询串（str）。
    :return: list[dict]，规整成 {content, score, metadata} 的研报 doc 列表。
    """
    return await nodes._search_kb(
        query=query, tenant_id=state["tenant_id"], recall=8, rerank=3)


async def _search_report_safe(state: dict, query: str) -> list[dict]:
    """搜一条子查询；失败记日志、返回空列表，不让单个查询打断整轮并行。

    :param state: 问答状态 dict，须含 tenant_id。
    :param query: 检索查询串（str）。
    :return: list[dict]，研报 doc 列表；失败返回空列表。
    """
    try:
        return await _search_report(state, query)
    except Exception as e:                          # noqa: BLE001 —— 单个查询失败按没搜到处理
        logger.warning("agentic.search_failed", query=query[:50], error=str(e)[:200])
        return []


async def _search_web(state: dict, query: str) -> list[dict]:
    """联网搜索工具（enable_web_search 才给用），真走 MCP 协议；归一化成研报 doc 同形状。

    :param state: 问答状态 dict（仅随 nodes.mcp_web_search 调用传入，未直接使用字段）。
    :param query: 搜索关键词（str）。
    :return: list[dict]，归一化成 {"content","score","metadata"} 形状的 doc 列表。
    """
    results = await nodes.mcp_web_search(query, 5)   # 失败已重试，仍不可用 → 空
    return [
        {"content": (r.get("content") or r.get("snippet") or "")[:600],
         "score": 1.0,
         "metadata": {"source_name": "联网", "url": r.get("url", "")}}
        for r in (results or [])
    ]


def _accumulate(evidence: list[dict], new_docs: list[dict], limit: int) -> None:
    """把新检回的材料并进累计证据：按正文前 100 字去重，超出上限不追加。

    :param evidence: 累计证据列表（原位修改，按 content 前 100 字去重）。
    :param new_docs: 本轮新检回的材料 doc 列表。
    :param limit: 证据条数上限（达到即停止追加）。
    :return: 无返回值（通过原位修改 evidence 回写）。
    """
    seen = {d["content"][:100] for d in evidence}
    for d in new_docs:
        key = d["content"][:100]
        if key not in seen:
            seen.add(key)
            evidence.append(d)
            if len(evidence) >= limit:
                break


async def _decide(step_prompt: str) -> AgentStepDecision:
    """让模型对这轮做决断。决策输出坏了就默认 answer（宁可收尾，不瞎搜浪费）。

    :param step_prompt: 本轮决策提示词（str）。
    :return: AgentStepDecision；解析失败或动作非法默认 action="answer"；search 时 queries 截断至 MAX_BROAD_QUERIES 条。
    """
    llm = get_structured_llm("qa_reports", AgentStepDecision, temperature=0)
    try:
        parsed = await llm.ainvoke(step_prompt)
    except Exception as e:
        logger.warning("agentic.decide_failed", error=str(e))
        parsed = None
    if parsed is None or parsed.action not in ("search", "answer"):
        return AgentStepDecision(action="answer")
    if parsed.action == "search":
        # 缺口的多个侧面至多拆成 MAX_BROAD_QUERIES 个子查询：去空、截断。
        parsed.queries = [q for q in (parsed.queries or []) if q.strip()][:nodes.MAX_BROAD_QUERIES]
    return parsed


# ── 主循环 ─────────────────────────────────────────────────────
async def run_agentic(state: dict, streamer=None) -> None:
    """跑完整 agent loop，直接改 state（answer/sources/answer_mode/confidence）。

    - 先按规则判定是否闲聊 G：闲聊不进循环，直接 general 答。
    - 否则进入 for 循环搜索/决断，直到 answer 或达到轮数上限。
    - 完全没搜到材料 → 落 direct（和现管线低置信兜底一致）。

    :param state: 问答状态 dict，读 messages/tenant_id/enable_web_search/user_profile_text，并就地写入 answer/sources/answer_mode/confidence。
    :param streamer: 可选 SSE 流式回调（None 为一次性）。
    :return: 无返回值（通过修改 state 就地回写结果）。
    """
    settings = get_settings()
    max_steps = settings.qa_agent_max_steps
    limit = settings.qa_agent_result_limit

    query, auto_web = nodes._extract_query_and_web_flag(
        nodes._last_user_query(state.get("messages", [])))
    state.setdefault("original_query", query)
    # 联网开关：请求参数 或 提问里带"联网搜"字样，任一命中都给工具。
    enable_web = bool(state.get("enable_web_search")) or auto_web

    # 闲聊 G：先按规则信号判；规则没拦住再用 L2 MiniLM 兜底（对齐 pipeline 的
    # classify_query_node），避免"2222"这类乱码/无意义输入被当投研问题进检索。
    if query.strip().lower() in nodes._GENERAL_EXACT or \
       any(kw in query for kw in nodes._GENERAL_KEYWORDS):
        state.update(await nodes.generate_general_node(state, streamer))
        return
    from backend.core.query_classifier import QueryClassifier
    label, _ = await asyncio.to_thread(QueryClassifier.get_instance().classify, query)
    if label == "general":
        state.update(await nodes.generate_general_node(state, streamer))
        return

    evidence: list[dict] = []          # 累计证据（去重后的研报/联网原文）
    summary = nodes._layered_memory(state)
    profile = state.get("user_profile_text")     # 跨会话画像，并入 agentic 决策/终局 prompt
    history = prompts.format_history_for_prompt(state.get("messages", [])[:-1])

    for _ in range(max_steps):
        evid_text, _ = prompts.build_context_and_sources(evidence)
        step_prompt = prompts.build_agent_loop_prompt(
            query=query, history=history, summary=summary, evidence=evid_text,
            profile=profile)
        dec = await _decide(step_prompt)

        if dec.action == "answer" or len(evidence) >= limit:
            break

        # search 这轮：把缺口拆成多个子查询，并行检索后合并去重。
        # 先把要搜的子查询列出来：去掉空的，若一个都不剩就用原问题兜底。
        sub_queries = []
        for q in (dec.queries or [query]):
            if q.strip():
                sub_queries.append(q)
        if not sub_queries:
            sub_queries = [query]

        # 每个子查询各自容错（搜不到返回空）；并行发出去，结果逐个去重累计。
        tasks = [_search_report_safe(state, q) for q in sub_queries]
        _result_lists = await asyncio.gather(*tasks)
        for docs in _result_lists:
            _accumulate(evidence, docs, limit)
        if enable_web:
            _accumulate(evidence, await _search_web(state, query), limit)

    # 一个都没搜到 → 落 direct（与现管线低置信一致），不硬编答案。
    if not evidence:
        state.update(await nodes.generate_direct_node(state, streamer))
        return

    # 终局：用累计证据出答案（数字只许来自材料）。
    context, sources = prompts.build_context_and_sources(evidence)
    answer_prompt = prompts.build_agent_answer_prompt(
        query=query, history=history, summary=summary, context=context,
        profile=profile)
    answer = await nodes._generate(streamer, 0.3, answer_prompt)
    # 同 RAG/联网：来源不内嵌正文，由前端经结构化 sources 在气泡外渲染。
    state.update({
        "answer": answer, "sources": sources,
        "answer_mode": "agentic",
        "confidence": 0.9,     # 走了多轮检索的才到这，视为高置信；具体分不评估
    })