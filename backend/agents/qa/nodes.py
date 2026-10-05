# backend/agents/qa/nodes.py
# 轨道 B「问已发布研报」的节点实现——全部是【普通 async 函数】，由 service.run_qa 用
# if/elif 顺序编排，取代参考工程的 LangGraph 图。每个节点入参一个 dict state、
# 返回对该 state 的部分更新（与参考节点返回值语义一致）。
#
# 消息约定：state["messages"] 是 [{role: "user"|"assistant", content: str}]，历史从
# qa_messages 表按 thread 加载（见 qa_repo.get_messages），不是 langchain 对象。

import asyncio
import re

from backend.agents.qa import prompts
from backend.config import get_settings
from backend.core import memory, qa_repo
from backend.core.llm_factory import get_llm
from backend.core.logger import get_logger
from backend.core.query_classifier import QueryClassifier
from backend.core.retry import FallbackResult, with_retry
from backend.mcp.client import call_mcp_tool

logger = get_logger(__name__)

# ── 检索相关常量 ───────────────────────────────────────────────
MAX_BROAD_QUERIES       = 3   # BROAD 最多并行子 Query
RECALL_TOP_K_PRECISE    = 8
RECALL_TOP_K_VAGUE      = 10
RECALL_TOP_K_BROAD_PER  = 4
RERANK_TOP_K            = 3
HIGH_CONFIDENCE_BAR     = 0.75


# ── 消息辅助 ───────────────────────────────────────────────────
def _last_user_query(messages: list[dict]) -> str:
    """取消息列表里最近一条用户提问的文本。

    :param messages: 消息列表，元素形如 {"role": str, "content": str}。
    :return: str，最近一条 role==user 的 content 文本；无则返回空串。
    """
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


# ── 联网指令识别（第一靶保留字段、不接线 web 基建）──────────────
_WEB_HINTS = ("联网", "上网查", "上网搜", "网上搜", "搜索一下", "帮我搜", "百度一下", "谷歌一下")


def _extract_query_and_web_flag(raw: str) -> tuple[str, bool]:
    """识别是否带联网指令，并剥离检索动作保留干净的查询文本。

    :param raw: 原始用户输入文本（str）。
    :return: (clean, needs_web) 二元组：clean 为去掉联网语汇后的查询文本；needs_web 表示原文是否含联网搜信号。
    """
    needs_web = any(h in raw for h in _WEB_HINTS)
    if not needs_web:
        return raw, False
    clean = re.sub(
        r"[，,。.？?！!\s]*(?:如果不知道|不知道的话|不清楚|你可以)?\s*(?:可以)?"
        r"(?:联网|上网|网上)?(?:查询|搜索|搜一下|查一查|百度|谷歌).*$",
        "", raw).strip()
    return (clean or raw), True


# ── 三层分类（classify_query_node：L1 关键词快判 → L2 MiniLM → L3 LLM 细分）────
# 设计：便宜的规则信号在前面挡掉大部分 query，贵的 LLM 只处理最难的专业问题。
#   L1 规则信号直接定 → 不碰任何模型
#   L2 MiniLM 微调二分类 general/specialized（只处理 L1 漏掉的模糊 query）
#   L3 LLM 细分 PRECISE/VAGUE/BROAD（只有 specialized 触发）

# 闲聊 / 寒暄：直接走通用应答，不进检索。
_GENERAL_EXACT = {"你好", "hi", "hello", "嗨", "hey", "谢谢", "谢谢你", "感谢",
                  "thanks", "你是谁", "你叫什么", "再见", "拜拜", "bye"}
_GENERAL_KEYWORDS = ("你是谁", "你叫什么", "你能做什么", "你有什么功能",
                     "介绍一下", "今天天气", "天气怎么样", "讲个笑话",
                     "今天是", "现在几点", "现在时间", "当前时间", "星期几")

# 模糊提问信号 → VAGUE（HyDE 扩语义）。问题越短信号越可信，长的交给 L3 精判。
_VAGUE_HINTS = ("没懂", "不懂", "不太懂", "讲讲", "解释一下", "啥意思", "什么意思", "看不懂")

# 宽泛提问信号 → BROAD（Multi-Query 多维度并查）。
_BROAD_HINTS = ("全面", "系统", "总结", "梳理", "路线", "对比", "区别", "全景", "有哪些")

# 强专业信号 → 直接 PRECISE 检索，不进分类器——能明确指向研报数据的就不折腾。
# 股票代码：6 位数字，可带 A 股板块后缀（600519 / 000858.SZ）。
_COMPANY_CODE = re.compile(r"\d{6}(\.(SZ|SH|BJ))?", re.IGNORECASE)
_RAG_STRONG_TERMS = ("营收", "营业收入", "收入", "毛利率", "净利润", "净利率", "利润",
                     "现金流", "EPS", "市盈率", "PE", "PB", "估值", "评级", "目标价",
                     "股价", "市值", "ROE", "ROIC", "研报", "预测", "业绩", "增速",
                     "同比", "市占率", "股息", "分红", "产能", "存货")

# 跟随式探测（承重墙：检索前改写用上文）。"那五粮液呢" 字面没有宽泛/模糊信号，
# 规则会判成 PRECISE 直接检索——单独检索必为空。靠「前序有答 + 指代信号」路由到
# BROAD，让 multi_query_rewrite 拿上文改写成可检索的具体 query。
# 判断只关心「历史有没有先答过 + 本轮带不带指代」，对跨越多长无感；但改写的
# 上下文只喂紧邻上一轮，长跨度靠三层记忆的摘要块兜底。
_FOLLOWUP_PREFIX = ("那", "它", "他", "他们", "此", "这家", "那家")
_FOLLOWUP_SUFFIX = ("呢", "怎么样", "怎么看", "如何看", "呢？", "怎么样？")


def _looks_like_followup(query: str) -> bool:
    """本轮是否带指代信号（以"那/它…"开头，或以"呢/怎么样…"结尾）。

    :param query: 本轮用户查询文本（str）。
    :return: bool，带指代信号返回 True。
    """
    q = query.strip()
    return bool(q) and (q.startswith(_FOLLOWUP_PREFIX) or q.endswith(_FOLLOWUP_SUFFIX))


def _has_prior_answer(messages: list[dict]) -> bool:
    """历史里（不含本轮）是否已有 AI 回答——「那 X 呢」才有话可据。

    :param messages: 消息列表，元素形如 {"role","content"}；最后一条视为本轮（不参与）。
    :return: bool，历史（[:-1]）中已有 assistant 回答返回 True。
    """
    return any(m.get("role") == "assistant" for m in messages[:-1])


def _strong_rag_signal(query: str) -> bool:
    """是否带强专业信号（股票代码或财务/研报名词）→ 值得直接检索。

    :param query: 查询文本（str）。
    :return: bool，含 6 位股票代码或任一 RAG 强专业词返回 True。
    """
    return bool(_COMPANY_CODE.search(query)) or any(t in query for t in _RAG_STRONG_TERMS)


def _classify_result(*, original_query: str, query_type: str, auto_web: bool) -> dict:
    """打包 classify 返回值；联网指令只保留 enable_web_search 标记。

    :param original_query: 剥离联网语汇后的原始查询（keyword-only 参数）。
    :param query_type: 判定出的查询类型（GENERAL/PRECISE/VAGUE/BROAD，keyword-only）。
    :param auto_web: 是否带联网兜底标记（keyword-only）。
    :return: dict，含 original_query/query_type；auto_web 为真时附 enable_web_search=True。
    """
    out = {"original_query": original_query, "query_type": query_type}
    if auto_web:
        out["enable_web_search"] = True
    return out


async def _llm_call(llm, prompt: str) -> str:
    """单次 LLM 调用：重试 + 每次 wait_for 超时，对齐投研四段的 with_retry 待遇。

    qa_reports 属 NON_DEGRADABLE —— 重试尽仍失败就【上抛】，绝不拿降级哨兵回一条
    假答案。上抛后由调用方处置：classify 自带 PRECISE 兜底；生成/改写被 SSE 错误帧
    （流式）或接口 500（一次性）接住。网络抖动/超时这类短暂故障在这里重试 2 次，
    与风控段同样不可降级 —— 宁可报错，也不糊弄用户。

    :param llm: LLM 句柄，须有 ainvoke 方法。
    :param prompt: 发给模型的单条 user 提示词（str）。
    :return: str，模型返回文本；重试尽仍失败时向上抛出。
    """
    @with_retry(agent_type="qa_reports")
    async def _invoke() -> str:
        """with_retry 装饰的单次调用体。

        :return: str，模型返回文本。
        """
        return await llm.ainvoke(prompt)

    return await _invoke()


async def _refine_strategy_llm(query: str) -> str:
    """（L3）对已判 specialized 的 query，让 LLM 细化 PRECISE/VAGUE/BROAD。

    LLM 输出不稳，从回文里捞词元即可；捞不到默认 PRECISE——宁可多走一次检索，
    也比漏掉研报相关问题稳妥。

    :param query: 已判为 specialized 的查询文本（str）。
    :return: str，PRECISE/VAGUE/BROAD 之一；失败或识别不出时默认返回 "PRECISE"。
    """
    try:
        llm = get_llm("qa_reports", temperature=0)
        out = (await _llm_call(llm, prompts.RAG_STRATEGY_PROMPT.format(
            query=query))).upper()
        for kind in ("PRECISE", "VAGUE", "BROAD"):
            if kind in out:
                return kind
    except Exception as e:
        logger.warning("qa.classify_llm_failed", error=str(e))
    return "PRECISE"


async def classify_query_node(state: dict) -> dict:
    """把 query 定成 GENERAL/PRECISE/VAGUE/BROAD，供 service.run_qa 分区编排。

    三层：
      L1 规则信号直接定（闲聊 / 跟随式 / 模糊 / 宽泛 / 强专业），不碰任何模型；
      L2 MiniLM 微调二分类，只处理 L1 漏掉的模糊 query，general → 通用答；
      L3 只有 specialized 才调 LLM 细分 PRECISE/VAGUE/BROAD。
    检索相关词（联网搜…）先剥离并带出 enable_web_search 标记。

    :param state: 问答状态 dict，需含 messages（消息列表），可选 original_query。
    :return: dict 部分更新，含 original_query/query_type；联网指令时附 enable_web_search=True。
    """
    raw = _last_user_query(state.get("messages", []))
    original_query, auto_web = _extract_query_and_web_flag(raw)

    # ── L1 规则信号直接定 ──────────────────────────────────────
    if original_query.strip().lower() in _GENERAL_EXACT or \
       any(kw in original_query for kw in _GENERAL_KEYWORDS):
        return _classify_result(original_query=original_query,
                                query_type="GENERAL", auto_web=auto_web)

    # 承重墙：前序有答 且 本轮指代 → 走 BROAD 改写，别直接检索。
    if _has_prior_answer(state.get("messages", [])) and _looks_like_followup(original_query):
        logger.info("qa.followup_route_rewrite", query=original_query[:50])
        return _classify_result(original_query=original_query,
                                query_type="BROAD", auto_web=auto_web)

    # 短模糊句的信号最可信，直接走 HyDE；长句交由 L3 判断避免误判。
    if len(original_query.strip()) <= 6 and any(h in original_query for h in _VAGUE_HINTS):
        return _classify_result(original_query=original_query,
                                query_type="VAGUE", auto_web=auto_web)

    if any(kw in original_query for kw in _BROAD_HINTS):
        return _classify_result(original_query=original_query,
                                query_type="BROAD", auto_web=auto_web)

    if _strong_rag_signal(original_query):
        return _classify_result(original_query=original_query,
                                query_type="PRECISE", auto_web=auto_web)

    # ── L2 MiniLM 二分类（只处理模糊 query）─────────────────────
    # 懒加载单例，首载在 to_thread 里跑，不阻塞事件循环。
    label, _ = await asyncio.to_thread(
        lambda: QueryClassifier.get_instance().classify(original_query))
    if label == "general":
        return _classify_result(original_query=original_query,
                                query_type="GENERAL", auto_web=auto_web)

    # ── L3 LLM 细分（只有 specialized 触发）────────────────────
    strategy = await _refine_strategy_llm(original_query)
    return _classify_result(original_query=original_query,
                            query_type=strategy, auto_web=auto_web)


# ── 分层记忆（短程原文在 history；长/中程在这块给改写与生成补更早上下文）──
def _layered_memory(state: dict) -> str | None:
    """分层记忆注入块；query 给定时中程分段按相关性挑（core.memory._choose_segments）。

    :param state: 问答状态 dict，可含 segments（中程分段）、existing_summary（长程摘要）、original_query。
    :return: str 或 None，分层记忆文本；无可用记忆返回 None。
    """
    return memory.build_layered_memory(
        state.get("segments"), state.get("existing_summary"),
        query=state.get("original_query")) or None


def _system_content(state: dict) -> str:
    """拼单条 system = 静态 + 动态(当前时间/历史摘要/分层记忆) + 用户画像。

    画像只在这条常驻进 system，不逐轮追加、不进 RAG context（state 在建初始状态
    时载入 user_profile_text）。

    :param state: 问答状态 dict，可含 user_profile_text（用户画像）、segments、existing_summary。
    :return: str，拼好的单条 system 文本。
    """
    return prompts.build_system_content(
        _layered_memory(state), profile=state.get("user_profile_text"))


# ── HyDE（VAGUE）：生成假设性研报片段替代原 query 检索 ───────────
async def hyde_generate_node(state: dict) -> dict:
    """HyDE 分支：为 VAGUE 查询生成一段假设性研报片段替代原 query 检索。

    :param state: 问答状态 dict，需含 original_query，可含 messages。
    :return: dict 部分更新，含 hyde_document（生成的假设研报文本）。
    """
    query = state["original_query"]
    history = prompts.format_history_for_prompt(state.get("messages", []))
    llm = get_llm("qa_reports", temperature=0.3)
    hyde_doc = (await _llm_call(llm, prompts.build_hyde_prompt(
        history, query, _layered_memory(state)))).strip()
    logger.info("qa.hyde_generated", query=query[:50], length=len(hyde_doc))
    return {"hyde_document": hyde_doc}


# ── Multi-Query（BROAD）：分析成 3 个独立子 query 并行检索 ───────
async def multi_query_rewrite_node(state: dict) -> dict:
    """Multi-Query 分支：把宽泛问题改写为多个可独立检索的子查询。

    :param state: 问答状态 dict，需含 original_query，可含 messages。
    :return: dict 部分更新，含 rewritten_queries（改写出的子查询列表，至多 MAX_BROAD_QUERIES 条；改写全部为空时回退为原 query 单条）。
    """
    query = state["original_query"]
    # 改写喂「最近几轮原文」而非紧邻一轮 —— 中间隔着寒暄（谢谢/不错）时，指代
    # 也能从窗口里更早的那轮上下文解出；比只给上一轮 AI 回答稳。本轮 user 已并入
    # query，故取 messages[:-1]。更早的历史由三层记忆的摘要块补充。
    history = prompts.format_history_for_prompt(state.get("messages", [])[:-1])
    llm = get_llm("qa_reports", temperature=0.3)
    raw = (await _llm_call(llm, prompts.build_multi_query_prompt(
        history, query, _layered_memory(state)))).strip()
    rewritten = []
    for line in raw.split("\n"):
        line = line.lstrip("0123456789.-）、) ").strip()
        if line and len(line) > 3:
            rewritten.append(line)
        if len(rewritten) >= MAX_BROAD_QUERIES:
            break
    if not rewritten:
        rewritten = [query]
    logger.info("qa.multi_rewritten", original=query[:50], count=len(rewritten))
    return {"rewritten_queries": rewritten}


# ── retrieve：按 query_type 检索 report_corpus ─────────────────
async def _search_kb(query: str, tenant_id: str, recall: int, rerank: int) -> list[dict]:
    """研报检索辅助：真走 MCP 协议调 search_knowledge_base（/mcp/kb）。

    MCP 返回的是扁平 dict（content/score/source_name/company_code/industry/
    report_type/published_at），没有嵌套的 metadata 键；这里规整成下游要的
    {content, score, metadata} 形态（metadata 从扁平字段拼回），后续消费者
    （ranked_chunks / agentic._accumulate / build_context_and_sources）不用改。

    与联网搜索一致：不在此静默切回直连函数，MCP 失败就把异常抛给调用方处理。

    :param query:  检索提问（str）。
    :param tenant_id: 租户 id。
    :param recall: 召回条数（int）。
    :param rerank: 精排返回条数（int）。
    :return: list[dict]，规整成 {content, score, metadata} 的研报切片列表。
    """
    settings = get_settings()
    # timeout=120：BROAD 多路并发时每个 rerank batch 在低端机上实测要 70s+，
    # 走默认 30s 会踩 httpx.ReadTimeout（str() 为空，记成 error='' 像是假失败）。
    # 与采集段 fetch_all_sources 放宽到 180s 同一条理由：别把慢当失败。
    flat = await call_mcp_tool(
        settings.kb_mcp_server_url, "search_knowledge_base",
        {"query": query, "tenant_id": tenant_id,
         "recall_top_k": recall, "rerank_top_k": rerank},
        timeout=120.0)
    docs = []
    for d in (flat or []):
        docs.append({
            "content": d.get("content", ""),
            "score": d.get("score", 0.0),
            "metadata": {
                "source_name": d.get("source_name") or "",
                "company_code": d.get("company_code") or "",
                "industry": d.get("industry") or "",
                "report_type": d.get("report_type") or "",
                "published_at": d.get("published_at"),
            },
        })
    return docs


async def retrieve_node(state: dict) -> dict:
    """按 query_type 检索已发布研报语料，返回精排 chunks 与置信度。

    :param state: 问答状态 dict，需含 tenant_id、query_type；BROAD 用 rewritten_queries、VAGUE 用 hyde_document、其它用 original_query。
    :return: dict 部分更新，含 ranked_chunks/confidence/is_high_confidence。
    """
    qtype = state.get("query_type", "PRECISE").upper()
    tenant_id = state["tenant_id"]

    async def _one(query: str, recall: int, rerank: int) -> list[dict]:
        """单条查询检索辅助：真走 MCP 调 search_knowledge_base。"""
        return await _search_kb(query, tenant_id, recall, rerank)

    if qtype == "BROAD" and state.get("rewritten_queries"):
        subs = state["rewritten_queries"][:MAX_BROAD_QUERIES]
        # 串行执行而不是 asyncio.gather 并发：BROAD 每路子 query 都要对
        # BGE-Reranker-large 做一次 CPU 精排（本地模型），在低核数机器（如 2 核 4G）
        # 上并发三路推理反而互抢 CPU/内存、把每路拖到远超超时阈值（实测 70s+），
        # 表现为「BROAD 必超时」。串行后每路独占算力，单路时有足够余量进超时。
        # 代价是总的多次检索变顺序耗时，但对本地慢模型，正确性/可用性优先于吞吐。
        result_lists = []
        for _q in subs:
            result_lists.append(
                await _one(_q, RECALL_TOP_K_BROAD_PER, RERANK_TOP_K))
        seen: dict[str, tuple[dict, float]] = {}
        for docs in result_lists:
            for doc in docs:
                key = doc["content"][:100]
                if key not in seen or doc["score"] > seen[key][0]["score"]:
                    seen[key] = (doc, doc["score"])
        merged = [d[0] for d in sorted(seen.values(), key=lambda x: x[1], reverse=True)]
        merged = merged[:RERANK_TOP_K]
    elif qtype == "VAGUE" and state.get("hyde_document"):
        merged = await _one(state["hyde_document"], RECALL_TOP_K_VAGUE, RERANK_TOP_K)
    else:
        merged = await _one(state["original_query"], RECALL_TOP_K_PRECISE, RERANK_TOP_K)

    confidence = merged[0]["score"] if merged else 0.0
    ranked_chunks = [{"content": d["content"], "score": d["score"],
                      "metadata": d["metadata"]} for d in merged]
    logger.info("qa.retrieved", qtype=qtype, ranked=len(ranked_chunks),
                confidence=round(confidence, 4))
    return {
        "ranked_chunks": ranked_chunks,
        "confidence": confidence,
        "is_high_confidence": confidence >= HIGH_CONFIDENCE_BAR,
    }


# ── 生成公共：流式或一次性 ──────────────────────────────────────
async def _generate(streamer, llm_temperature: float, prompt: str) -> str:
    """生成公共：流式或一次性两种模式取 LLM 完成文本。

    :param streamer: SSE 流式回调（可为 None）；为 None 走一次性 with_retry 路径，否则逐字推送 token 且不重试。
    :param llm_temperature: LLM 采样温度（float）。
    :param prompt: 发给模型的完整 user 提示词（str）。
    :return: str，去首尾空白后的完整回答文本。
    """
    llm = get_llm("qa_reports", temperature=llm_temperature,
                  streaming=streamer is not None)
    # 一次性：走 _llm_call（重试 + wait_for 超时），短暂故障重试 2 次，重试尽上抛。
    if streamer is None:
        return (await _llm_call(llm, prompt)).strip()
    # 流式：不重试 —— 半截已推给 SSE，重拉会发重复 token；只靠 httpx 120s 传输超时
    # 兜底，中途失败由 run_qa_stream 的 __error__ 帧让前端回滚本轮。
    parts: list[str] = []
    async for chunk in llm.astream(prompt):
        if chunk:
            parts.append(chunk)
            await streamer.token(chunk)
    return "".join(parts).strip()


# ── 生成：RAG（高置信，严格基于研报）───────────────────────────
async def generate_rag_node(state: dict, streamer=None) -> dict:
    """生成 RAG 回答（高置信，严格基于研报上下文）。

    :param state: 问答状态 dict，用 ranked_chunks/original_query/messages/user_profile_text/confidence。
    :param streamer: 可选 SSE 流式回调（None 为一次性）。
    :return: dict 部分更新，含 answer/sources/answer_mode("rag")/confidence。
    """
    chunks = state.get("ranked_chunks", [])
    query = state["original_query"]
    context, sources = prompts.build_context_and_sources(chunks)
    system = _system_content(state)
    history = prompts.format_history_for_prompt(
        state.get("messages", [])[:-1])            # 排除本轮 user（已拼进 query）
    prompt = prompts.build_rag_prompt(
        system=system, history=history, context=context, query=query)

    answer = await _generate(streamer, 0.3, prompt)
    final = answer
    if sources:
        final += "\n\n📚 **参考来源**\n" + "\n".join(f"  • {s}" for s in sources)
    return {"answer": final, "sources": sources, "answer_mode": "rag",
            "confidence": state.get("confidence", 0.0)}


# ── 联网搜索：真走 MCP 协议（发到 /mcp/web-search 的 tools/call）──
# 「先重试再报错」由 with_retry 兜：重试 2 次（间隔 1s/3s），仍失败时返回
# FallbackResult 哨兵（故意不是 dict）。这里检测到哨兵就当作「联网不可用」返回空，
# 让调用方走 _web_failed_ 降级（低置信再落到 llm_direct）。不在此静默切回直连函数
# ——那是绕过 MCP 的旧行为，按「重试后报错」语义移除。研报检索仍走 core 直连不经 MCP。
@with_retry(agent_type="qa_web")
async def _mcp_call(query: str, max_results: int) -> list[dict]:
    """真走 MCP 协议调 web_search 工具（带 with_retry 重试）。

    :param query: 搜索关键词（str）。
    :param max_results: 期望返回的结果数量上限（int）。
    :return: list[dict]，MCP 返回的搜索结果；重试尽仍失败时返回 FallbackResult 哨兵。
    """
    settings = get_settings()
    return await call_mcp_tool(
        settings.web_search_mcp_url, "web_search",
        {"query": query, "max_results": max_results})


async def mcp_web_search(query: str, max_results: int = 5) -> list[dict]:
    """真走 MCP 协议联网搜索。失败先重试（with_retry 内），仍不可用则记错返回空。

    :param query: 搜索关键词（str）。
    :param max_results: 结果数量上限，默认 5。
    :return: list[dict]，搜索结果；联网不可用时返回空列表。
    """
    result = await _mcp_call(query, max_results)
    if isinstance(result, FallbackResult):
        logger.warning("qa.web_mcp_unavailable", agent_type=result.agent_type,
                       layer=result.layer, note=result.note)
        return []
    return result or []


# ── 生成：联网兜底（低置信 + enable_web_search，实时网页补齐）────────
async def generate_web_node(state: dict, streamer=None) -> dict:
    """生成联网兜底回答（低置信 + enable_web_search，用实时网页补齐）。

    :param state: 问答状态 dict，用 original_query/messages；联网失败时返回 _web_failed_ 标记。
    :param streamer: 可选 SSE 流式回调（None 为一次性）。
    :return: dict；联网不可用时仅含 answer_mode="_web_failed_"，否则含 answer/sources/answer_mode("web")/confidence。
    """
    query = state["original_query"]
    results = await mcp_web_search(query, 5)     # 真走 MCP；失败已重试，仍不可用 → 空
    if not results:
        return {"answer_mode": "_web_failed_"}

    context, urls = prompts.build_web_context_and_sources(results)
    system = _system_content(state)
    history = prompts.format_history_for_prompt(state.get("messages", [])[:-1])
    prompt = prompts.build_web_prompt(
        system=system, history=history, context=context, query=query)

    answer = await _generate(streamer, 0.3, prompt)
    if urls:
        # 统一用「参考来源」作唯一标签：RAG 检索(nodes.py:445)、本联网兜底、agentic
        # 三路都拼这个，避免同一批链接在「🔗 联网来源」和「📚 参考来源」两个标签下重复出现。
        answer += "\n\n📚 **参考来源**\n" + "\n".join(f"  • {u}" for u in urls)
    return {"answer": answer, "sources": urls, "answer_mode": "web",
            "confidence": state.get("confidence", 0.0)}


# ── 生成：LLM 直答兜底（低置信或未命中）─────────────────────────
async def generate_direct_node(state: dict, streamer=None) -> dict:
    """生成 LLM 直答兜底（低置信或未命中研报语料）。

    :param state: 问答状态 dict，用 original_query/messages。
    :param streamer: 可选 SSE 流式回调（None 为一次性）。
    :return: dict 部分更新，含 answer/sources(空)/answer_mode("llm_direct")/confidence。
    """
    query = state["original_query"]
    system = _system_content(state)
    history = prompts.format_history_for_prompt(state.get("messages", [])[:-1])
    prompt = prompts.build_direct_prompt(
        system=system, history=history, query=query)

    answer = await _generate(streamer, 0.3, prompt)
    # 说明/免责声明不在这里硬编码附加——「需不需要加、该不该加」由模型按提示里的
    # 条件自行判断（见 DIRECT_ANSWER_PROMPT 规则 4/5）：具体事实未命中语料时补一句，
    # 询问系统能力/语料内容的元问题不追加任何说明。
    return {"answer": answer, "sources": [], "answer_mode": "llm_direct",
            "confidence": state.get("confidence", 0.0)}


# ── 生成：闲聊/时间（GENERAL）──────────────────────────────────
async def generate_general_node(state: dict, streamer=None) -> dict:
    """生成闲聊/时间回答（GENERAL 分支）。

    :param state: 问答状态 dict，用 original_query/messages。
    :param streamer: 可选 SSE 流式回调（None 为一次性）。
    :return: dict 部分更新，含 answer/sources(空)/answer_mode("general")/confidence(恒 1.0)。
    """
    query = state["original_query"]
    history = prompts.format_history_for_prompt(state.get("messages", []))
    prompt = prompts.build_general_prompt(
        current_time=prompts.current_datetime_str(),
        history=history, query=query)
    answer = await _generate(streamer, 0.3, prompt)
    return {"answer": answer, "sources": [], "answer_mode": "general",
            "confidence": 1.0}


# ── enqueue_pending：低置信度问题入库供补料 ─────────────────────
async def enqueue_pending_node(state: dict) -> dict:
    """低置信度问题入库（knowledge_pending_queue）供补料；失败进死信。

    :param state: 问答状态 dict，用 tenant_id/user_id/original_query/confidence/thread_id。
    :return: dict，恒为空 {}（更新无回写）。
    """
    try:
        await qa_repo.enqueue_pending(
            tenant_id=state["tenant_id"], user_id=state["user_id"],
            question=state["original_query"],
            confidence=state.get("confidence", 0.0))
    except Exception as e:
        logger.warning("qa.enqueue_failed", error=str(e))
        # F 铁律：入队失败的载荷不丢，进死信供人工补料。
        try:
            await qa_repo.add_dead_letter(
                state["tenant_id"], "qa_enqueue_pending", state.get("thread_id", ""),
                {"user_id": state.get("user_id"), "question": state["original_query"],
                 "confidence": state.get("confidence", 0.0)},
                str(e))
        except Exception as ce:
            logger.error("qa.dead_letter_failed", error=str(ce)[:200])
    return {}


# ── enqueue_sediment：agentic 高质量答案入沉淀桶供离线双落 ───────
# 与 enqueue_pending 的区别：那边是「没答案的低置信问题」靠联动脚本 LLM 生成；
# 这里是「已有现成好答案」（answer_mode=agentic, 有 evidence）直接落桶，不动文本。
# 门槛在 service.run_qa 收口：仅 answer_mode=="agentic" 才调本节点。
async def enqueue_sediment_node(state: dict) -> dict:
    """agentic 高质量答案入沉淀桶（qa_sediment_queue）供离线双落；失败进死信。

    :param state: 问答状态 dict，用 tenant_id/user_id/original_query/answer/sources/thread_id。
    :return: dict，恒为空 {}。
    """
    try:
        await qa_repo.enqueue_sediment(
            tenant_id=state["tenant_id"], user_id=state["user_id"],
            question=state["original_query"],
            answer=state.get("answer", ""),
            sources="\n".join(state.get("sources") or []))
    except Exception as e:
        logger.warning("qa.sediment_enqueue_failed", error=str(e))
        # F 铁律：入桶失败的载荷不丢，进死信供人工复核。
        try:
            await qa_repo.add_dead_letter(
                state["tenant_id"], "qa_sediment_enqueue", state.get("thread_id", ""),
                {"user_id": state.get("user_id"), "question": state["original_query"],
                 "answer": state.get("answer"), "sources": state.get("sources")},
                str(e))
        except Exception as ce:
            logger.error("qa.sediment_dead_letter_failed", error=str(ce)[:200])
    return {}


# ── save_memory：落消息 + 短/中/长程三层记忆 ────────────────────
# 压缩编排（游标 / 触发 / 折叠）抽到 core.memory；这里只保留 F 铁律的落库死信。
async def save_memory_node(state: dict) -> dict:
    """落消息 + 短/中/长程三层记忆；落库失败进死信。

    :param state: 问答状态 dict，用 thread_id/messages/answer/existing_summary/segments/tenant_id/user_id。
    :return: dict，恒为空 {}（更新无回写）。
    """
    thread_id = state["thread_id"]
    raw_query = _last_user_query(state.get("messages", []))
    answer = state.get("answer", "")
    summary = state.get("existing_summary")          # 长程
    segments = list(state.get("segments") or [])     # 中程（最近在前）

    try:
        await qa_repo.append_messages(thread_id, raw_query, answer)
    except Exception as e:
        logger.warning("qa.save_messages_failed", error=str(e))
        # F 铁律：落库失败的载荷不丢，进死信供人工复核。
        try:
            await qa_repo.add_dead_letter(
                state["tenant_id"], "qa_save_messages", thread_id,
                {"user": raw_query, "assistant": answer}, str(e))
        except Exception as ce:
            logger.error("qa.dead_letter_failed", error=str(ce)[:200])
        return {}

    # 三层折叠（core.memory）：按 DB 权威游标只压「未覆盖回合」→ 一块中程分段；
    # 块数超上限折入长程。未触发则幂等写回当前 (summary, segments)，不消耗 LLM。
    try:
        llm = get_llm("qa_reports", temperature=0)
        segments, summary = await memory.fold_recent(
            thread_id, raw_messages=state.get("messages", []),
            summary=summary, segments=segments, llm=llm)
        await memory.save(thread_id, state["tenant_id"], state["user_id"],
                          summary=summary, segments=segments)
    except Exception as e:
        logger.warning("qa.save_summary_failed", error=str(e))
    return {}