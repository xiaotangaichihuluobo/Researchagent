# backend/agents/qa/prompts.py
# 轨道 B「问已发布研报」的各节点提示词。从通用问答范式移植，改写为研报语境。
# 末尾的 build_*（纯函数）把「系统 + 本轮 + 历史 + 上下文」拼成一条 user prompt 字符串，
# 因为本工程 llm_factory 的 handle 只收单条 user prompt（无 langchain message 列表）。

from datetime import datetime

# ── 系统提示：静态 / 动态分界（对齐 s10「运行时组装、静态段缓存」）──
# 这条 SYSTEM 每轮都进 prompt，是这段问答引擎里唯一「反复发送」的常驻文本，
# 故按 s10 的「静态 / 动态」画出分界，留给未来接 API prompt cache 的接缝：
#     静态段 = build_system_static()  → SYSTEM_PROMPT，恒定不变，可做 cache 前缀
#     动态段 = build_system_dynamic() → 当前时间 + 分层记忆摘要，随轮次/状态变
# 未来接 cache 时，只需在静态段末尾插 SYSTEM_PROMPT_DYNAMIC_BOUNDARY 分隔符
# （参考 s10_system_prompt：静态划进 global cache block、动态 cacheScope=null），
# 此处的函数边界就是现成的插点，不用再拆分。前提注意：llm_factory 现阶段只喂
# 单条 user prompt，这条 system 是被 build_* 平铺进 prompt 开头的 —— 所以接 cache
# 时要么让 _complete 收 system 参数、要么在此拼好边界符，二选一，见注释。
SYSTEM_PROMPT = """你是投研知识助手，负责基于【本机构已发布的研究报告】回答分析师的问题。

【你的角色】
- 只依据提供的研报参考内容回答，不引入研报之外的信息
- 语言风格：专业、简洁、基于事实；涉及数字时直接引用
- 回答长度：视问题复杂度适当调整，不过度展开，不简单敷衍

【回答规范】
- 先给结论，再给支撑（引用研报观点/数据）
- 参考内容不足以完整回答时，明确哪些来自研报、哪些是补充说明
- 不编造不确定的信息；涉及具体金额/增速若不在一材料中，宁缺勿编"""


def current_datetime_str() -> str:
    """取格式化后的当前时间字符串（含星期与时分）。

    :return: str，形如 "2026年09月27日 星期一 14:30"。
    """
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    now = datetime.now()
    return now.strftime(f"%Y年%m月%d日 {weekdays[now.weekday()]} %H:%M")


def build_system_static() -> str:
    """静态段：只含角色指令，恒定不变 —— 未来 API prompt cache 的可缓存前缀。

    单独暴露出来，让「恒定 vs 随会话变」的分界可被代码触达；当前 build_system_content
    把它和动态段拼在一起，拼接结果与改写前逐字一致，行为没变。

    :return: str，恒为 SYSTEM_PROMPT 静态段文本。
    """
    return SYSTEM_PROMPT


def build_system_dynamic(summary: str | None = None,
                         profile: str | None = None) -> str:
    """动态段：当前时间 + 可选的历史摘要 + 可选的用户画像，随轮次 / 会话状态变。

    当前时间虽一小时不变，仍是慢变量 —— 归到动态段；画像属「跨会话稳定偏好」，
    只在建初始状态载入（build_initial_state → state["user_profile_text"]）并拼进
    system 一次：常驻、不逐轮变、不进 RAG context（见 docs/跨会话记忆设计.md §5）。

    :param summary: 历史对话摘要，可空（默认 None）。
    :param profile: 用户画像文本，可空（默认 None）。
    :return: str，含当前时间并按需追加摘要/画像的动态段文本。
    """
    content = f"\n\n【当前时间】{current_datetime_str()}"
    if summary:
        content += f"\n\n【历史对话摘要】\n{summary}"
    if profile:
        content += f"\n\n【用户画像（该分析师的长期偏好，回答时体量这些）】\n{profile}"
    return content


def build_system_content(summary: str | None = None,
                         profile: str | None = None) -> str:
    """拼出单条 system 文本 = 静态段 + 动态段。为兼容保留原名，调用点不动。

    :param summary: 历史摘要，转发给 build_system_dynamic（可空，默认 None）。
    :param profile: 用户画像，转发给 build_system_dynamic（可空，默认 None）。
    :return: str，静态段 + 动态段拼接后的单条 system 文本。
    """
    return f"{build_system_static()}{build_system_dynamic(summary, profile)}"


def format_history_for_prompt(messages: list[dict]) -> str:
    """把最近几轮消息格式化为提示词可用的历史文本。消息形如 {"role","content"}。

    :param messages: 消息列表（至多取最近 6 条），元素含 role 与 content。
    :return: str，多行历史文本；无消息时返回 "（无历史对话）"。
    """
    lines = []
    for msg in messages[-6:]:
        if msg.get("role") == "user":
            lines.append(f"分析师：{msg['content']}")
        else:
            lines.append(f"AI：{msg.get('content', '')[:200]}...")
    return "\n".join(lines) if lines else "（无历史对话）"


# ── RAG 策略精校（classify：规则判为 VAGUE/BROAD 且问题较长时）────
RAG_STRATEGY_PROMPT = """判断以下问题应采用哪种研报检索策略。

【策略定义】
PRECISE：问题表达明确，含具体标的/财务指标，可直接向量检索。
         例："贵州茅台 2025 年营收是多少" / "五粮液的毛利率近几年变化"
VAGUE  ：问题模糊，只给宽泛意图，直接检索效果差，需先生成假设文档扩充语义再检索（HyDE）。
         例："解释一下" / "我没太懂这块" / "能说说吗"
BROAD  ：问题范围过宽或极度简短，需拆成多个子问题并行检索扩大召回（Multi-Query）。
         例："没懂" / "讲讲五粮液" / "白酒行业怎么样"

【问题】
{query}

只输出一个词：PRECISE 或 VAGUE 或 BROAD"""


# ── HyDE 假设文档生成（VAGUE 分支）──────────────────────────────
HYDE_PROMPT = """你是一位投研分析助手。请根据以下对话上下文，推断提问的具体意图，
并生成一段高质量的研报观点片段作为假设性回答。

【对话上下文（最近几轮）】
{history}
{summary}

【分析师当前输入】
{query}

请直接输出一段专业的研报内容（150-300字），表达要具体（点名可能涉及的公司/指标），
不要包含"假设"或"可能"等不确定语气。输出格式：纯文本。"""


# ── Multi-Query 改写（BROAD 分支）───────────────────────────────
MULTI_QUERY_REWRITE_PROMPT = """你是一位投研助手，负责把模糊/宽泛的问题改写为多个具体的检索问题。

【最近几轮对话（供推断指代与上下文，中间可能隔着寒暄）】
{history}
{summary}

【分析师当前输入】
{query}

请把问题改写为 3-5 个具体、独立、可单独检索的研报问题，每行一个。
要求：
- 每个问题独立完整（含公司/指标，能直接拿去检索研报语料）
- 覆盖不同角度（营收 / 毛利率 / 行业地位 / 估值）
- 不要编号，直接输出问题文本

输出示例：
贵州茅台 2025 年营业收入同比增长多少？
贵州茅台近三年毛利率是多少？
贵州茅台 2025 年盈利预测如何？"""


# ── RAG 回答生成（高置信，严格基于研报）────────────────────────
RAG_ANSWER_PROMPT = """请基于以下研报参考内容，回答分析师的问题。

【研报参考内容】
{context}

【问题】
{query}

【回答要求】
1. 严格基于参考内容回答，不要引入参考内容之外的信息
2. 如果参考内容不足以完整回答问题，明确说明哪些部分来自研报、哪些是补充说明
3. 涉及具体数字（营收/毛利/增速）时直接引用并在适当时标注所属公司
4. 回答简洁清晰，直接切入要点"""


# ── LLM 直答兜底（低置信，未命中研报语料）──────────────────────
DIRECT_ANSWER_PROMPT = """请根据你的投研通用知识回答以下问题。

【问题】
{query}

【回答要求】
1. 基于通用金融/投研知识回答，不要声称来自本机构研报
2. 如果不确定，明确说明不确定的部分
3. 回答简洁准确"""


# ── 联网兜底（低置信 + enable_web_search：用实时网页内容补充回答）──
WEB_ANSWER_PROMPT = """请基于以下联网检索到的网页内容，回答分析师的问题。

【网页参考内容】
{context}

【问题】
{query}

【回答要求】
1. 忠于网页内容回答，不要编造检索结果里没有的数据
2. 内容不足以回答时，明确说明哪些来自网页、哪些是补充判断
3. 涉及具体数字（营收/毛利/增速）直接引用，并标注数据来源时点
4. 回答简洁清晰，直接切入要点"""


# ── agentic 多轮自搜：每轮决策（search/answer）───────────────────
# 只在 qa_engine_mode=agentic 时用。让模型看当前问题 + 已搜到的材料，
# 决定是再搜还是够了直接答。材料不足时把缺口拆成【多个独立子查询】并行检索，
# 覆盖缺失的不同侧面（对齐 BROAD 的 multi-query 展开，见 nodes.multi_query_rewrite_node）。
# 输出形状由 AgentStepDecision schema 约束，temperature=0，保证每一轮决断稳定。
AGENT_LOOP_PROMPT = """你是投研知识助手，正在进行「多轮自搜」回答。目标是：能用研报原文回答就回答，
材料不够就换更聚焦的关键词再搜，直到足够或达到轮数上限。

【分析师问题】
{query}

【最近对话】
{history}
{summary}

【目前已搜到的材料（可引用，勿编造其中没有的数字）】
{evidence}

请决断本轮动作：
- 若以上材料已能严谨回答问题 → action=answer
- 若材料不足，或某块关键信息缺失 → action=search，并把缺失信息拆成 2~3 个
  【相互独立】的子查询：每个覆盖一个缺口侧面（如不同标的/不同指标/不同年份），
  各自做成能直接命中研报语料的检索串。不要改写原问句，也不要重复已搜过的角度。

只输出决断结构：action（search/answer）+ queries（子查询列表）+ reason。"""


# ── agentic 终局回答：只许引用累计证据 ──────────────────────────
# 数字闸口就落在这里：回答里出现的营收/增速/估值/市值等数字，必须来自 {evidence}
# 中某条原文；材料里没有的数据一律不写。这是「数字不经 LLM」在 agentic 的落点。
AGENT_ANSWER_PROMPT = """请基于以下经过多轮检索收集的研究报告材料，回答分析师的问题。

【分析师问题】
{query}

【最近对话】
{history}
{summary}

【累计材料】
{context}

【回答要求】
1. 严格基于 {context} 中的材料回答，不要引入材料之外的研报信息
2. 涉及具体数字（营收/毛利/增速/市值等）时，只引用材料中出现的数字；材料里
   没有的数字一律不写——宁可说明"材料未提供"，也不编造
3. 如果用了多轮检索才凑齐信息，可在回答里用一两句说明各个关键点取自哪轮材料
4. 材料不足以完整回答时，明确说明哪些来自研报、哪些是缺失
5. 简洁清晰，直接切入要点"""


# ── 通用问题直答（GENERAL，跳过 RAG）────────────────────────────
GENERAL_ANSWER_PROMPT = """你是投研知识助手。

【当前时间】{current_time}

【历史对话（最近几轮）】
{history}

【问题】
{query}

请直接回答。
- 语言友善、简洁
- 如果问题涉及时间/日期，直接根据【当前时间】作答
- 若与研报/股票无关，按闲聊自然回答，并提示可询问题研报内容"""


def _summary_block(summary: str | None, header: str) -> str:
    """把会话摘要拼成可选注入块；没有摘要则返回空串，不留空标题。

    :param summary: 摘要文本，空值返回空串。
    :param header: 注入块的标题行文案（str）。
    :return: str，摘要为空白时返回 ""，否则返回标题 + 摘要（截断至 800 字）。
    """
    if not summary:
        return ""
    return f"{header}\n{summary[:800]}"


# 三层记忆（中程分段 / 长程折叠 / 分层注入块）已抽到 core.memory —— 见
# backend/core/memory.py 的 build_segment_summary_prompt / build_fold_prompt /
# build_layered_memory。这里只保留「单条摘要」的注入辅助与各回答 prompt。


def build_hyde_prompt(history: str, query: str, summary: str | None = None) -> str:
    """拼 HyDE 假设文档生成的用户提示。

    :param history: 最近几轮对话文本（str）。
    :param query: 当前查询（str）。
    :param summary: 长会话摘要，可空（默认 None）。
    :return: str，填好参数的 HYDE_PROMPT。
    """
    return HYDE_PROMPT.format(
        history=history, summary=_summary_block(summary, "【历史摘要（长会话用，帮助理解更早的指代）】"),
        query=query)


def build_multi_query_prompt(history: str, query: str,
                             summary: str | None = None) -> str:
    """拼 Multi-Query 改写（BROAD 分支）的用户提示。

    :param history: 最近几轮对话文本（str）。
    :param query: 当前查询（str）。
    :param summary: 长会话摘要，可空（默认 None）。
    :return: str，填好参数的 MULTI_QUERY_REWRITE_PROMPT。
    """
    return MULTI_QUERY_REWRITE_PROMPT.format(
        history=history,
        summary=_summary_block(summary, "【历史摘要（长会话用，帮助理解更早的指代）】"),
        query=query)


def build_rag_prompt(*, system: str, history: str, context: str, query: str) -> str:
    """拼 RAG 回答生成的完整 user 提示（system 已内联开头）。

    :param system: system 上下文文本（keyword-only）。
    :param history: 历史对话文本（keyword-only）。
    :param context: 研报参考上下文（keyword-only）。
    :param query: 当前查询（keyword-only）。
    :return: str，完整的 RAG 生成提示。
    """
    return f"{system}\n\n【历史对话】\n{history}\n\n{RAG_ANSWER_PROMPT.format(context=context, query=query)}"


def build_direct_prompt(*, system: str, history: str, query: str) -> str:
    """拼 LLM 直答兜底（DIRECT）的完整 user 提示。

    :param system: system 上下文文本（keyword-only）。
    :param history: 历史对话文本（keyword-only）。
    :param query: 当前查询（keyword-only）。
    :return: str，完整的直答提示。
    """
    return f"{system}\n\n【历史对话】\n{history}\n\n{DIRECT_ANSWER_PROMPT.format(query=query)}"


def build_web_prompt(*, system: str, history: str, context: str, query: str) -> str:
    """拼联网兜底回答（WEB）的完整 user 提示。

    :param system: system 上下文文本（keyword-only）。
    :param history: 历史对话文本（keyword-only）。
    :param context: 联网检索到的网页上下文（keyword-only）。
    :param query: 当前查询（keyword-only）。
    :return: str，完整的联网回答提示。
    """
    return f"{system}\n\n【历史对话】\n{history}\n\n{WEB_ANSWER_PROMPT.format(context=context, query=query)}"


def build_general_prompt(*, current_time: str, history: str, query: str) -> str:
    """拼 GENERAL（闲聊/时间）回答的完整 user 提示。

    :param current_time: 当前时间字符串（keyword-only）。
    :param history: 历史对话文本（keyword-only）。
    :param query: 当前查询（keyword-only）。
    :return: str，完整的 GENERAL 提示。
    """
    return GENERAL_ANSWER_PROMPT.format(
        current_time=current_time, history=history, query=query)


def build_context_and_sources(ranked_chunks: list[dict]) -> tuple[str, list[str]]:
    """把精排 chunks 拼成【参考{i}】context，并收集去重后的可读来源标签。

    :param ranked_chunks: 精排后的文档块列表，每项含 content 与 metadata（可含 source_name）。
    :return: (context, sources) 二重组：context 为拼好的参考文本，sources 为去重后的来源标签 list。
    """
    parts, sources = [], []
    for i, chunk in enumerate(ranked_chunks, 1):
        parts.append(f"【参考{i}】\n{chunk['content']}")
        src = chunk.get("metadata", {}).get("source_name", "已发布研报")
        if src not in sources:
            sources.append(src)
    return "\n\n".join(parts), sources


def build_web_context_and_sources(results: list[dict]) -> tuple[str, list[str]]:
    """把联网搜索结果（{title,url,content}）拼成 context，并收集去重后的 URL。

    只取每个结果正文前 500 字（与参考工程 web 兜底一致，避免塞满 prompt）。

    :param results: 联网搜索结果列表，每项可含 title/url/content/snippet。
    :return: (context, urls) 二重组：context 为拼好的来源文本，urls 为去重后的 URL 列表。
    """
    parts, urls = [], []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        body = (r.get("content") or r.get("snippet") or "")[:600]
        parts.append(f"【来源{i}】{title}\n{body}")
        url = r.get("url")
        if url and url not in urls:
            urls.append(url)
    return "\n\n".join(parts), urls


# ── agentic 多轮自搜：决策与终局回答的拼装 ───────────────────────
def _profile_block(profile: str | None) -> str:
    """把用户画像拼成可选注入块；没有则返回空串，不留空标题。

    :param profile: 用户画像文本，空值返回空串。
    :return: str，经 _summary_block 拼出的用户画像注入块文本。
    """
    return _summary_block(profile, "【用户画像（该分析师的长期偏好，回答时体量这些）】")


def build_agent_loop_prompt(*, query: str, history: str,
                            summary: str | None, evidence: str,
                            profile: str | None = None) -> str:
    """把当前问题 + 历史 + 画像 + 已搜材料拼成「这轮搜还是答」的决策输入。

    :param query: 分析师问题（keyword-only）。
    :param history: 最近对话文本（keyword-only）。
    :param summary: 历史摘要，可空（keyword-only）。
    :param evidence: 已搜到材料文本；为空时注入 "本轮还没搜到任何材料" 占位。
    :param profile: 用户画像，可空（默认 None）。
    :return: str，完整的 agentic 决策提示（AGENT_LOOP_PROMPT）。
    """
    evid = evidence or "（本轮还没搜到任何材料）"
    return AGENT_LOOP_PROMPT.format(
        query=query, history=history,
        summary=_summary_block(summary, "【历史摘要（长会话用，帮助理解更早的指代）】")
                + _profile_block(profile),
        evidence=evid)


def build_agent_answer_prompt(*, query: str, history: str,
                              summary: str | None, context: str,
                              profile: str | None = None) -> str:
    """拼终局回答输入：把累计证据 + 数字闸口指令交给模型。

    :param query: 分析师问题（keyword-only）。
    :param history: 最近对话文本（keyword-only）。
    :param summary: 历史摘要，可空（keyword-only）。
    :param context: 累计证据文本（keyword-only）。
    :param profile: 用户画像，可空（默认 None）。
    :return: str，完整的终局回答提示（AGENT_ANSWER_PROMPT）。
    """
    return AGENT_ANSWER_PROMPT.format(
        query=query, history=history,
        summary=_summary_block(summary, "【历史摘要（长会话用，帮助理解更早的指代）】")
                + _profile_block(profile),
        context=context)