# backend/core/memory.py
# 轨道 B「问已发布研报」的会话记忆：短/中/长程三层上下文的压缩编排 + 存储薄包装。
#
# 分层（边界是结构性的，非顺口）：
#   短程 = 近 SHORT_TERM_TURNS 轮原文逐字（prompt 【历史对话】只喂这些）；
#   中程 = 每攒满 SEGMENT_TURNS 个「未覆盖」用户回合压成一块（最近在前），
#          池上限 MID_TERM_RETENTION —— 超出才折入长程，早前块保留为「可检索原子」；
#   长程 = 池真正溢出时，把最旧一块的稳定结论并入一条紧凑 summary。
# 注入顺序 短程原文 → 中程分段 → 长程摘要，呈漏斗（见 build_layered_memory）：
#   有 query 时中程分段按 相关性(relevance) 挑，而不是无脑最近在前 —— 早前
#   某段（如「那家厂的毛利率」）能在注入时被精准捞回，不必等折叠碰运气。
#
# 设计约束：core 不 import agents。本模块自包含 —— prompt 模板也放这；LLM 由调用方
# 注入（_rollup/fold_recent 收 llm 参数），保持可测 & 依赖方向干净。存储走 core.qa_repo。

import re
from typing import Optional

from backend.core import qa_repo
from backend.core.logger import get_logger

logger = get_logger(__name__)

# ── 三层参数 ────────────────────────────────────────────────────
SHORT_TERM_TURNS = 6      # 短程：近 N 轮原文逐字（prompt 注入侧用）
SEGMENT_TURNS    = 8      # 中程：每攒满 N 个「未覆盖」用户回合压成一块
MID_TERM_RETENTION = 12   # 中程检索池最多留几块；超出把最旧一块折入长程（扩容：早前块保留为可检索原子）
MID_TERM_INJECT    = 2    # 有 query 时每次注入的分段数（最近 1 条保连续性 + 相关性 Top）
SUMMARY_MAX_CHARS = 4000  # 原始历史总字符超预算即触发（token 维度）
SEGMENT_SUMMARY_MAX_CHARS = 800   # 分段摘要落地前截断上限；折叠、注入时也复用


# ── 分段摘要压缩 prompt（把一段未覆盖回合压成一片自包含摘要）────────
SEGMENT_SUMMARY_PROMPT = """请把下面这段对话压缩成简短的中文段落摘要，保留关键讨论主题与结论
（涉及的公司、指标、数字都要保留）。该段是独立自包含的——不给旧摘要，就按这段原始对话产出
一段能独立指引后续检索的摘要即可。

【对话片段】
{window}

只输出摘要正文，不要其它说明。"""


def build_segment_summary_prompt(window: str) -> str:
    """把一段对话窗口渲染进分段摘要模板。

    :param window: 一段未覆盖回合的格式化文本
    :return: 填入该窗口后的完整分段摘要 prompt 字符串
    """
    return SEGMENT_SUMMARY_PROMPT.format(window=window)


# ── 长程折叠 prompt（中程块超上限时把最旧一块并入全局摘要）──────────
FOLD_SEGMENT_PROMPT = """你现在维护一条「长程会话摘要」（只留稳定核心结论）。有一段「早前的分段片段
摘要」要并入它。请合并两端：去重、保留关键公司与数字结论；若旧摘要不存在就写「（无）」。
保持摘要精简，仍是一条紧凑结论，不要列流水账。

【旧长程摘要】
{old_summary}

【待并入的分段片段摘要】
{segment}

只输出合并后的长程摘要正文，不要其它说明。"""


def build_fold_prompt(old_summary: str | None, segment: str) -> str:
    """把旧长程摘要与待并入的分段片段渲染进折叠模板。

    :param old_summary: 现有长程摘要，None 时按「（无）」处理；截断到前 2000 字符
    :param segment: 待并入的分段片段摘要；截断到前 2000 字符
    :return: 填入两端的完整折叠 prompt 字符串
    """
    return FOLD_SEGMENT_PROMPT.format(
        old_summary=(old_summary or "（无）")[:2000], segment=(segment or "")[:2000])


# ── 中程分段的相关性打分（轻量、无模型：字符二元组字面重叠）────────
def _bigrams(text: str) -> frozenset[str]:
    """去空白后的字符二元组集合，用于含中文的 query/摘要字面重叠比对。

    :param text: 待切分二元组的原始文本
    :return: 该文本的字符二元组 frozenset
    """
    compact = re.sub(r"\s+", "", text or "")
    return frozenset(compact[i:i + 2] for i in range(len(compact) - 1))


def segment_scores(query: str, segments: list[dict]) -> list[int]:
    """query 与每段 summary 的相关性分（并行对齐，下标一致）。

    分 = 二元组重叠数；query 若含 6 位股票代码且段里命中，额外 +3
    （股票代码是研报语境最硬的指代信号，词面都未必出现在摘要正文里也能加分）。
    纯字面、确定性：不接模型，记忆注入是热路径，只能便宜。

    :param query: 当前问题文本，作相关性打分用的比对基准
    :param segments: 待打分的中程分段列表 [{summary, ...}]，顺序对齐返回值
    :return: 与 segments 下标一致的整数相关性分列表
    """
    qg = _bigrams(query)
    codes = re.findall(r"\d{6}", query)
    out: list[int] = []
    for seg in segments:
        text = seg.get("summary") or ""
        score = len(qg & _bigrams(text))
        if codes and any(c in text for c in codes):
            score += 3
        out.append(score)
    return out


def _choose_segments(segments: list[dict] | None, query: str | None,
                     limit: int) -> list[dict]:
    """从检索池挑要注入的分段。

    无 query：最近在前取前 limit（旧行为兼容）。有 query：先占住最近的 1 条保
    连续性，其余按 segment_scores 取相关性 Top，凑满 limit。返回仍「最近在前」的展示序。

    :param segments: 中程分段列表（最近在前），None 按空处理
    :param query: 当前问题文本；None 表示走「最近在前取前 limit」的旧行为
    :param limit: 最多挑出的分段条数；<=0 时返回空列表
    :return: 挑出的分段列表，按「最近在前」的展示序排列
    """
    segs = list(segments or [])
    if not segs or limit <= 0:
        return []
    if not query:
        return segs[:limit]
    recent = segs[:1]
    rest = segs[1:]
    scores = segment_scores(query, rest)
    ranked = sorted(zip(rest, scores), key=lambda x: x[1], reverse=True)
    picks: list[dict] = list(recent)
    for seg, _s in ranked:
        if len(picks) >= limit:
            break
        picks.append(seg)
    return picks[:limit]


# ── 分层记忆注入块：长程摘要（恒注入）+ 相关性挑出的中程分段 ───────
def build_layered_memory(segments: list[dict] | None, summary: str | None,
                         *, query: str | None = None, mid_view: int = 5) -> str:
    """把 长程（稳定核心结论）→ 中程（分段摘要）拼成一块。

    没任何分层时可返回空串（调用方传 None 表示不注入）。分层让长会话里「更早但还没被
    逐字记住」的事实有落点：短程原文只保近 SHORT_TERM_TURNS 轮，这里补齐更早到最近的
    逐块摘要 + 全局结论。query 给定时中程分段按相关性筛选（_choose_segments），
    不给则按最近在前取前 mid_view 条（兼容旧调用）。

    :param segments: 中程分段列表，None 表示不注入中程，只拼长程摘要
    :param summary: 长程摘要文本，None 表示不注入长程
    :param query: 当前问题文本；给定时中程分段按相关性筛选
    :param mid_view: 无 query 时注入的中程分段条数，默认 5
    :return: 拼接好的分层记忆文本（含长程摘要与分段块；无任何内容时返回空串）
    """
    parts: list[str] = []
    if summary:
        parts.append(f"【长程摘要（会话核心结论）】\n{summary[:800]}")
    limit = MID_TERM_INJECT if query else mid_view
    for i, seg in enumerate(_choose_segments(segments, query, limit), 1):
        text = (seg.get("summary") or "").strip()
        if text:
            parts.append(f"【早前分段摘要{i}（更早对话，供理解跨轮指代）】\n{text[:600]}")
    return "\n\n".join(parts)


# ── 存储薄包装（SQL 在 qa_repo，这里只转一层语义名）──────────────
async def load(thread_id: str) -> tuple[Optional[str], list[dict]]:
    """读会话三层记忆：(长程 summary, 中程 segments)。

    :param thread_id: 对话线程 ID
    :return: 二元组 (summary, segments)；summary 可空，segments 为
        [{seq_end, summary}] 列表（最近在前）
    """
    return await qa_repo.get_memory(thread_id)


async def save(thread_id: str, tenant_id: str, user_id: str, *,
               summary: Optional[str], segments: Optional[list] = None) -> None:
    """写回三层记忆（summary 只在有值时更新，segments 整体覆盖）。

    :param thread_id: 对话线程 ID
    :param tenant_id: 租户隔离键
    :param user_id: 用户标识
    :param summary: 长程摘要，None 表示沿用已有值不更新
    :param segments: 中程分段列表，整体覆盖，None 按空列表处理
    :return: 无返回值
    """
    await qa_repo.save_memory_blocks(thread_id, tenant_id, user_id,
                                     summary=summary, segments=segments)


# ── 压缩触发 ────────────────────────────────────────────────────
def messages_exceed_budget(messages: list[dict], max_chars: int = SUMMARY_MAX_CHARS) -> bool:
    """载入历史总字符数是否超过预算（token 维度）。预算卡的是「进过本轮 prompt 的
    原始历史」而非摘要 —— 摘要本身已是一次压缩，再按摘要长度触发没有意义。

    :param messages: 进过本轮 prompt 的原始历史消息列表 [{content, ...}]
    :param max_chars: 字符预算上限，默认取 SUMMARY_MAX_CHARS
    :return: 历史总字符数是否超过预算
    """
    total = sum(len(m.get("content", "") or "") for m in messages)
    return total > max_chars


def should_compress(uncovered_user_turns: int, raw_messages: list[dict]) -> bool:
    """两种触发任一即压：① 攒满 SEGMENT_TURNS 个新用户回合（兜底）；② 原始历史超预算。

    :param uncovered_user_turns: 游标之后未覆盖的用户回合数
    :param raw_messages: 进过本轮 prompt 的原始历史消息列表
    :return: 任一触发条件成立则为 True
    """
    return uncovered_user_turns >= SEGMENT_TURNS or messages_exceed_budget(raw_messages)


# ── 折叠流水：未覆盖回合 → 一块中程 → 超限折入长程 ──────────────
def _format_window(messages: list[dict]) -> str:
    """把一段「未覆盖回合」格式化成压缩 prompt 的输入文本（分析师/助手逐行）。

    :param messages: 未覆盖回合的消息列表 [{role, content, ...}]
    :return: 按 分析师/AI 逐行排布的多行文本；空输入返回「（空）」
    """
    lines = []
    for m in messages:
        content = m.get("content", "")
        if m["role"] == "user":
            lines.append(f"分析师：{content}")
        else:
            lines.append(f"AI：{content[:400]}")
    return "\n".join(lines) or "（空）"


async def roll_up(uncovered: list[dict], segments: list[dict],
                  summary: str | None, llm) -> tuple[list[dict], str | None]:
    """把一段未覆盖回合压成一块中程分段；块数超 MID_TERM_RETENTION 把最旧一块折入长程。

    segments 最近在前（push 头部）。llm 由调用方注入（core 不接 agents 的 get_llm）。
    返回 (segments, summary)。

    :param uncovered: 游标之后未覆盖回合的消息列表
    :param segments: 当前中程分段列表，新块插入其头部；可能被原位修改
    :param summary: 当前长程摘要，可空
    :param llm: 注入的大模型句柄（提供 ainvoke），用于生成分段摘要与折叠
    :return: 二元组 (新 segments, 新 summary)；summary 在触发折叠后才非空
    """
    block = (await llm.ainvoke(
        build_segment_summary_prompt(_format_window(uncovered)))).strip()
    max_seq = max((m["seq"] for m in uncovered), default=0)
    segments.insert(0, {"seq_end": max_seq,
                        "summary": block[:SEGMENT_SUMMARY_MAX_CHARS]})
    while len(segments) > MID_TERM_RETENTION:
        oldest = segments.pop()
        summary = (await llm.ainvoke(
            build_fold_prompt(summary, oldest["summary"]))).strip()
    logger.info("memory.rolled_up", segment_bytes=len(block),
                segments=len(segments), folded=summary is not None)
    return segments, summary


async def fold_recent(thread_id: str, *, raw_messages: list[dict],
                      summary: str | None, segments: list[dict],
                      llm) -> tuple[list[dict], str | None]:
    """三层折叠：游标 = 已覆盖最大 seq；把游标之后的「未覆盖回合」攒满 SEGMENT_TURNS 轮
    或超 token 预算时压成一块中程，超 MID_TERM_RETENTION 折入长程。未触发则原样返回（幂等写回）。

    raw_messages 是「进过本轮 prompt 的原始历史」（预算信号）；seq 窗口来自 DB 权威游标。
    返回 (segments, summary)。

    :param thread_id: 对话线程 ID，用于从 DB 取权威消息序列
    :param raw_messages: 进过本轮 prompt 的原始历史，作压缩预算信号
    :param summary: 当前长程摘要
    :param segments: 当前中程分段列表
    :param llm: 注入的大模型句柄（提供 ainvoke）
    :return: 二元组 (segments, summary)；未触发压缩时原样返回（幂等）
    """
    msgs = await qa_repo.get_messages(thread_id)
    cursor = segments[0]["seq_end"] if segments else 0
    uncovered = [m for m in msgs if m["seq"] > cursor]
    uncovered_user = sum(1 for m in uncovered if m["role"] == "user")
    if not should_compress(uncovered_user, raw_messages):
        return segments, summary
    return await roll_up(uncovered, segments, summary, llm)