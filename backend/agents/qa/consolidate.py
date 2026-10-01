# backend/agents/qa/consolidate.py
# 轨道 B「问已发布研报」的待补料队列消费端：把低置信问题闭环成 FAQ。
#
# 背景：低置信问题已入 knowledge_pending_queue 供教师补库参考，但此前「只进不出」。
# 这里把 status='pending' 的问题按语义聚类成一条条 FAQ（LLM 一次调用），落进
# qa_faq 表（PG 主存、可审计），可选回灌 report_corpus（report_type='faq'，
# 让它能被正常 RAG 检索），随后把对应 pending 行标记为 consumed —— 闭环。
#
# 设计取舍：
# - LLM 聚类是「尽力而为」：解析失败/空结果时退化为「每问题自成一条 FAQ」，
#   绝不因为 LLM 抽风而丢问题。
# - 回灌 report_corpus 需要 BGE 嵌入 + Milvus，走 asyncio.to_thread、逐个 try，
#   单条失败只记日志不中断（闭环主体在 PG，already 成立）。
# - 消费映射用「规范化子串匹配」：FAQ 的问题与某 pending 问题归一化后互为子串
#   即视为已覆盖。归一化只去空白，跨行/近义的只能覆盖代表问，旁支留下轮再闭环。

import asyncio
import hashlib
import re
from datetime import datetime, timezone

from backend.core import qa_repo
from backend.core.logger import get_logger

logger = get_logger(__name__)

BATCH_SIZE = 50   # 一次 LLM 调用最多纳入的问题条数，避免撑爆单条 prompt


# ── 聚类 prompt：把一批近似问题合并成一条 FAQ，Q/A 成块输出 ──────
FAQ_CLUSTER_PROMPT = """你是投研知识库整理助手。下面是一批用户问过、但研报语料原本答不上
来的问题。请把【近似重复/同一主题】的问题合并成一条 FAQ，并基于你的通用投研知识补一个简洁、
准确的回答。

要求：
- 每一条合并结果占一个块，块内两行，严格用以下前缀（用于解析，别写其它文字）：
  Q=<合并后的问题文本>（直接复用其中一个原问题原文，别新造）
  A=<简洁回答，2-4 句，给关键数字/结论>
- 块之间用空行分隔；不同主题的问题各自成块。
- 无法归并的单独问题也各自成一条。

【这批问题】
{questions}"""


def _normalize(s: str) -> str:
    """去掉所有空白做归一化，用于问题文本的近似匹配。

    :param s: 原始文本（str）。
    :return: str，去除所有空白字符后的归一化文本。
    """
    return re.sub(r"\s+", "", s or "")


def _matches(pending_q: str, faq_q: str) -> bool:
    """pending 问题与 FAQ 问题归一化后互为子串 → 视为已被该 FAQ 覆盖。

    :param pending_q: 待补料队列里的问题文本（str）。
    :param faq_q: FAQ 的问题文本（str）。
    :return: bool，任一为空白返回 False；归一化后互为子串返回 True。
    """
    a, b = _normalize(pending_q), _normalize(faq_q)
    if not a or not b:
        return False
    return a in b or b in a


def _parse_blocks(text: str) -> list[dict]:
    """把 LLM 输出解析成 [{question, answer}]。容忍 Q/A/Q= 中的中文与英文冒号。

    :param text: LLM 输出的原始文本（str）。
    :return: list[dict]，每项含 question 与 answer（两者皆有才保留）。
    """
    out: list[dict] = []
    cur: dict | None = None
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if m := re.match(r"^(?:问题|Q)[:：=]\s*(.*)$", line, re.I):
            if cur:
                out.append(cur)
            cur = {"question": m.group(1).strip(), "answer": ""}
            continue
        if (m := re.match(r"^(?:回答|A)[:：=]\s*(.*)$", line, re.I)) and cur is not None:
            cur["answer"] = (cur["answer"] + ("\n" if cur["answer"] else "") +
                             m.group(1).strip())
    if cur:
        out.append(cur)
    return [b for b in out if b.get("question") and b.get("answer")]


async def _ingest_faqs_to_corpus(tenant_id: str, faqs: list[dict]) -> int:
    """把 FAQ 逐条回灌 report_corpus（report_type='faq'），可被正常 RAG 检索。

    每条 FAQ 单独一份合成研报（内容 = 问题 + 回答），report_key 用 md5 保证幂等。
    骨架写入复用 report_ingest.ingest_report；失败只记日志，不阻断清点。

    :param tenant_id: 租户 ID（str），FAQ 回灌进该租户的 report_corpus。
    :param faqs: FAQ 列表，每项含 question 与 answer。
    :return: int，成功回灌的条数。
    """
    from backend.core.report_ingest import ingest_report

    written = 0
    for faq in faqs:
        report_key = "faq:" + hashlib.md5(
            _normalize(faq["question"]).encode()).hexdigest()[:16]
        content = f"【FAQ】{faq['question']}\n{faq['answer']}"
        try:
            await ingest_report(
                tenant_id=tenant_id,
                report_key=report_key,
                title=faq["question"][:60],
                content=content,
                company_code="faq",
                industry="",
                report_type="faq",
                published_at=datetime.now(timezone.utc),
            )
            written += 1
        except Exception as e:
            logger.warning("qa.faq_corpus_ingest_failed",
                           question=faq["question"][:50], error=str(e))
    return written


async def consolidate_pending_faq(tenant_id: str, *, llm=None,
                                  ingest: bool = False) -> dict:
    """消费某租户 status='pending' 的问题并闭环成 FAQ。

    - llm:  可注入的 LLM 句柄（要求有 ainvoke）。缺省取 get_llm("qa_reports", t=0.2)。
    - ingest: True 时额外把 FAQ 回灌进 Milvus report_corpus（真实跑脚本才开，测试关）。

    返回 {pending_seen, faqs_added, consumed, left_pending}。

    :param tenant_id: 租户 ID（str），只消费该租户 status="pending" 的问题。
    :param llm: 可注入的 LLM 句柄（须有 ainvoke），缺省用 get_llm("qa_reports", temperature=0.2)。
    :param ingest: 是否额外把 FAQ 回灌 Milvus report_corpus（默认 False，测试关闭、真实跑脚本才开）。
    :return: dict，结构 {"pending_seen": int, "faqs_added": int, "consumed": int, "left_pending": int}。
    """
    pending = await qa_repo.list_pending(tenant_id)
    if not pending:
        return {"pending_seen": 0, "faqs_added": 0, "consumed": 0, "left_pending": 0}

    questions = [p["question"] for p in pending]
    if llm is None:
        from backend.core.llm_factory import get_llm
        llm = get_llm("qa_reports", temperature=0.2)

    all_blocks: list[dict] = []
    for i in range(0, len(questions), BATCH_SIZE):
        chunk = questions[i:i + BATCH_SIZE]
        prompt = FAQ_CLUSTER_PROMPT.format(
            questions="\n".join(f"{n}. {q}" for n, q in enumerate(chunk, 1)))
        raw = (await llm.ainvoke(prompt)).strip()
        blocks = _parse_blocks(raw)
        if not blocks:
            # 兜底：LLM 没给出可解析的块 → 每问题自成一条，不丢问题
            blocks = [{"question": q, "answer": "（待复核后补充）"} for q in chunk]
        all_blocks.extend(blocks)

    # 去重 FAQ（同问题只按一条处理）
    seen: set[str] = set()
    faqs: list[dict] = []
    for b in all_blocks:
        norm = _normalize(b["question"])
        if norm and norm not in seen:
            seen.add(norm)
            faqs.append(b)

    consumed = 0
    if faqs:
        faq_norms = [_normalize(f["question"]) for f in faqs]
        consumed_qs = [q for q in questions if any(_matches(q, fq) for fq in faq_norms)]
        for b in faqs:
            try:
                await qa_repo.add_faq(tenant_id, b["question"], b["answer"], 1)
            except Exception as e:
                logger.warning("qa.faq_add_failed",
                               question=b["question"][:50], error=str(e))
        if ingest:
            await _ingest_faqs_to_corpus(tenant_id, faqs)
        consumed = await qa_repo.mark_pending_consumed(tenant_id, consumed_qs)

    return {"pending_seen": len(questions), "faqs_added": len(faqs),
            "consumed": consumed, "left_pending": len(questions) - consumed}


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(consolidate_pending_faq("tenant_default", ingest=False))