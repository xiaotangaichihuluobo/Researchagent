# backend/core/qa_repo.py
# 轨道 B「问已发布研报」的会话/消息持久化仓储。读写全收口在这里（仿 research_repo）。
#
# 与 research_repo 的分工：投研流水线是「一次任务一段」的状态机；这里的问答是
# 「一个 thread 多轮累积」的对话。多轮历史以 qa_messages 为唯一真相来源
# （重启可读 /sessions/{id}/history），qa_sessions 只存滚动摘要（summary）。

import json
import uuid
from typing import Any, Optional

from sqlalchemy import text

from backend.dependencies import AsyncSessionLocal
from backend.core.logger import get_logger

logger = get_logger(__name__)


def _row_to_dict(row) -> dict:
    """将 SQLAlchemy 行对象转成普通字典。

    :param row: SQLAlchemy Row 对象（带 _mapping 属性）
    :return: 行数据的字典表示
    """
    return dict(row._mapping)


def build_thread_id(user_id: str, session_id: str) -> str:
    """按 (用户, 会话) 唯一标识一个对话线程。照搬参考实现，多轮历史靠它隔离。

    :param user_id: 用户标识，拼进线程键前缀
    :param session_id: 会话标识，拼进线程键后缀
    :return: 形如 user_{user_id}_session_{session_id} 的线程 ID 字符串
    """
    return f"user_{user_id}_session_{session_id}"


async def get_messages(thread_id: str, limit: int = 100) -> list[dict]:
    """按出现顺序取该线程的历史消息：[{role, content, seq}]。seq 供分层压缩的游标用。

    :param thread_id: 对话线程 ID（build_thread_id 生成）
    :param limit: 最多返回的历史消息条数，默认 100
    :return: 消息列表，每项含 role/content/seq 三个键，按 seq 升序排列
    """
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("SELECT role, content, seq FROM qa_messages"
                 " WHERE thread_id = :tid ORDER BY seq ASC LIMIT :lim"),
            {"tid": thread_id, "lim": limit},
        )
        return [{"role": r[0], "content": r[1], "seq": r[2]} for r in result.all()]


async def delete_user_session(tenant_id: str, user_id: str,
                              session_id: str) -> int:
    """删单个会话（先校验归属再删消息+会话行）。返回删除的会话行数（0=不存在/非本人）。

    qa_messages 只在 qa_sessions 有 thread_id 这一个键，无外键级联，需手动先删消息。

    :param tenant_id: 租户隔离键，用于校验会话归属
    :param user_id: 用户标识，用于校验归属，防止误删他人会话
    :param session_id: 会话标识，与 user_id 一起定位目标线程
    :return: 删除的会话行数；0 表示不存在或非本人所有
    """
    thread_id = build_thread_id(user_id, session_id)
    async with AsyncSessionLocal() as session:
        owned = (await session.execute(
            text("SELECT 1 FROM qa_sessions WHERE thread_id = :t"
                 "  AND tenant_id = :ten AND user_id = :uid"),
            {"t": thread_id, "ten": tenant_id, "uid": user_id},
        )).fetchone()
        if not owned:
            return 0
        await session.execute(
            text("DELETE FROM qa_messages WHERE thread_id = :t"), {"t": thread_id})
        n = (await session.execute(
            text("DELETE FROM qa_sessions WHERE thread_id = :t"), {"t": thread_id})).rowcount
        await session.commit()
    return n or 0


async def delete_all_user_sessions(tenant_id: str, user_id: str) -> int:
    """删该用户全部会话（消息 + 会话行）。返回删除的会话行数。

    :param tenant_id: 租户隔离键，用于限定删除范围
    :param user_id: 用户标识，仅删该用户的会话
    :return: 删除的会话行数；0 表示没有可删的会话
    """
    async with AsyncSessionLocal() as session:
        n = (await session.execute(
            text("DELETE FROM qa_messages WHERE thread_id IN"
                 " (SELECT thread_id FROM qa_sessions"
                 "  WHERE tenant_id = :ten AND user_id = :uid)"),
            {"ten": tenant_id, "uid": user_id},
        )).rowcount
        m = (await session.execute(
            text("DELETE FROM qa_sessions"
                 " WHERE tenant_id = :ten AND user_id = :uid"),
            {"ten": tenant_id, "uid": user_id},
        )).rowcount
        await session.commit()
    return m or 0


async def list_user_sessions(tenant_id: str, user_id: str,
                             limit: int = 100) -> list[dict]:
    """列某用户在 qa_sessions 里留下的会话（按更新时间倒序）。

    返回 [{session_id, summary, updated_at, turns}]。thread_id 形如
    user_{user_id}_session_{session_id}，过滤 (tenant_id, user_id) 即把
    别人的会话隔离掉；session_id 从 thread_id 尾段剥出，turns 用子查询数 user 消息。

    :param tenant_id: 租户隔离键，用于过滤会话
    :param user_id: 用户标识，仅列出该用户的会话
    :param limit: 最多返回的会话条数，默认 100
    :return: 会话列表，每项含 session_id / summary / updated_at / turns 四个键
    """
    async with AsyncSessionLocal() as session:
        result = await session.execute(text("""
            SELECT
                split_part(s.thread_id, '_session_', 2)  AS session_id,
                s.summary,
                s.updated_at,
                (SELECT COALESCE(SUM(CASE WHEN role = 'user' THEN 1 ELSE 0 END), 0)
                 FROM qa_messages m WHERE m.thread_id = s.thread_id) AS turns
            FROM qa_sessions s
            WHERE s.tenant_id = :t AND s.user_id = :uid
            ORDER BY s.updated_at DESC
            LIMIT :lim
        """), {"t": tenant_id, "uid": user_id, "lim": limit})
        return [dict(r._mapping) for r in result.all()]


async def get_summary(thread_id: str) -> Optional[str]:
    """读会话滚动摘要；没有则返回 None。

    :param thread_id: 对话线程 ID
    :return: 会话的滚动摘要字符串；若该会话无摘要则返回 None
    """
    async with AsyncSessionLocal() as session:
        row = (await session.execute(
            text("SELECT summary FROM qa_sessions WHERE thread_id = :tid"),
            {"tid": thread_id},
        )).fetchone()
    return row[0] if row else None


async def get_memory(thread_id: str) -> tuple[Optional[str], list[dict]]:
    """一次读回三层记忆：[0]=长程摘要(summary)，[1]=中程分段(segments，最近在前)。

    segments 是 qa_sessions JSONB 列，存 [{seq_end, summary}]——seq_end 是那块覆盖到的
    最大消息 seq（压缩游标：已覆盖到哪），summary 是块摘要。

    :param thread_id: 对话线程 ID
    :return: 二元组 (长程摘要, 中程分段)。长程摘要为 None 表示无；中程分段为
         [{seq_end, summary}] 列表，最近在前，无则返回空列表
    """
    async with AsyncSessionLocal() as session:
        row = (await session.execute(
            text("SELECT summary, segments FROM qa_sessions WHERE thread_id = :tid"),
            {"tid": thread_id},
        )).fetchone()
    if not row:
        return None, []
    segments = row[1] or []
    if isinstance(segments, str):                 # 防御：个别驱动把 jsonb 当字符串回
        import json
        try:
            segments = json.loads(segments)
        except Exception:
            segments = []
    return row[0], segments or []


async def append_messages(thread_id: str, user_content: str,
                          assistant_content: str) -> None:
    """把一轮「用户问 + 助手答」追加进消息流，seq 递增保证顺序。

    :param thread_id: 对话线程 ID
    :param user_content: 用户本轮提问原文
    :param assistant_content: 助手本轮回答原文
    :return: 无返回值
    """
    async with AsyncSessionLocal() as session:
        next_seq = int((await session.execute(
            text("SELECT COALESCE(MAX(seq), 0) FROM qa_messages WHERE thread_id = :tid"),
            {"tid": thread_id},
        )).scalar_one()) + 1
        await session.execute(
            text("INSERT INTO qa_messages (thread_id, role, content, seq)"
                 " VALUES (:tid, 'user', :u, :seq0), (:tid, 'assistant', :a, :seq1)"),
            {"tid": thread_id, "u": user_content, "a": assistant_content,
             "seq0": next_seq, "seq1": next_seq + 1},
        )
        await session.commit()


async def save_memory_blocks(thread_id: str, tenant_id: str,
                             user_id: str, *, summary: Optional[str],
                             segments: Optional[list] = None) -> None:
    """UPSERT 会话行，同时写 长程(summary) 与 中程(segments)。幂等按 thread_id 冲突。

    - summary 只在非空时更新（沿用 COALESCE）：本轮没触发折叠就不动长程。
    - segments 整体覆盖（分层块的当前形态即权威值）。

    :param thread_id: 对话线程 ID，用于定位要 upsert 的会话行
    :param tenant_id: 租户隔离键，写入会话行的归属
    :param user_id: 用户标识，写入会话行的归属
    :param summary: 长程摘要；None 时沿用已有值不更新（COALESCE 语义）
    :param segments: 中程分段列表 [{seq_end, summary}]，整体覆盖，None 按空列表处理
    :return: 无返回值
    """
    import json as _json
    async with AsyncSessionLocal() as session:
        await session.execute(
            text("""
                INSERT INTO qa_sessions
                    (id, tenant_id, user_id, thread_id, summary, segments, summary_version)
                VALUES (:id, :tenant_id, :user_id, :thread_id, :summary, :seg, 1)
                ON CONFLICT (thread_id) DO UPDATE
                    SET summary         = COALESCE(EXCLUDED.summary, qa_sessions.summary),
                        segments        = COALESCE(EXCLUDED.segments, qa_sessions.segments),
                        summary_version = qa_sessions.summary_version + 1,
                        updated_at      = NOW()
            """),
            {"id": str(uuid.uuid4()), "tenant_id": tenant_id,
             "user_id": user_id, "thread_id": thread_id,
             "summary": summary, "seg": _json.dumps(segments or [], ensure_ascii=False)},
        )
        await session.commit()


async def enqueue_pending(tenant_id: str, user_id: str, question: str,
                          confidence: float) -> None:
    """低置信度问题入队供教师补库参考。幂等（同问题同租户只留一行）。

    :param tenant_id: 租户隔离键
    :param user_id: 提出该问题的用户标识
    :param question: 待补库的问题文本
    :param confidence: 低置信度分值，入队留档参考
    :return: 无返回值
    """
    async with AsyncSessionLocal() as session:
        await session.execute(
            text("""
                INSERT INTO knowledge_pending_queue
                    (id, tenant_id, question, user_id, confidence, status)
                VALUES (:id, :tenant_id, :question, :user_id, :confidence, 'pending')
                ON CONFLICT DO NOTHING
            """),
            {"id": str(uuid.uuid4()), "tenant_id": tenant_id,
             "user_id": user_id, "question": question, "confidence": confidence},
        )
        await session.commit()


async def list_pending(tenant_id: str, limit: int = 200) -> list[dict]:
    """拉某租户 status='pending' 的去重问题（供 FAQ 闭环消费）。

    返回 [{id, question, confidence, user_id}]。去重后仍取最早入队的行。

    :param tenant_id: 租户隔离键，仅拉该租户的待处理问题
    :param limit: 最多返回的问题条数，默认 200
    :return: 去重后的问题列表，每项含 id / question / confidence / user_id 四个键
    """
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("""
                SELECT DISTINCT ON (question) id, question, confidence, user_id
                FROM knowledge_pending_queue
                WHERE tenant_id = :t AND status = 'pending'
                ORDER BY question, created_at ASC
                LIMIT :lim
            """),
            {"t": tenant_id, "lim": limit},
        )
        return [dict(r._mapping) for r in result.all()]


async def mark_pending_consumed(tenant_id: str, questions: list[str]) -> int:
    """把已闭环成 FAQ 的 pending 行标为 consumed（回写，闭环）。

    :param tenant_id: 租户隔离键，仅更新该租户的行
    :param questions: 已消费的问题文本列表（可空，空则直接返回 0）
    :return: 实际被更新的行数
    """
    if not questions:
        return 0
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("""
                UPDATE knowledge_pending_queue
                SET status = 'consumed'
                WHERE tenant_id = :t AND question = ANY(:q) AND status = 'pending'
            """),
            {"t": tenant_id, "q": questions},
        )
        await session.commit()
    return (result.rowcount or 0)


async def enqueue_sediment(tenant_id: str, user_id: str, question: str,
                           answer: str, sources: str) -> None:
    """agentic 高质量答案入沉淀桶。幂等（同问题同租户只留一行，已有则静默）。

    :param tenant_id: 租户隔离键
    :param user_id: 提出该问题的用户标识
    :param question: 问题文本
    :param answer: agentic 生成的高质量答案文本
    :param sources: 答案引用的来源信息（可读字符串）
    :return: 无返回值
    """
    async with AsyncSessionLocal() as session:
        await session.execute(
            text("""
                INSERT INTO qa_sediment_queue
                    (id, tenant_id, question, answer, sources, user_id, status)
                VALUES (:id, :tenant_id, :question, :answer, :sources, :user_id, 'pending')
                ON CONFLICT DO NOTHING
            """),
            {"id": str(uuid.uuid4()), "tenant_id": tenant_id,
             "question": question, "answer": answer, "sources": sources,
             "user_id": user_id},
        )
        await session.commit()


async def list_sediment_pending(tenant_id: str, limit: int = 200) -> list[dict]:
    """拉某租户 status='pending' 的待沉淀答案（供离线脚本双落消费）。

    返回 [{id, question, answer, sources}]。去重后仍取最早入桶的行。

    :param tenant_id: 租户隔离键，仅拉该租户的待沉淀答案
    :param limit: 最多返回的答案条数，默认 200
    :return: 去重后的答案列表，每项含 id / question / answer / sources 四个键
    """
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("""
                SELECT DISTINCT ON (question) id, question, answer, sources
                FROM qa_sediment_queue
                WHERE tenant_id = :t AND status = 'pending'
                ORDER BY question, created_at ASC
                LIMIT :lim
            """),
            {"t": tenant_id, "lim": limit},
        )
        return [dict(r._mapping) for r in result.all()]


async def mark_sediment_consumed(tenant_id: str, ids: list[str]) -> int:
    """把已双落的沉淀桶行标为 consumed（回写，闭环）。

    :param tenant_id: 租户隔离键，仅更新该租户的行
    :param ids: 已处理的行 id 列表（可空，空则直接返回 0）
    :return: 实际被更新的行数
    """
    if not ids:
        return 0
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("""
                UPDATE qa_sediment_queue
                SET status = 'consumed'
                WHERE tenant_id = :t AND id = ANY(CAST(:ids AS uuid[]))
                  AND status = 'pending'
            """),
            {"t": tenant_id, "ids": ids},
        )
        await session.commit()
    return (result.rowcount or 0)


async def add_faq(tenant_id: str, question: str, answer: str,
                  source_count: int = 1) -> None:
    """落一条 FAQ（幂等：同租户同问题重复消费只留一行，answer 更新）。

    :param tenant_id: 租户隔离键
    :param question: FAQ 问题文本
    :param answer: FAQ 标准答案文本
    :param source_count: 支撑该答案的来源条数，默认 1
    :return: 无返回值
    """
    async with AsyncSessionLocal() as session:
        await session.execute(
            text("""
                INSERT INTO qa_faq (tenant_id, question, answer, source_count)
                VALUES (:t, :q, :a, :c)
                ON CONFLICT (tenant_id, question) DO UPDATE
                    SET answer       = EXCLUDED.answer,
                        source_count = EXCLUDED.source_count,
                        status       = 'active'
            """),
            {"t": tenant_id, "q": question, "a": answer, "c": source_count},
        )
        await session.commit()


async def list_faq(tenant_id: str, limit: int = 100) -> list[dict]:
    """列已消费的 FAQ（教师面板 / 审计用）。

    :param tenant_id: 租户隔离键，仅列出该租户的已激活 FAQ
    :param limit: 最多返回的 FAQ 条数，默认 100
    :return: FAQ 列列表，每项含 question / answer / source_count / created_at
    """
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("SELECT question, answer, source_count, created_at"
                 " FROM qa_faq WHERE tenant_id = :t AND status = 'active'"
                 " ORDER BY created_at DESC LIMIT :lim"),
            {"t": tenant_id, "lim": limit},
        )
        return [dict(r._mapping) for r in result.all()]


async def add_dead_letter(tenant_id: str, channel: str, thread_id: str,
                          payload: dict, error: str) -> None:
    """把问答落库/入队的失败载荷写进死信表，供人工复核（F 铁律：载荷不丢）。

    :param tenant_id: 租户隔离键
    :param channel: 失败发生的通道名（用于区分是落库还是入队）
    :param thread_id: 关联的对话线程 ID
    :param payload: 失败的原始载荷字典，会序列化为 jsonb 存储
    :param error: 失败原因文本，截断到前 2000 字符
    :return: 无返回值
    """
    async with AsyncSessionLocal() as session:
        await session.execute(
            text("""
                INSERT INTO qa_dead_letter (tenant_id, channel, thread_id, payload, error)
                VALUES (:t, :c, :tid, CAST(:payload AS jsonb), :err)
            """),
            {"t": tenant_id, "c": channel, "tid": thread_id,
             "payload": json.dumps(payload, ensure_ascii=False), "err": str(error)[:2000]},
        )
        await session.commit()


async def _purge_tenant(tenant_id: str) -> int:
    """测试用：清掉某租户的会话数据（qa_sessions 只按 thread_id 有唯一键）。

    :param tenant_id: 要清空的租户标识
    :return: 删除的消息行数
    """
    async with AsyncSessionLocal() as session:
        n = (await session.execute(
            text("DELETE FROM qa_messages WHERE thread_id IN"
                 " (SELECT thread_id FROM qa_sessions WHERE tenant_id = :t)"),
            {"t": tenant_id},
        )).rowcount
        await session.execute(
            text("DELETE FROM qa_sessions WHERE tenant_id = :t"), {"t": tenant_id})
        await session.execute(
            text("DELETE FROM knowledge_pending_queue WHERE tenant_id = :t"),
            {"t": tenant_id})
        await session.commit()
    return n or 0