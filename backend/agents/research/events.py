# backend/agents/research/events.py
# 研报任务完成事件的进程内发布/订阅 —— 后台流水线做完了，主动推给 SSE 订阅者，
# 前端不再靠 5 秒轮询等结果（轮询仍保留作断线兜底，这是纯增量的广播）。
#
# 订阅按 tenant 隔离：一个租户的事件只进那个租户的订阅者队列。进程重启即失联，
# 丢的事件由前端下次轮询 /tasks/{id} 兜回 —— 广播是「省一次轮询」的优化，
# 不是「状态唯一真相」（真相在 PG 业务表，跑得了的以 DB 为准）。

import asyncio
from collections import defaultdict

_subscribers: dict[str, set[asyncio.Queue]] = defaultdict(set)

# outcome（_drive 的返回）→ 对外暴露的 research_tasks.status。
_OUTCOME_TO_STATUS = {
    "published": "published",
    "rejected": "rejected",
    "failed": "failed",
    "paused": "awaiting_risk_review",   # 停在风控待审，研究员侧可见的状态变了
}


def subscribe(tenant_id: str) -> asyncio.Queue:
    """订阅某租户的事件流，返回一个 asyncio.Queue；断开时须 unsubscribe。

    :param tenant_id: 要订阅的租户，事件按租户隔离。
    :return: 承载该租户推送事件的 asyncio.Queue。
    """
    q: asyncio.Queue = asyncio.Queue()
    _subscribers[tenant_id].add(q)
    return q


def unsubscribe(tenant_id: str, q: asyncio.Queue) -> None:
    """取消订阅：从某租户的订阅者集合中移除指定队列。

    :param tenant_id: 订阅时用到的租户。
    :param q: 之前 subscribe 返回的 asyncio.Queue。
    :return: 无返回值。
    """
    _subscribers[tenant_id].discard(q)


def publish(tenant_id: str, event: dict) -> None:
    """向某租户的所有订阅者广播一个事件。订阅者队列既不阻塞也不丢。

    :param tenant_id: 事件归属的租户。
    :param event: 要广播的事件 dict（通常来自 outcome_event）。
    :return: 无返回值。
    """
    for q in list(_subscribers.get(tenant_id, ())):
        q.put_nowait(event)


def outcome_event(task_id: str, outcome: str) -> dict:
    """把流水线 outcome 转成一个前端可消费的 {type, task_id, status} 事件。

    :param task_id: 任务唯一标识。
    :param outcome: 流水线 outcome（published/rejected/failed/paused），
        paused 会被映射为对外状态 awaiting_risk_review。
    :return: {"type": "research_update", "task_id": ..., "status": ...} 事件 dict。
    """
    return {
        "type": "research_update",
        "task_id": task_id,
        "status": _OUTCOME_TO_STATUS.get(outcome, outcome),
    }