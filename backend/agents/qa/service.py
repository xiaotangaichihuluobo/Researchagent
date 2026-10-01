# backend/agents/qa/service.py
# 轨道 B「问已发布研报」的顺序编排（去 LangGraph 的图拓扑，用 if/elif 表达条件边）。
#
# run_qa(state)              → 一次答完，返回最终 state（/chat 普通接口）。
# run_qa(state, streamer)    → 流式：generate 阶段逐 token 推给 streamer.token()，
#                              由 SSE 出口用 asyncio.Queue 消费（_TokenStreamer）。
# 拓扑（等价参考工程 build_qa_graph）：
#   classify → GENERAL: generate_general
#            → PRECISE: retrieve → [high]generate_rag | [low]generate_direct→enqueue
#            → VAGUE  : hyde → retrieve → …
#            → BROAD  : multi_query_rewrite → (并行)retrieve → …
#   一律 → save_memory

import asyncio

from backend.agents.qa import agentic, nodes
from backend.config import get_settings
from backend.core import usage_ctx

logger = nodes.logger


class _TokenStreamer:
    """把生成 token 推给 SSE 消费者：await token(chunk) 经 asyncio.Queue 送达外层生成器。"""

    def __init__(self, queue: asyncio.Queue):
        self._queue = queue

    async def token(self, text: str) -> None:
        await self._queue.put(("token", text))


async def run_qa(state: dict, streamer: _TokenStreamer | None = None) -> dict:
    """跑完一轮问答，返回带 answer/sources/answer_mode/confidence 的最终 state。

    引擎分派：state 带 qa_engine_mode（request 覆盖）或走配置默认。
    - agentic  → 多轮自搜循环（backend.agents.qa.agentic）
    - pipeline → 现有确定性单次检索管线（下方 _run_classic）
    两种都统一收尾：save_memory_node 落历史 + 三层记忆。
    """
    engine = state.get("qa_engine_mode") or get_settings().qa_engine_mode.lower()

    # 包一层 usage 上下文：本轮问答里的 LLM 调用都带 tenant_id/thread_id 落 llm_usage，
    # 按会话聚合成本（轨道 B 无 task_id，thread_id 即关联键）。
    async with usage_ctx.usage_ctx(tenant_id=state.get("tenant_id") or "tenant_default",
                                   thread_id=state.get("thread_id")):
        if engine == "agentic":
            await agentic.run_agentic(state, streamer)
            # 只沉淀「真检索过」的终局（有 evidence → answer_mode=agentic）；闲聊
            # general / 无证据兜底 direct 不沉淀。落桶是 fire-and-forget，不拖慢收尾。
            if state.get("answer_mode") == "agentic":
                state.update(await nodes.enqueue_sediment_node(state))
        else:
            await _run_classic(state, streamer)
    state.update(await nodes.save_memory_node(state))
    return state


async def _run_classic(state: dict, streamer: _TokenStreamer | None = None) -> None:
    """现有确定性单次检索管线（不返回值，直接改 state）。"""
    state.update(await nodes.classify_query_node(state))
    qtype = state["query_type"]

    if qtype == "GENERAL":
        state.update(await nodes.generate_general_node(state, streamer))
    else:
        if qtype == "BROAD":
            state.update(await nodes.multi_query_rewrite_node(state))
            state.update(await nodes.retrieve_node(state))
        elif qtype == "VAGUE":
            state.update(await nodes.hyde_generate_node(state))
            state.update(await nodes.retrieve_node(state))
        else:                                    # PRECISE
            state.update(await nodes.retrieve_node(state))

        if state.get("is_high_confidence", False):
            state.update(await nodes.generate_rag_node(state, streamer))
        else:
            # 低置信：开了联网就先用实时网页兜底；失败/无结果落回 llm_direct。
            # 已联网回答的问题不再入待补料队列（它不缺内容，只是不在研报语料里）。
            if state.get("enable_web_search"):
                state.update(await nodes.generate_web_node(state, streamer))
                if state.get("answer_mode") != "web":
                    state.update(await nodes.generate_direct_node(state, streamer))
                    state.update(await nodes.enqueue_pending_node(state))
            else:
                state.update(await nodes.generate_direct_node(state, streamer))
                state.update(await nodes.enqueue_pending_node(state))


async def run_qa_stream(state: dict, initial_progress: str = "理解问题中..."):
    """流式编排出口：逐个 yield (kind, payload) 事件供 SSE 转发。

    - ("progress", label)
    - ("token", text)
    - ("meta", final_state)   （流结束前最后推一次，含 answer/sources/…）
    """
    yield ("progress", initial_progress)

    queue: asyncio.Queue = asyncio.Queue()
    streamer = _TokenStreamer(queue)

    async def _producer():
        try:
            final = await run_qa(state, streamer)
        except Exception as e:                    # 顶层兜底：不中断外层生成器
            logger.error("qa.stream_error", error=str(e), exc_info=True)
            await queue.put(("__error__", str(e)))
            return
        await queue.put(("__done__", final))

    task = asyncio.get_running_loop().create_task(_producer())
    try:
        while True:
            kind, payload = await queue.get()
            if kind == "__error__":
                raise RuntimeError(payload)
            if kind == "__done__":
                yield ("meta", payload)
                return
            yield (kind, payload)
    finally:
        if not task.done():
            task.cancel()