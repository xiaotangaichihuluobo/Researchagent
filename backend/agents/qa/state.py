# backend/agents/qa/state.py
# 轨道 B「问已发布研报」的 QAState 形状与构造。节点直接操作 dict state（与参考工程的
# TypedDict 语义一致），这里只定义「一份干净的初始状态」长什么样、以及如何从线程历史
# 恢复它 —— 多轮历史在此是检索前改写（承重墙）的输入。

from backend.core import memory, profile_repo, qa_repo


async def build_initial_state(*, tenant_id: str, user_id: str,
                              session_id: str, message: str,
                              enable_web_search: bool = False,
                              qa_engine_mode: str | None = None) -> dict:
    """按 (user, session) 构造本次问答的初始 state。

    从 qa_messages 载入该线程既有历史 + qa_sessions 的滚动摘要，再接上本轮 user 消息。
    state["messages"] 形如 [{role, content}]，本轮 user 在末尾（classify 取其最后一条）。
    """
    thread_id = qa_repo.build_thread_id(user_id, session_id)
    prior = await qa_repo.get_messages(thread_id)
    summary, segments = await memory.load(thread_id)
    # 跨会话用户画像：以 (tenant, user) 为键、稳定偏好。只在建初始状态载入并注入
    # system 一次（常驻、不逐轮变），由 profile_repo 渲染成可读文本。
    profile = await profile_repo.get_profile(tenant_id, user_id)
    profile_text = profile_repo.build_profile_text(profile) or None
    return {
        "messages": prior + [{"role": "user", "content": message}],
        "user_id": user_id,
        "tenant_id": tenant_id,
        "session_id": session_id,
        "thread_id": thread_id,
        "existing_summary": summary,             # 长程：全局滚动摘要（稳定核心结论）
        "segments": segments,                    # 中程：分段块摘要（最近在前，{seq_end, summary}）
        "user_profile_text": profile_text,       # 跨会话用户画像（拼进 system，空则 None）
        "enable_web_search": enable_web_search,   # 低置信时可让 generate_web_node 联网兜底
        "qa_engine_mode": qa_engine_mode,         # 引擎覆盖：agentic/pipeline；None=听配置
        "course_id": None,                       # 不绑定特定语料
    }