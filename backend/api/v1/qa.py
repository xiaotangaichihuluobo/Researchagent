# backend/api/v1/qa.py
# 轨道 B「问已发布研报」的多轮问答出口：/chat（非流式）、/chat/stream（SSE 流式）、
# /sessions/{id}/history（历史）。事件帧类型与参考工程一致：progress / token / meta /
# done / error。仿 companies.py 的 current_user 用法与 research_repo 的 DB 访问。

import json

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from backend.agents.qa.service import run_qa, run_qa_stream
from backend.agents.qa.state import build_initial_state
from backend.core import profile_repo, qa_repo
from backend.core.logger import get_logger
from backend.dependencies import get_current_user

router = APIRouter()
logger = get_logger(__name__)


# ── 请求 / 响应模型 ───────────────────────────────────────────
class ChatRequest(BaseModel):
    session_id: str = Field(..., description="会话 ID（同一会话多轮累积历史）")
    message: str = Field(..., min_length=1, max_length=2000)
    enable_web_search: bool = Field(False, description="低置信时联网兜底")
    engine_mode: str | None = Field(None, description="pipeline(默认) / agentic(多轮自搜)；None=听配置")


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    answer_mode: str        # "rag" / "llm_direct" / "general"
    confidence: float
    sources: list[str]


class SessionMessage(BaseModel):
    role: str    # "user" / "assistant"
    content: str
    sources: list[str] = []   # 助手消息的参考来源；正文不再内嵌「📚 参考来源」段，前端经此字段在气泡外渲染


class SessionSummaryResponse(BaseModel):
    session_id: str
    summary: str | None
    updated_at: str | None
    turns: int


class HistoryResponse(BaseModel):
    session_id: str
    messages: list[SessionMessage]
    summary: str | None
    total_turns: int


class DeleteResponse(BaseModel):
    deleted: int


class ProfileUpdate(BaseModel):
    """显式写入跨会话用户画像（键 (tenant, user)，列表字段合并去重、不整表覆盖）。"""
    preferred_style: str | None = Field(None, description="回答风格：详实/简洁/结论先行…")
    interests: list[str] = Field(default_factory=list, description="关注话题/行业，合并去重")
    watchlist: list[str] = Field(default_factory=list, description="关注标的/公司代码池，合并去重")


class ProfileResponse(BaseModel):
    preferred_style: str | None = None
    interests: list[str] = Field(default_factory=list)
    watchlist: list[str] = Field(default_factory=list)


@router.get("/profile", response_model=ProfileResponse)
async def get_profile(current_user: dict = Depends(get_current_user)):
    """读当前用户的跨会话画像（键 tenant+user；无则返回空字段）。

    :param current_user: 当前用户信息 dict（Depends(get_current_user) 注入）；用其 tenant_id 与 user_id 定位画像。
    :return: ProfileResponse，含 preferred_style/interests/watchlist；无画像则返回各空字段。
    """
    profile = await profile_repo.get_profile(
        current_user["tenant_id"], current_user["user_id"])
    return ProfileResponse(**(profile or {}))


@router.put("/profile", response_model=ProfileResponse)
async def upsert_profile(req: ProfileUpdate,
                         current_user: dict = Depends(get_current_user)):
    """显式写入/合并当前用户的画像：列表字段 add-only 去重，preferred_style 有值才改。

    只做「显式信号」写入，不在这里做任何 LLM 提炼（低成本 + 不污染画像，
    见 docs/跨会话记忆设计.md）。写后读回并返回最新值。

    :param req: 请求体（ProfileUpdate）：preferred_style（回答风格）、interests（关注话题，合并去重 add-only）、watchlist（关注标的，合并去重 add-only）。
    :param current_user: 当前用户信息 dict（Depends(get_current_user) 注入）。
    :return: ProfileResponse，写后读回的最新画像。
    """
    await profile_repo.upsert_profile(
        current_user["tenant_id"], current_user["user_id"],
        preferred_style=req.preferred_style,
        interests=req.interests, watchlist=req.watchlist)
    profile = await profile_repo.get_profile(
        current_user["tenant_id"], current_user["user_id"])
    return ProfileResponse(**(profile or {}))


@router.get("/sessions", response_model=list[SessionSummaryResponse])
async def list_sessions(current_user: dict = Depends(get_current_user)):
    """列出当前用户的历史会话（给会话侧栏做导航；点开再取 /sessions/{id}/history）。

    :param current_user: 当前用户信息 dict（Depends(get_current_user) 注入）。
    :return: list[SessionSummaryResponse]，每项含 session_id/summary/updated_at/turns。
    """
    rows = await qa_repo.list_user_sessions(
        current_user["tenant_id"], current_user["user_id"])
    return [
        SessionSummaryResponse(
            session_id=r["session_id"],
            summary=r["summary"],
            updated_at=r["updated_at"].isoformat() if r["updated_at"] else None,
            turns=r["turns"] or 0,
        )
        for r in rows
    ]


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, current_user: dict = Depends(get_current_user)):
    """非流式问答：跑完一整轮返回最终答案（幂等落库）。

    :param req: 请求体（ChatRequest）：session_id（会话 ID）、message（用户提问，1~2000 字）、enable_web_search（低置信联网兜底，默认 False）、engine_mode（pipeline/agentic 二选一，None=听配置）。
    :param current_user: 当前用户信息 dict（Depends(get_current_user) 注入）。
    :return: ChatResponse，含 session_id/answer/answer_mode/confidence/sources。
    """
    state = await build_initial_state(
        tenant_id=current_user["tenant_id"],
        user_id=current_user["user_id"],
        session_id=req.session_id,
        message=req.message,
        enable_web_search=req.enable_web_search,
        qa_engine_mode=req.engine_mode,
    )
    final = await run_qa(state)
    return ChatResponse(
        session_id=req.session_id,
        answer=final.get("answer", ""),
        answer_mode=final.get("answer_mode", "llm_direct"),
        confidence=final.get("confidence", 0.0),
        sources=final.get("sources", []),
    )


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, current_user: dict = Depends(get_current_user)):
    """流式问答（SSE）。token 帧逐字推送正文，收尾推 meta + done。

    :param req: 请求体（ChatRequest），字段含义同 /chat。
    :param current_user: 当前用户信息 dict（Depends(get_current_user) 注入）。
    :return: EventSourceResponse 事件流：progress/token/meta/done 帧；中途异常推 error 帧提示改用普通接口。
    """
    state = await build_initial_state(
        tenant_id=current_user["tenant_id"],
        user_id=current_user["user_id"],
        session_id=req.session_id,
        message=req.message,
        enable_web_search=req.enable_web_search,
        qa_engine_mode=req.engine_mode,
    )

    async def event_generator():
        """把 run_qa_stream 输出的 (kind, payload) 转成 SSE 事件帧的生成器。

        :return: 逐条产出 {"data": JSON字符串} 的事件帧；中途异常推 error 帧后返回。
        """
        try:
            async for kind, payload in run_qa_stream(state):
                if kind == "progress":
                    yield {"data": json.dumps(
                        {"type": "progress", "stage": payload}, ensure_ascii=False)}
                elif kind == "token":
                    yield {"data": json.dumps(
                        {"type": "token", "content": payload}, ensure_ascii=False)}
                elif kind == "meta":
                    yield {"data": json.dumps({
                        "type": "meta",
                        "session_id": req.session_id,
                        "answer": payload.get("answer", ""),
                        "answer_mode": payload.get("answer_mode", "llm_direct"),
                        "confidence": payload.get("confidence", 0.0),
                        "sources": payload.get("sources", []),
                    }, ensure_ascii=False)}
        except Exception as e:
            logger.error("qa.chat_stream_error", error=str(e), exc_info=True)
            yield {"data": json.dumps(
                {"type": "error", "message": "流式输出异常，请使用普通接口重试"},
                ensure_ascii=False)}
            return
        yield {"data": json.dumps({"type": "done"})}

    return EventSourceResponse(event_generator())


@router.delete("/sessions", response_model=DeleteResponse)
async def delete_all_sessions(current_user: dict = Depends(get_current_user)):
    """删当前用户全部会话（不会删掉别人的）。

    :param current_user: 当前用户信息 dict（Depends(get_current_user) 注入）。
    :return: DeleteResponse，含 deleted（删除条数）。
    """
    n = await qa_repo.delete_all_user_sessions(
        current_user["tenant_id"], current_user["user_id"])
    return DeleteResponse(deleted=n)


@router.delete("/sessions/{session_id}", response_model=DeleteResponse)
async def delete_session(session_id: str,
                         current_user: dict = Depends(get_current_user)):
    """删单个会话（先核归属，非本人/不存在返回 404）。

    :param session_id: 会话 ID，URL 路径参数。
    :param current_user: 当前用户信息 dict（Depends(get_current_user) 注入）。
    :return: DeleteResponse，含 deleted；会话不存在或非本人返回 404。
    """
    n = await qa_repo.delete_user_session(
        current_user["tenant_id"], current_user["user_id"], session_id)
    if not n:
        raise HTTPException(status_code=404, detail="会话不存在或不属于当前用户")
    return DeleteResponse(deleted=n)


@router.get("/sessions/{session_id}/history", response_model=HistoryResponse)
async def get_session_history(session_id: str,
                              current_user: dict = Depends(get_current_user)):
    """读取该会话的消息流与滚动摘要（历史以 qa_messages 为唯一真相来源）。

    :param session_id: 会话 ID，URL 路径参数。
    :param current_user: 当前用户信息 dict（Depends(get_current_user) 注入）；用其 user_id 与 session_id 构造 thread_id 读取。
    :return: HistoryResponse，含 session_id/messages/summary/total_turns。
    """
    thread_id = qa_repo.build_thread_id(current_user["user_id"], session_id)
    msgs = await qa_repo.get_messages(thread_id)
    summary = await qa_repo.get_summary(thread_id)
    return HistoryResponse(
        session_id=session_id,
        messages=[SessionMessage(role=m["role"], content=m["content"],
                                 sources=m.get("sources") or []) for m in msgs],
        summary=summary,
        total_turns=sum(1 for m in msgs if m["role"] == "user"),
    )