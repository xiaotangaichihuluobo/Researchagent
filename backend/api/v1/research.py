# backend/api/v1/research.py
# 研究任务接口：提交式 + 轮询（设计文档 §8.3 / §8.5）+ SSE 完成推送（E：替代轮询等结果）

import asyncio
import json
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from backend.core.logger import get_logger
from backend.core import research_repo as repo
from backend.agents.research.runner import start_pipeline
from backend.agents.research import events as events_hub
from backend.agents.risk.nodes import (
    MIN_COMMENTS_LEN, REDO_STAGES, VALID_DECISIONS,
)
from backend.dependencies import get_current_user, require_role

router = APIRouter()
logger = get_logger(__name__)

# 终态：前端看到这几个状态就停止轮询。
# awaiting_risk_review / approved 都是**非终态** —— 前端必须继续轮询，
# 否则审核人签完字它就再也看不到「已发布」了。
TERMINAL_STATUSES = ("published", "rejected", "failed")


class TaskCreate(BaseModel):
    """提交研究任务的请求体。一次任务 = 一个标的（设计文档 §1.3）。"""
    company_code: str = Field(..., min_length=2, max_length=32, description="标的代码")


@router.post("/tasks", status_code=status.HTTP_202_ACCEPTED)
async def create_task(
    req: TaskCreate,
    # researcher 是本动作的职责角色；admin 是超集角色，一并放行。
    # 不放 risk_control：风控若也能发起研究，就等于自己提案自己审 —— 职责分离的前提是提交方与审核方不是同一个人。
    current_user: dict = Depends(require_role("researcher", "admin")),
):
    """提交研究任务，返回 202 与 task_id；流水线在后台跑。

    :param req: 请求体（TaskCreate），其中 company_code 为标的代码（2~32 字符，Body 必填）。
    :param current_user: 当前用户信息 dict（由 Depends(require_role("researcher","admin")) 注入），仅 researcher/admin 可发起；其 tenant_id 定位租户、user_id 记为创建人。
    :return: dict，结构 {"task_id": str, "status": "pending", "thread_id": str}；成功返回 HTTP 202 Accepted。
    """
    tenant_id = current_user["tenant_id"]

    company = await repo.get_company_by_code(tenant_id, req.company_code)
    if not company:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="标的不存在")

    thread_id = f"research-{uuid.uuid4()}"
    task_id = await repo.create_task(
        tenant_id=tenant_id,
        company_id=str(company["id"]),
        created_by=current_user["user_id"],
        thread_id=thread_id,
    )
    await repo.write_audit_log(tenant_id, current_user["user_id"],
                               "research_task.create", "research_task", task_id,
                               {"company_code": req.company_code})

    start_pipeline({
        "task_id": task_id, "tenant_id": tenant_id,
        "company_id": str(company["id"]), "company_code": company["code"],
        "industry": company.get("industry") or "", "thread_id": thread_id,
        "redo_count": 0,
    })
    return {"task_id": task_id, "status": "pending", "thread_id": thread_id}


@router.get("/tasks")
async def list_tasks(
    limit:  int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    # status 过滤是 P6 加的：风控待审列表只要 awaiting_risk_review 那几条。
    # 不校验取值是否合法：这不是一个「改任务状态」的入口，传个不存在的状态只会
    # 返回空列表，比 422 更符合「列表查询」的语义。日后再加枚举校验也不迟。
    status_filter: Optional[str] = Query(None, alias="status", max_length=32),
    current_user: dict = Depends(get_current_user),
):
    """任务列表。研究员只看自己的；风控与管理员看全租户。

    :param limit: 返回条数上限，URL Query 参数，默认 20，范围 1~100。
    :param offset: 分页偏移量，URL Query 参数，默认 0。
    :param status_filter: 状态过滤（URL Query 参数，别名为 status，最大 32 字符）；如 awaiting_risk_review，传不存在的状态返回空列表。
    :param current_user: 当前用户信息 dict（Depends(get_current_user) 注入）；role 为 researcher 时只看自己的任务，否则看全租户。
    :return: dict，结构 {"items": list, "limit": int, "offset": int}。
    """
    tenant_id = current_user["tenant_id"]
    created_by = current_user["user_id"] if current_user["role"] == "researcher" else None
    rows = await repo.list_tasks(tenant_id, created_by=created_by, status=status_filter,
                                 limit=limit, offset=offset)
    return {"items": rows, "limit": limit, "offset": offset}


@router.get("/tasks/{task_id}")
async def get_task(
    # 声明成 uuid.UUID 而不是 str：畸形路径参数由 FastAPI 在进入函数体前拦下并返回 422。
    # 若留成 str 直接透传给 repo.get_task()，SQL 的 WHERE id = :id 作用在 UUID 列上，
    # asyncpg 会抛 DataError: invalid UUID，而 main.py 没注册异常处理器 ⇒ HTTP 500。
    task_id: uuid.UUID,
    current_user: dict = Depends(get_current_user),
):
    """任务详情（状态 + 各阶段事件），供前端 5 秒一次轮询。

    :param task_id: 任务 UUID，来自 URL 路径参数 /tasks/{task_id}；畸形值由 FastAPI 在进入函数体前返回 422。
    :param current_user: 当前用户信息 dict（Depends(get_current_user) 注入）；研究员只能看自己的任务（否则 403），跨租户一律 404。
    :return: dict，含 task_id/status/current_stage/redo_count/last_error/is_terminal/result_available/stages。
    """
    tenant_id = current_user["tenant_id"]
    row = await repo.get_task(tenant_id, task_id)
    if not row:
        # 跨租户返回 404 而不是 403：403 会泄漏"这个 task_id 存在"
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")

    if (current_user["role"] == "researcher"
            and str(row["created_by"]) != current_user["user_id"]):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="权限不足")

    events = await repo.list_stage_events(tenant_id, task_id)
    report = await repo.get_report(tenant_id, task_id)

    return {
        "task_id":          str(row["id"]),
        "status":           row["status"],
        "current_stage":    row["current_stage"],
        "redo_count":       row["redo_count"],
        "last_error":       row["last_error"],
        "is_terminal":      row["status"] in TERMINAL_STATUSES,   # 前端据此停止轮询
        "result_available": report is not None and report["status"] == "published",
        "stages": [
            {"stage": e["stage"], "status": e["status"],
             "detail": e["detail"], "occurred_at": e["occurred_at"]}
            for e in events
        ],
    }


@router.get("/tasks/{task_id}/events")
async def list_task_events(
    # 同上：uuid.UUID 让畸形输入变成 422 而不是运行期的 asyncpg DataError → 500
    task_id: uuid.UUID,
    current_user: dict = Depends(get_current_user),
):
    """阶段进度事件（轮询用）。只追加、可跨阶段，是审计入口。

    :param task_id: 任务 UUID，URL 路径参数 /tasks/{task_id}/events；畸形输入返回 422。
    :param current_user: 当前用户信息 dict（Depends(get_current_user) 注入）；跨租户返回 404。
    :return: dict，结构 {"events": list}。
    """
    tenant_id = current_user["tenant_id"]
    row = await repo.get_task(tenant_id, task_id)
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")
    events = await repo.list_stage_events(tenant_id, task_id)
    return {"events": events}


# ── SSE 完成推送（E）────────────────────────────────────────────
# 后台流水线在发布/驳回/失败/待审时向本租户广播 research_update 事件，
# 前端订阅这里即可及时刷新，不必等 5 秒轮询。轮询（/tasks/{id}）仍保留作断线兜底：
# 广播是进程内内存态，进程重启即失联，DB 里的 status 才是真相。

@router.get("/events")
async def stream_research_events(
    current_user: dict = Depends(require_role("researcher", "admin")),
):
    """任务完成/状态变更事件的 SSE 流。连接即收，断开即退订。

    :param current_user: 当前用户信息 dict（Depends(require_role("researcher","admin")) 注入）；其 tenant_id 决定订阅的事件范围。
    :return: EventSourceResponse 事件流：握手先发 {"type":"connected"}，随后按租户订阅实时推送 research_update 事件，客户端断开即退订。
    """
    tenant_id = current_user["tenant_id"]

    async def event_generator():
        """按租户订阅事件并向 SSE 推流的生成器。

        :return: 逐条产出 {"data": JSON字符串} 的 SSE 事件帧；客户端断开时退订。
        """
        q = events_hub.subscribe(tenant_id)
        try:
            # 握手：让前端一上来就知道连接已成，而不是干等到第一个事件
            yield {"data": json.dumps(
                {"type": "connected", "tenant_id": tenant_id}, ensure_ascii=False)}
            while True:
                ev = await q.get()
                yield {"data": json.dumps(ev, ensure_ascii=False)}
        except asyncio.CancelledError:
            raise                                          # 客户端断开 → 正常退订
        finally:
            events_hub.unsubscribe(tenant_id, q)

    return EventSourceResponse(event_generator())


# ── 研报详情（P5）──────────────────────────────────────────────
# 发布后的研报是对内的公开产物，三个角色都可读 —— 与 /review 那条
# 「只有风控与管理员能读待审内容」是刻意相反的：审阅对象与交付产物
# 的可见范围本来就不同。

@router.get("/tasks/{task_id}/report")
async def get_published_report(
    task_id: uuid.UUID,
    current_user: dict = Depends(require_role("researcher", "risk_control", "admin")),
):
    """已发布研报的详情。

    未发布返回 404，**不回落到 draft**：草稿是风控台的审阅对象，
    研究员不该从这条路径读到它（要看待审内容走 /review，那条有角色限制）。

    :param task_id: 任务 UUID，URL 路径参数 /tasks/{task_id}/report；畸形输入返回 422。
    :param current_user: 当前用户信息 dict（Depends(require_role("researcher","risk_control","admin")) 注入）；三个角色均可读。
    :return: dict，含 task_id/title/content/rating/risk_disclosure/published_at/status/valuation/references；任务或研报未发布返回 404。
    """
    tenant_id = current_user["tenant_id"]
    row = await repo.get_task(tenant_id, task_id)
    if not row:
        # 跨租户 404 而不是 403：403 会泄漏「这个 id 存在」
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")

    report = await repo.get_report(tenant_id, task_id)
    if report is None or report["status"] != "published":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="该任务尚未发布研报")

    valuation_row = await repo.fetch_valuation_result(tenant_id, task_id)
    if valuation_row is None:
        valuation = None
    else:
        available = bool(valuation_row["is_available"])
        valuation = {
            # 库列是 value_low/high（沿用既有列），对外叫 equity_value_* ——
            # 响应里同时出现「股权价值」与「每股价值」两套口径，只叫 value_* 分不清是哪套。
            "method": valuation_row["method"] if available else None,
            "equity_value_low": valuation_row["value_low"],
            "equity_value_high": valuation_row["value_high"],
            "per_share_low": valuation_row["per_share_low"],
            "per_share_high": valuation_row["per_share_high"],
            "currency": valuation_row["currency"],
            "assumptions": valuation_row["assumptions"],
            "rationale": valuation_row["rationale"],
            "is_available": available,
        }

    return {
        "task_id":         str(row["id"]),
        "title":           report["title"],
        "content":         report["content"],
        "rating":          report["rating"],
        "risk_disclosure": report["risk_disclosure"],
        "published_at":    report["published_at"],
        "status":          report["status"],
        # 结构化字段给界面用；正文给人读。前端不解析正文文本取数字。
        "valuation":       valuation,
        "references": {
            # 这两个值来自 research_reports.comparison_points 那一列（本任务 Step 4
            # 新增）。它们原本只在 State 里、没落业务表 —— 不落库的话，详情页的
            # 结构化字段就只能靠解析正文，而规格明确禁止那样做。
            "has_reference": bool(report.get("comparison_points")),
            "comparison_points": report.get("comparison_points") or [],
        },
    }


# ── 风控人工签字（P6）───────────────────────────────────────────
# 这三个接口是「未签字不得发布」这条规则在人这一侧的入口。图已经把闸门焊死了，
# 这里负责把待审内容摆给人看、把签字收上来、并挡住不该签的人。

class RiskDecision(BaseModel):
    """签字请求体。

    字段刻意宽松（decision 是 str、redo_targets 是自由 dict），校验放在处理函数里：
    这样非法载荷回的是 422 与一句能读懂的中文，而不是 pydantic 的一串英文路径。
    """
    decision: str = Field(..., max_length=16, description="approve / modify / reject")
    comments: str = Field(..., max_length=2000, description="签字意见，不得短于 10 字")
    redo_targets: Optional[dict] = Field(None, description="驳回时的重做目标")


async def _load_task_for_review(tenant_id: str, task_id, current_user: dict) -> dict:
    """取任务行并做「跨租户 → 404」这层判定，供两个审核接口共用。

    :param tenant_id: 当前租户 ID（str），用于限定任务查询范围。
    :param task_id: 任务 ID（可为 uuid.UUID）。
    :param current_user: 当前用户信息 dict（此处未直接使用，仅签名占位）。
    :return: dict，任务行；任务不存在时抛 HTTPException 404。
    """
    row = await repo.get_task(tenant_id, task_id)
    if not row:
        # 与 GET /tasks/{id} 同一条约定：跨租户 404 而不是 403，403 会泄漏「这个 id 存在」
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")
    return row


@router.get("/tasks/{task_id}/review")
async def get_task_review(
    task_id: uuid.UUID,
    # 只有风控与管理员能读：这是待审视图，研究员（提交方）不该看到风控给它的批注草稿
    current_user: dict = Depends(require_role("risk_control", "admin")),
):
    """待审内容：草稿全文 + 预检清单 + 待签字记录 + 阶段事件。

    内容一律从业务表读，不碰 checkpoint —— persist_draft 在闸门【之前】就把它写好了，
    正是为了让这个接口不需要理解图的状态（设计文档 §4.3 的分工）。

    :param task_id: 任务 UUID，URL 路径参数 /tasks/{task_id}/review；畸形输入返回 422。
    :param current_user: 当前用户信息 dict（Depends(require_role("risk_control","admin")) 注入）；仅风控与管理员可读待审内容。
    :return: dict，含 stages/task_id/status/current_stage/redo_count/reviewable/report/checklist/review/history。
    """
    tenant_id = current_user["tenant_id"]
    row = await _load_task_for_review(tenant_id, task_id, current_user)

    report = await repo.get_report(tenant_id, task_id)
    review = await repo.get_latest_risk_review(tenant_id, task_id)
    events = await repo.list_stage_events(tenant_id, task_id)

    return {
        # 阶段事件一并返回：审核人要判断的不只是「草稿写了什么」，还有「它是怎么来的」
        # （哪一步跳过了、哪一步失败过）。单独再调一次 /events 会多一个往返，
        # 而这个页面本来就是一次读取。
        "stages": [
            {"stage": e["stage"], "status": e["status"],
             "detail": e["detail"], "occurred_at": e["occurred_at"]}
            for e in events
        ],
        "task_id":       str(row["id"]),
        "status":        row["status"],
        "current_stage": row["current_stage"],
        "redo_count":    row["redo_count"],
        "reviewable":    row["status"] == "awaiting_risk_review",
        "report": None if report is None else {
            "id":              str(report["id"]),
            "title":           report["title"],
            "content":         report["content"],
            "rating":          report["rating"],
            "status":          report["status"],
            "risk_disclosure": report["risk_disclosure"],
        },
        "checklist": (review or {}).get("checklist") or {},
        "review": None if review is None else {
            "id":           str(review["id"]),
            "decision":     review["decision"],
            "comments":     review["comments"],
            "redo_targets": review["redo_targets"],
            "signed_at":    review["signed_at"],
        },
        "history": [
            {"decision": r["decision"], "comments": r["comments"],
             "redo_targets": r["redo_targets"], "signed_at": r["signed_at"]}
            for r in await repo.list_risk_reviews(tenant_id, task_id) if r["decision"]
        ],
    }


@router.post("/tasks/{task_id}/risk-decision", status_code=status.HTTP_202_ACCEPTED)
async def submit_risk_decision(
    task_id: uuid.UUID,
    req: RiskDecision,
    # 提交方与审核方必须不是同一个人：researcher 不能给自己提交的研报签字。
    # 这与 POST /tasks 排除 risk_control 是同一条规则的两半，缺一半闭环就不成立。
    current_user: dict = Depends(require_role("risk_control", "admin")),
):
    """提交风控签字，202；获批的流水线在后台继续跑。

    四道闸，顺序有意为之 —— 先判「能不能签」，再判「签的内容合不合法」，
    最后才判「图还在不在等」：
      ① 跨租户 → 404
      ② 状态不是待审 → 409（挡住重复签字）
      ③ 载荷非法 → 422
      ④ 暂停态丢失 → 409（后端重启过；**绝不能**让它 resume）

    :param task_id: 任务 UUID，URL 路径参数 /tasks/{task_id}/risk-decision。
    :param req: 请求体（RiskDecision）：decision（approve/modify/reject 之一）、comments（签字意见，去空白后不得短于 10 字）、redo_targets（驳回时的重做目标 dict，reject 时须含合法 stage）。
    :param current_user: 当前用户信息 dict（Depends(require_role("risk_control","admin")) 注入）；签字方与提交方不可为同一人。
    :return: dict，结构 {"task_id": str, "decision": str, "status": "resuming"}；成功返回 HTTP 202 Accepted。
    """
    from backend.agents.research.runner import resume_pipeline

    tenant_id = current_user["tenant_id"]
    row = await _load_task_for_review(tenant_id, task_id, current_user)

    # ② 只有待审状态才签得动。用业务表的状态判，而不是 checkpoint：
    #    两者不一致时（重启后）业务表是更可信的那一边。
    if row["status"] != "awaiting_risk_review":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"任务当前不在待审状态（{row['status']}），不能签字",
        )

    # ③ 载荷校验。三条都必须自己判，不能指望 DB 约束兜底：
    #    comments 太短会撞 chk_signed_complete 变成 500；
    #    reject 缺重做阶段会走到路由的「目标阶段不可识别」分支，
    #    任务同样停住，但失败理由变成「状态损坏」，真实原因一个字都不剩。
    decision = req.decision
    if decision not in VALID_DECISIONS:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail=f"decision 必须是 {'/'.join(VALID_DECISIONS)} 之一")
    comments = req.comments.strip()
    if len(comments) < MIN_COMMENTS_LEN:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail=f"签字意见不得短于 {MIN_COMMENTS_LEN} 字")
    redo_targets = req.redo_targets
    if decision == "reject":
        stage = (redo_targets or {}).get("stage")
        if stage not in REDO_STAGES:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"驳回必须指定重做阶段（{'/'.join(REDO_STAGES)}），实际 {stage!r}",
            )

    # ④ 最后一关，也是本轮最容易埋雷的一关：这个任务上【真的还有】一份待签的
    #    暂停点吗？去 LangGraph 后不再有 aget_state().next 可查，改用轻量 checkpoint
    #    （runner.run_risk_stage 在首跑落草稿时写入 pipeline_checkpoints）。它与业务表
    #    的 awaiting_risk_review 应永远一致；这里再查一次，正是把「重复签字 / 对已完成
    #    任务签字」挡在它们会触发一次无谓的续跑之前。它挡的情况：这条任务当前没有
    #    待签暂停点（已签过、或从未进入待审）。无 checkpoint 即无暂停 —— 前进式续跑
    #    无从记起，所以停在这里，不让 runner 去猜。
    if await repo.get_checkpoint(tenant_id, str(task_id)) is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="该任务当前没有待签字的暂停点（可能已签字完成或从未进入待审），"
                   "无需再次签字",
        )

    await repo.write_audit_log(tenant_id, current_user["user_id"],
                               "research_task.risk_sign", "research_task", str(task_id),
                               {"decision": decision, "redo_targets": redo_targets})

    resume_pipeline(
        {"task_id": str(row["id"]), "tenant_id": tenant_id, "thread_id": row["thread_id"]},
        {"decision": decision, "comments": comments, "redo_targets": redo_targets,
         "reviewer_id": current_user["user_id"]},
    )
    return {"task_id": str(row["id"]), "decision": decision, "status": "resuming"}
