# backend/agents/risk/state.py

from typing import Optional

from typing_extensions import TypedDict


class RiskState(TypedDict):
    """风控子图 State。

    重构后子图模型仍然是「显式声明它需要读哪些键」：steps.run_risk_stage 把
    父 State 的字段挑进来喂给各节点，节点不猜。闸门不再用 LangGraph 的
    interrupt —— 改成「写 checkpoint + 返回哨兵」，恢复值（签字）由
    runner.resume_pipeline 注入 state["review_payload"]。
    """

    task_id: str
    tenant_id: str
    current_stage: str

    # ── 从父图继承 ──────────────────────────────────────────────
    company_id: str                     # 写 research_reports.company_id（NOT NULL）
    company_code: str                   # 研报标题与正文里的标的标识
    industry: str                       # 研报正文的「所属行业」一行
    rating: str                         # 待审草稿的评级
    rating_note: str                    # 评级说明，随评级一起披露
    has_reference: bool                 # 预检与风险揭示都要回答「有没有历史参照」
    comparison_points: list[str]        # 研报正文「历史研报参照」一节的内容
    valuation_available: bool           # 预检项之一，且要写进风险揭示
    valuation: Optional[dict]           # 研报正文「估值」一节的真实产物
    redo_count: int                     # 驳回后 +1 并回写，供路由判上限

    # ── 风控子图产物 ───────────────────────────────────────────
    compliance_checklist: dict          # 自动预检结果（快照进 risk_reviews.checklist）
    risk_decision: Optional[str]        # approve / modify / reject
    redo_targets: Optional[dict]
    risk_comments: Optional[str]
    draft_report_id: Optional[str]      # 待审草稿的 id，审核人视图要带给前端。
    # 签字内容的长久留痕在 risk_reviews 表里，不在 State 里。
    review_payload: Optional[dict]      # apply_decision 读它；resume 时由 runner 注入