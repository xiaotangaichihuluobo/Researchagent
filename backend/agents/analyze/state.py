# backend/agents/analyze/state.py

from typing import Optional

from typing_extensions import TypedDict


class AnalyzeState(TypedDict):
    """分析子图 State。

    dimension_results 原来靠 reducer 合并四维并发的产出；
    重构后并发由 steps.run_analyze_stage 的 asyncio.gather 显式执行、
    合并手写在编排层，这里只声明字段形状。
    """

    task_id: str
    tenant_id: str
    company_id: str
    company_code: str
    industry: str
    current_stage: str
    redo_targets: Optional[dict]        # 父图传入 {"stage":"analyze","dimensions":[...]}
    redo_dimensions: list[str]          # prepare_evidence 从 redo_targets 推导
    dimension_results: dict
    rating: str
    rating_note: str
    has_buy_sell_advice: bool
    fallback_used: bool