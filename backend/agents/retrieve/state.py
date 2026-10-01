# backend/agents/retrieve/state.py

from typing import Optional

from typing_extensions import TypedDict


class RetrieveState(TypedDict):
    """检索子图 State。

    retrieved_chunks 【故意】只声明在子图里，不声明在父 State 中：
    大对象不进父快照（重构前靠 LangGraph「子图独有键被静默丢弃」；
    重构后由 steps.run_retrieve_stage 显式筛选回写的键）。top-K 限死并截断正文。
    """

    task_id: str
    tenant_id: str
    company_code: str
    industry: str
    current_stage: str
    retrieved_chunks: list[dict]        # content / score / source / published_at
    comparison_points: list[str]
    has_reference: bool

    # ── 子图内部中间键（steps 显式决定不同步回父 State）───────────
    _query: str                       # prepare_query 拼好的检索文本
    search_failed: bool               # 检索段是否失败（与「提炼失败」区分）
    _search_note: Optional[str]       # 失败原因，供 extract 阶段留痕