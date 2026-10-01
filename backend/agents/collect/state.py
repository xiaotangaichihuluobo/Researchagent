# backend/agents/collect/state.py

from typing_extensions import TypedDict

from backend.agents.collect.adapters import RawDataItem


class CollectState(TypedDict):
    """采集子图 State。

    正文（财报全文、新闻正文）【不】进 State —— 它们落在 research_data_items 表里，
    State 只留 collected_ids 这份引用列表。重构后不再有 checkpoint 全量序列化，
    「大对象不进 State」的分工保留，因为它本就是合理的：父 State 只需引用。
    """

    task_id: str
    tenant_id: str
    company_id: str
    company_code: str
    industry: str
    current_stage: str                 # 阶段机推进依赖；由入口节点写入
    # 本次采集到的原始条目（已截断）。【只存在于子图】：父 State 不声明这个键，
    # 采集正文不需要进入父图快照 —— combine 时由 steps.py 显式决定带哪些键。
    raw_items: list[RawDataItem]
    collected_ids: list[str]           # 写入 research_data_items 后的 ID
    source_stats: dict                 # 各源成功/失败统计
    errors: list[dict]                 # 部分失败记录