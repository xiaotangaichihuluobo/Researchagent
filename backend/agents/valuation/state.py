# backend/agents/valuation/state.py

from typing import Optional

from typing_extensions import TypedDict


class ValuationState(TypedDict):
    task_id: str
    tenant_id: str
    company_code: str
    current_stage: str

    # ── 从父图继承 ──────────────────────────────────────────────
    dimension_results: dict
    comparison_points: list[str]      # 横向检索的产物

    # ── 本子图内部（steps 显式决定哪些键回写父 State）────────────
    industry: str
    company_id: str
    financial_facts: Optional[dict]   # extract_facts_node 的产物；missing 则 None
    inputs_ok: bool                   # prepare_inputs 判定；下游节点据此短路
    valuation_inputs: Optional[dict]  # 基期营收/现金流/净利润/总股本/基期报告标题
    assumption_sources: list[dict]    # 每条假设的 {name, value, source}
    degrade_reason: Optional[str]     # prepare_inputs 判定的降级原因
    dcf_result: Optional[dict]        # {"low": float, "high": float}
    comparable_result: Optional[dict]

    # ── 交付给父图 ─────────────────────────────────────────────
    # {method, equity_value_low, equity_value_high, per_share_low, per_share_high,
    #  currency, assumptions, rationale, is_available}
    # 股权价值单位亿元、每股价值单位元。不可用时为 None —— 绝不给默认区间。
    valuation: Optional[dict]
    valuation_available: bool