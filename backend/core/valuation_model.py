# backend/core/valuation_model.py
# 估值纯函数。零 LLM、零 IO、可独立测试。
#
# 本模块【只算数】：不判断数据够不够、不写库、不调模型 —— 那些在 valuation 子图的节点里。
# 数字根本不经过 LLM 是硬约束（规格偏离 1）：这样「不编造数字」是结构性事实，
# 而不是一句要靠提示词被遵守的祈愿，并且可以被一条断言钉住
# （断言 LLM 的输出 schema 里没有任何数值字段，见 tests/test_valuation_agent.py）。
#
# ⚠️ GROWTH_BAND 是【经验值，未经实测标定】（总设计 §12 的待标定项）。
# 它不是统计意义上的置信区间，而是一个**明示的敏感性带宽**：区间宽度完全由它决定。

from dataclasses import dataclass

# 增长率的不确定性带宽：DCF 区间 = 在 growth ± BAND 两点各算一次。
# （经验值，未经实测标定 —— 待标定后替换）
GROWTH_BAND = 0.02


@dataclass(frozen=True)
class Range:
    """一个区间。构造时就挡住倒挂 —— 倒挂的区间在下游会被当成「负宽度」用出各种花样。"""
    low: float
    high: float

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ValueError(f"区间倒挂：low={self.low} > high={self.high}")


def run_dcf(*, base_cashflow_yi: float, growth_rate: float, years: int,
            discount_rate: float, terminal_growth: float) -> Range:
    """两段式 DCF：明确预测期 + 永续期（Gordon 增长）。

    区间来自增长率的不确定性（growth ± GROWTH_BAND），不是猜一个上下限。
    """
    if base_cashflow_yi <= 0:
        raise ValueError("基期现金流必须为正 —— 负现金流用本模型算出负数没有意义")
    if years <= 0:
        raise ValueError("预测年限必须为正")
    if discount_rate <= terminal_growth:
        raise ValueError("折现率必须大于永续增长率，否则永续期现值发散")

    values: list[float] = []
    for rate in (growth_rate - GROWTH_BAND, growth_rate + GROWTH_BAND):
        cashflow = base_cashflow_yi
        present = 0.0
        for year in range(1, years + 1):
            cashflow = cashflow * (1 + rate)
            present += cashflow / (1 + discount_rate) ** year
        terminal = cashflow * (1 + terminal_growth) / (discount_rate - terminal_growth)
        present += terminal / (1 + discount_rate) ** years
        values.append(present)

    return Range(low=min(values), high=max(values))


def run_comparable(*, net_profit_yi: float, pe_low: float, pe_high: float) -> Range:
    """可比公司法：净利润 × PE 区间。PE 区间来自配置，标注为模型参数。"""
    if pe_low <= 0 or pe_high <= 0:
        raise ValueError("PE 必须为正")
    if net_profit_yi <= 0:
        raise ValueError("净利润必须为正 —— 亏损企业不适用 PE 法")
    return Range(low=net_profit_yi * pe_low, high=net_profit_yi * pe_high)


def blend(*ranges: Range) -> Range:
    """合成：取所有区间的并集外包。"""
    if not ranges:
        raise ValueError("没有可合成的区间")
    return Range(low=min(r.low for r in ranges), high=max(r.high for r in ranges))


def per_share(equity_value_yi: float, total_shares_wan: float) -> float:
    """每股价值（元）。

    单位换算写死在这里，不靠心算：数据侧「万元」「亿元」混着来，漏掉 1e4
    时结果差四个数量级却看起来一切正常。

        per_share = 股权价值(亿元) × 1e8 / (总股本(万股) × 1e4)
                  = 股权价值 / 总股本 × 1e4
    """
    if total_shares_wan <= 0:
        raise ValueError("总股本必须为正")
    return equity_value_yi / total_shares_wan * 1e4


def select_range(method: str, dcf: Range, comparable: Range) -> Range:
    """按 LLM 选定的方法取最终区间。方法名之外的判断一律不下放给 LLM。"""
    if method == "dcf":
        return dcf
    if method == "comparable":
        return comparable
    if method == "blended":
        return blend(dcf, comparable)
    raise ValueError(f"未知估值方法：{method!r}")
