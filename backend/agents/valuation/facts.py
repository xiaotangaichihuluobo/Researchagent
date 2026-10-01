# backend/agents/valuation/facts.py
# 财务提数的确定性收尾:把 LLM 提取出的【年报行表】转成估值所需的事实。
#
# 原核心层用正则模式表提数(core/financial_extract.py),对真实公告措辞变体太脆。
# 现改为结构化 LLM 提数(见 nodes.py 的 extract_facts_node),但 LLM 的裸数字不能
# 直接喂进 DCF —— 读错一个数字比编造一个数字更难被发现,它看起来像是从数据来的。
# 所以这里在 LLM 之后做两件**确定性**的事,把「错得自然」挡在计算之外:
#   ① 交叉校验:披露的营收同比 vs 由相邻两期扣除的营收反算出的同比,差太多判错;
#   ② MISSING:缺哪些必要输入,精确到项,缺了就触发降级,绝不补零。
# 本模块零 LLM、零 IO,可独立测试:喂一份 rows,出来一组带出处的数字。

from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, Field

# 交叉校验容差(百分点):披露的增速 vs 由相邻两期年报营收反算的增速。
CROSS_CHECK_TOLERANCE_PCT = 0.5

REQUIRED_FIELDS: tuple[str, ...] = (
    "revenue_yi", "revenue_yoy_pct", "net_profit_yi",
    "gross_margin_pct", "operating_cashflow_yi",
)


@dataclass(frozen=True)
class Extracted:
    """一个提取出来的数值,连同它在原文中的出处。"""
    value: float
    excerpt: str        # 命中处附近的原文片段(LLM 自证的引文)
    source_title: str   # 来自哪一份资料


@dataclass(frozen=True)
class FinancialFacts:
    """一次估值所需的全部输入。每一项要么带值带出处,要么是 None。"""
    revenue_yi: Optional[Extracted]
    revenue_yoy_pct: Optional[Extracted]
    net_profit_yi: Optional[Extracted]
    gross_margin_pct: Optional[Extracted]
    operating_cashflow_yi: Optional[Extracted]
    total_shares_wan: Optional[Extracted]
    base_report_title: Optional[str]
    cross_checked: bool          # 营收增速是否通过了「披露值 vs 反算值」校验


# LLM 输出契约里的「一份年报」行。数值可空(提不到给 null),excerpt 必填。
class ReportExtraction(BaseModel):
    index: int                     # 对应提示词里给的资料序号
    year: int                      # 年报所属年份
    revenue_yi: Optional[float] = Field(default=None, description="营业收入(亿元)")
    revenue_yoy_pct: Optional[float] = Field(default=None, description="营收同比增速(%)")
    net_profit_yi: Optional[float] = Field(default=None, description="归母净利润(亿元)")
    gross_margin_pct: Optional[float] = Field(default=None, description="毛利率(%)")
    operating_cashflow_yi: Optional[float] = Field(default=None, description="经营现金流净额(亿元)")
    total_shares_wan: Optional[float] = Field(default=None, description="总股本(万股),有则填")
    excerpt: str = Field(..., description="支撑上述数字的原文摘要/引文,必须来自真实财报原文")


class ExtractionResult(BaseModel):
    """一次提数的全部输出:仅年度报告;半年报/季报/公告不输出。"""
    docs: list[ReportExtraction] = Field(..., description="识别出的所有年度报告")


def _cross_check(base_yoy_pct: float, base_revenue: float,
                 prev_revenue: Optional[float]) -> bool:
    """披露的增速 vs 由相邻两期年报营收反算的增速。

    只有一期年报时无法反算 —— 那种情况下调用方把 cross_checked 记 False,
    但**照常参与计算**(降级会因数据正常而频繁误触发)。
    """
    if prev_revenue is None or prev_revenue == 0:
        return True
    implied = (base_revenue / prev_revenue - 1) * 100
    return abs(implied - base_yoy_pct) <= CROSS_CHECK_TOLERANCE_PCT


def to_financial_facts(rows: list[ReportExtraction],
                       source_titles: dict[int, str],
                       tolerate_cross_check_failure: bool = False) -> FinancialFacts:
    """把 LLM 输出的年报行表【确定性】转成事实。

    LLM 只输出年度报告(半年报/季报已由提示词排除),基期取年份最大的最新一期;
    cross-check 用基期披露的营收同比 vs 相邻上一期营收反算出的同比。

    base_report_title 用源资料标题而不是 LLM 内部年份 —— 出处要对齐到真实资料。
    """
    if not rows:
        return None
    base = max(rows, key=lambda r: r.year)
    if len(rows) >= 2:
        prev = max((r for r in rows if r.year < base.year), key=lambda r: r.year, default=None)
    else:
        prev = None

    title = source_titles.get(base.index) or ""
    revenue = _as_extracted(base, "revenue_yi", title)
    yoy_in = base.revenue_yoy_pct

    # 交叉校验:把披露的增速与反算的增速对一遍。
    # 判错就作废该假设、走降级,而不是任选一个值继续算。
    cross_checked = True
    yoy = None
    if revenue and yoy_in is not None and prev is not None and prev.revenue_yi is not None:
        if _cross_check(yoy_in, revenue.value, prev.revenue_yi):
            yoy = _as_extracted(base, "revenue_yoy_pct", title)
        else:
            cross_checked = False
            if not tolerate_cross_check_failure:
                return None
    elif yoy_in is not None:
        yoy = _as_extracted(base, "revenue_yoy_pct", title)

    # 总股本可能出现在任何一份资料里,全行扫首个非空。
    total = next((_as_extracted(r, "total_shares_wan", source_titles.get(r.index) or "")
                  for r in rows if r.total_shares_wan is not None), None)

    facts = FinancialFacts(
        revenue_yi=revenue,
        revenue_yoy_pct=yoy,
        net_profit_yi=_as_extracted(base, "net_profit_yi", title),
        gross_margin_pct=_as_extracted(base, "gross_margin_pct", title),
        operating_cashflow_yi=_as_extracted(base, "operating_cashflow_yi", title),
        total_shares_wan=total,
        base_report_title=title or None,
        cross_checked=cross_checked,
    )
    # 容错模式下【不】因缺项整体作废:缺哪一项就报哪一项,由调用方决定是否降级。
    if MISSING_FIELDS(facts) and not tolerate_cross_check_failure:
        return None
    return facts


def _as_extracted(row: ReportExtraction, field: str,
                  source_title: str) -> Optional[Extracted]:
    value = getattr(row, field)
    if value is None:
        return None
    return Extracted(value=float(value), excerpt=row.excerpt, source_title=source_title)


def MISSING_FIELDS(facts: Optional[FinancialFacts]) -> list[str]:
    """缺哪些【必要】输入。总股本不在其中:它缺了只丢每股口径,不降级。"""
    if facts is None:
        return list(REQUIRED_FIELDS)
    return [name for name in REQUIRED_FIELDS if getattr(facts, name) is None]