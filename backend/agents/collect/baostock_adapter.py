# backend/agents/collect/baostock_adapter.py
# 财报真实源：走 baostock（BSD 许可、免费、免注册）。
#
# 四个源里为什么只有财报/新闻接真实源、公告/行业不接：
#   财报有 baostock 这种干净、宽松许可、返回结构化数据行的免费接口，接得不亏；
#   公告只能爬巨潮/东方财富（GPL + 页面脆弱），行业景气是纯付费 —— 收益配不上
#   引入的维护与许可成本，继续用 fixture 是诚实的取舍。
#
# baostock 是【同步】库，而采集节点用 asyncio.gather 并发 await —— fetch 里必须用
# asyncio.to_thread 包住同步调用，否则同步调用会阻塞事件循环，把其余三个源拖停。
#
# 与新闻源同一条哲学：真实源失败【不抛异常】、降级到同名 fixture。财报不可用不该
# 让整个采集失败，fixture 财报仍是兜底素材。

import asyncio
from datetime import date, datetime

from backend.agents.collect.adapters import RawDataItem, _fixture
from backend.core.logger import get_logger

logger = get_logger(__name__)

# 财报条目并存两类，服务两个下游：
#   ① 年度报告（散文格式，标题"YYYY 年年度报告"）→ 估值提数用（nodes.extract_facts_node）
#   ② 季度报告（盈利能力/成长性）→ 新鲜度高，喂基本面维度保证"证据充分"参与评级
_ANNUAL_ITEMS = 2          # 产出最近 N 期年报（估值营收增速交叉校验需相邻两期）
_QUARTER_ITEMS = 2         # 产出最近 N 期季度条目（基本面维度新鲜度）
_PROBE_ANNUAL_YEARS = 4    # 往回探测多少年报年度（够算 2 期年报各自的同比）
_PROBE_QUARTERS_BACK = 6   # 往回探测多少报告季度（避开未披露期）


def _to_baostock_code(company_code: str) -> str:
    """把 600519.SH 转成 baostock 的 sh.600519 前缀格式；认不出后缀就原样返回。

    :param company_code: 形如「数字.交易所后缀」的股票代码，如 600519.SH。
    :return: 前缀化后的 baostock 代码（sh./sz./bj.）；后缀不可识别时原样返回。
    """
    code, _, suffix = company_code.partition(".")
    prefix = {"SH": "sh", "SZ": "sz", "BJ": "bj"}.get(suffix.upper())
    return (f"{prefix}.{code}") if prefix else company_code


def _recent_quarters(now: date) -> list[tuple[int, int]]:
    """从 now 所在季度往回推若干 (year, quarter)，新的在前。

    :param now: 当前日期，用于定位起始季度。
    :return: (year, quarter) 元组列表，按从新到旧排列，长度由 _PROBE_QUARTERS_BACK 决定。
    """
    quarters: list[tuple[int, int]] = []
    y, q0 = now.year, (now.month - 1) // 3
    for _ in range(_PROBE_QUARTERS_BACK):
        quarters.append((y, q0 + 1))
        q0 -= 1
        if q0 < 0:
            y, q0 = y - 1, 3
    return quarters


def _parse_dt(value: str) -> datetime | None:
    """'2026-04-17' → 带 UTC 时区的 datetime；解析不出返回 None（不抛）。

    :param value: 日期字符串（可为 "-" 或空）。
    :return: datetime；naive 值补 UTC 时区，解析失败返回 None。
    """
    if not value or value in ("", "-"):
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    return dt.replace(tzinfo=None) if dt.tzinfo is None else dt


def _fmt_pct(value: str) -> str | None:
    """baostock 小数 → 百分数（0.8955 → '89.56%'）；'-'/空返回 None。

    :param value: baostock 返回的小数点形式比率字符串。
    :return: 格式化后的百分数字符串；空/'-'/解析失败返回 None。
    """
    try:
        if value in ("", "-"):
            return None
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return None


def _fmt_money_yi(value: str) -> str | None:
    """元 → 亿元（92278072083 → '922.8 亿'）；'-'/空返回 None。

    :param value: baostock 返回的金额字符串（单位为元）。
    :return: 格式化后的亿元字符串（保留 1 位小数）；空/'-'/解析失败返回 None。
    """
    try:
        if value in ("", "-"):
            return None
        return f"{float(value) / 1e8:.1f} 亿"
    except (TypeError, ValueError):
        return None


def _collect_annuals(bs, code: str, now: date) -> list[dict]:
    """取最近若干年报年度的财务事实，新的在前。同比由相邻两期真实营收/净利算。

    :param bs: 已登录的 baostock 模块。
    :param code: baostock 前缀格式的股票代码。
    :param now: 当前日期，决定回溯多少年报年度。
    :return: 年报事实 dict 列表，按年份从新到旧；每项含 year/rev_yi/net_yi/
        gross_margin_pct/shares_wan/net_yoy_pct/rev_yoy_pct/cfo_yi 等键。
    """
    years = [now.year - 1 - i for i in range(_PROBE_ANNUAL_YEARS)]
    recs: list[dict] = []
    for y in years:
        p = bs.query_profit_data(code=code, year=y, quarter=4)
        if p.error_code != "0" or not p.data:
            continue
        row = p.data[0]
        fidx = {f: i for i, f in enumerate(p.fields)}
        pf = lambda k: row[fidx[k]]                      # noqa: E731
        if pf("MBRevenue") in ("", "-"):
            continue
        rev_yi = float(pf("MBRevenue")) / 1e8
        rec: dict = {
            "year": y,
            "rev_yi": rev_yi,
            "net_yi": float(pf("netProfit")) / 1e8,
            "gross_margin_pct": float(pf("gpMargin")) * 100,
            "shares_wan": (float(pf("totalShare")) / 1e4
                           if pf("totalShare") not in ("", "-") else None),
            "net_yoy_pct": None,
            "cfo_yi": None,
            "published_at": _parse_dt(pf("pubDate")) or _parse_dt(pf("statDate")),
        }
        g = bs.query_growth_data(code=code, year=y, quarter=4)
        if g.error_code == "0" and g.data:
            rec["net_yoy_pct"] = float(g.data[0][g.fields.index("YOYPNI")]) * 100
        c = bs.query_cash_flow_data(code=code, year=y, quarter=4)
        if c.error_code == "0" and c.data:
            # 经营现金流 = CFOToOR（经营现金流/营收）× 营收。baostock 的现金流
            # 接口只给比率；绝对值由两个真实数字推导得出，source 可溯。
            rec["cfo_yi"] = float(c.data[0][c.fields.index("CFOToOR")]) * rev_yi
        recs.append(rec)
    recs.sort(key=lambda r: r["year"], reverse=True)
    for i, rec in enumerate(recs):
        prev = recs[i + 1] if i + 1 < len(recs) else None
        if rec["net_yoy_pct"] is None and prev and prev["net_yi"]:
            rec["net_yoy_pct"] = (rec["net_yi"] / prev["net_yi"] - 1) * 100
        if prev and prev["rev_yi"]:
            rec["rev_yoy_pct"] = (rec["rev_yi"] / prev["rev_yi"] - 1) * 100
        else:
            rec["rev_yoy_pct"] = None
    return recs


def _annual_item(company_code: str, a: dict) -> RawDataItem:
    """把一期真实年报组装的散文条目 —— 估值 LLM 提数的输入载体。

    自估值提数改 LLM 后（见 facts.py / nodes.extract_facts_node），这里不再要求严格
    正则格式，只保证五类数字完整落进文本供 LLM 提取。标题保持「YYYY 年年度报告」，
    便于提数提示词区分年报与季报/公告。每个数都可溯源，同比由相邻两期真实值算或取披露值。

    :param company_code: 股票代码，拼进条目标题。
    :param a: `_collect_annuals` 产出的一期年报事实 dict（rev_yi/net_yi 等键）。
    :return: 该期年报组装成的 RawDataItem（financial_report 类型）。
    """
    parts = [
        f"营业收入 {a['rev_yi']:.1f} 亿元，同比增长 {a['rev_yoy_pct']:.1f}%",
        f"归属于上市公司股东的净利润 {a['net_yi']:.1f} 亿元，同比增长 {a['net_yoy_pct']:.1f}%",
        f"毛利率 {a['gross_margin_pct']:.1f}%",
        f"经营活动产生的现金流量净额 {a['cfo_yi']:.1f} 亿元",
    ]
    if a.get("shares_wan"):
        parts.append(f"总股本 {a['shares_wan']:.0f} 万股")
    return RawDataItem(
        source_type="financial_report", source_name="baostock",
        title=f"{company_code} {a['year']} 年年度报告",
        content="；".join(parts), published_at=a["published_at"])


def _collect_quarters(bs, bs_code: str, company_code: str,
                      now: date) -> list[RawDataItem]:
    """季报（新鲜度，喂基本面维度）。

    :param bs: 已登录的 baostock 模块。
    :param bs_code: baostock 前缀格式的股票代码。
    :param company_code: 股票代码，拼进条目标题。
    :param now: 当前日期，用于回溯最近报告季度。
    :return: 盈利能力/成长性两类季报条目组成的 RawDataItem 列表。
    """
    items: list[RawDataItem] = []
    periods_filled = 0
    for year, quarter in _recent_quarters(now):
        if periods_filled >= _QUARTER_ITEMS:
            break
        if quarter == 4:
            continue                       # 年报走年报路径，季度路径只收 Q1–Q3 当季
        profit = bs.query_profit_data(code=bs_code, year=year, quarter=quarter)
        growth = bs.query_growth_data(code=bs_code, year=year, quarter=quarter)
        profit_rows = profit.data if profit.error_code == "0" else []
        growth_rows = growth.data if growth.error_code == "0" else []
        if not profit_rows and not growth_rows:
            continue
        added_here = False
        first = profit_rows or growth_rows
        fields = profit.fields if profit_rows else growth.fields
        label = first[0][fields.index("statDate")].replace("-", "/")
        published = (_parse_dt(first[0][fields.index("pubDate")])
                     or _parse_dt(first[0][fields.index("statDate")]))

        if profit_rows:
            win = {f: i for i, f in enumerate(profit.fields)}
            parts = []
            for fld, name in (("gpMargin", "销售毛利率"), ("npMargin", "销售净利率"),
                              ("roeAvg", "净资产收益率")):
                v = _fmt_pct(profit_rows[0][win[fld]])
                if v:
                    parts.append(f"{name} {v}")
            for fld, name in (("MBRevenue", "营业收入"), ("netProfit", "归母净利润")):
                v = _fmt_money_yi(profit_rows[0][win[fld]])
                if v:
                    parts.append(f"{name} {v}")
            if parts:
                items.append(RawDataItem(
                    source_type="financial_report", source_name="baostock",
                    title=f"{company_code} {label} 盈利能力",
                    content="；".join(parts), published_at=published))
                added_here = True
        if growth_rows:
            gin = {f: i for i, f in enumerate(growth.fields)}
            parts = []
            for fld, name in (("YOYPNI", "归母净利润同比"), ("YOYNI", "净利润同比"),
                              ("YOYEquity", "净资产同比")):
                v = _fmt_pct(growth_rows[0][gin[fld]])
                if v:
                    parts.append(f"{name} {v}")
            if parts:
                items.append(RawDataItem(
                    source_type="financial_report", source_name="baostock",
                    title=f"{company_code} {label} 成长性",
                    content="；".join(parts), published_at=published))
                added_here = True
        if added_here:
            periods_filled += 1
    return items


def _fetch_sync(company_code: str, now: date) -> list[RawDataItem]:
    """真实走 baostock：登录 → 年报 + 季报 → 组装。异常向上抛给调用方降级。

    :param company_code: 股票代码。
    :param now: 当前日期，传给年报/季报收集。
    :return: 组装好的 RawDataItem 列表（年报 + 季报）。异常上抛，不发 _ANNUAL_ITEMS
        之外的期数。
    """
    import baostock as bs

    bs_code = _to_baostock_code(company_code)
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"baostock 登录失败：{login.error_code} {login.error_msg}")
    try:
        # 只产出「估值可用的年报」：营收/净利同比、经营现金流三者都齐才发 ——
        # 最老的一期没有相邻上期可算同比，滤掉，不然会吐一张缺数的年报。
        annuals = [a for a in _collect_annuals(bs, bs_code, now)
                   if a.get("rev_yoy_pct") is not None
                   and a.get("net_yoy_pct") is not None and a.get("cfo_yi") is not None]
        items = [_annual_item(company_code, a) for a in annuals[:_ANNUAL_ITEMS]]
        items.extend(_collect_quarters(bs, bs_code, company_code, now))
        return items
    finally:
        bs.logout()


async def fetch_financial_real(company: dict) -> list[RawDataItem]:
    """从 baostock 取财报。失败自动回落本地 fixture（不抛，不中断采集）。

    :param company: 公司 dict 用于读代码与回落 fixture。
    :return: 财报 RawDataItem 列表；失败或取空时回落同名 fixture 结果。
    """
    code = company["code"]
    try:
        items = await asyncio.to_thread(_fetch_sync, code, date.today())
    except Exception as e:                      # noqa: BLE001 —— 真实源不可用不该中断采集
        logger.warning("collect.baostock_failed", company=code, error=str(e))
        return await _fixture("financial_report", company)
    if not items:
        return await _fixture("financial_report", company)
    logger.info("collect.baostock_loaded", company=code, count=len(items))
    return items