# backend/agents/valuation/prompts.py
"""估值子图的提示词。

两类提示词：
  · build_extraction_prompt —— LLM 提数。⚠️ 这里的数字带引文（excerpt）是底线：
    每个提取的数值都要附支撑它的原文片段，否则错数无从查证。数值本身还会被
    facts.to_financial_facts 做交叉校验 + MISSING 判定，双重兜底。
  · build_valuation_prompt —— 合成。只让模型**选路径与写理由**，不产生任何数字。
    这不是靠这句话约束住的 —— 约束在 ValuationDecision 这个 schema 上（无数值字段）。
"""


def build_extraction_prompt(company_code: str, items: list[dict]) -> str:
    """构造提数提示词：列出每份资料，要求只提年度报告的财务数字，每个带原文引文。

    items 是 repo.fetch_data_items 的行，取 title/content。序号从 1 开始 ——
    与 nodes.extract_facts_node 里 source_titles = {i+1: ...} 对齐，保证出处可溯回资料。
    """
    lines = [f"标的：{company_code}", "",
             "下面是从采集阶段得到的若干份公司资料。请【仅针对【年度报告】】(不含半年报/季报/公告)做提取。",
             "要求：", "  · 每份被识别为年度报告的资料输出一行；",
             "  · 只填能从原文读到的数字，读不到就给 null，绝不编造；",
             "  · 总股本若出现在分红公告里也可提取（字段 total_shares_wan）；",
             "  · 每行必须附 #excerpt#：支撑这些数字的原文片段（照抄，别改写）。", "", "资料列表："]
    for i, item in enumerate(items, start=1):
        title = item.get("title") or ""
        content = (item.get("content") or "").strip()
        lines.append(f"\n[{i}] {title}\n{content}")
    lines.append("\n输出所有识别出的年度报告，按 year 升序。")
    return "\n".join(lines)


def build_valuation_prompt(company_code: str, industry: str, facts_lines: list[str],
                           dcf: dict, comparable: dict) -> str:
    return "\n".join([
        f"标的：{company_code}（{industry or '未标注行业'}）",
        "",
        "【估值输入】",
        *facts_lines,
        "",
        "【DCF 路径结果】",
        f"股权价值区间：{dcf['low']:.2f} ~ {dcf['high']:.2f} 亿元",
        "",
        "【可比公司法路径结果】",
        f"股权价值区间：{comparable['low']:.2f} ~ {comparable['high']:.2f} 亿元",
        "",
        "请判断用哪条路径，并说明理由。",
    ])