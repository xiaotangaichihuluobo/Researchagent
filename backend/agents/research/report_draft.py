# backend/agents/research/report_draft.py
# 研报草稿的拼装：把业务表里**已经有的事实**组织成可读文本。
#
# 为什么是「拼装」而不是「生成」：这一版没有多调用一次 LLM，也不编造任何内容。
# 四维结论、证据、评级、数据充分性都已经在 dimension_analyses 里，这里只是换个排布。
# 生成式叙述属于 P5 之后的活；在那之前，一份「如实但朴素」的草稿远好过一份
# 「流畅但没人知道哪句是编的」草稿 —— 投研域的读者无法分辨后者。
#
# 位置说明：风控子图的 persist_draft 需要它（草稿必须在人工闸门之前落库），
# 但它描述的是投研域的产出，所以放在 research 包下，不放进 risk 包。

from typing import Optional

# 维度中文名：正文给人看，不能出现 fundamental 这种内部标识
DIMENSION_LABELS = {
    "fundamental": "基本面",
    "technical":   "技术面",
    "sentiment":   "市场情绪",
    "industry":    "行业景气",
}

SUFFICIENCY_LABELS = {
    "sufficient":   "充分",
    "partial":      "部分",
    "insufficient": "不足",
}


def build_title(company_name: Optional[str], company_code: str, rating: str) -> str:
    """拼研报标题：公司名（若空则用代码）加代码与评级。

    :param company_name: 公司名称；为 None 时回退用 company_code 占位。
    :param company_code: 股票代码，写进标题。
    :param rating: 综合评级，写进标题。
    :return: 标题字符串。
    """
    name = company_name or company_code
    return f"{name}（{company_code}）投资研究报告 —— 评级 {rating}"


def build_content(company_name: Optional[str], company_code: str, industry: str,
                  rating: str, rating_note: str,
                  dimensions: dict, has_reference: bool,
                  valuation_available: bool,
                  comparison_points: Optional[list[str]] = None,
                  valuation: Optional[dict] = None) -> str:
    """按「评级 → 四维 → 参照 → 估值」的顺序拼正文。

    dimensions 形如 fetch_current_dimension_results() 的返回值。
    缺失的维度不省略、不补零：明确写「未产出」—— 读者必须能看出哪一块是空的。

    comparison_points 是历史参照一节的条目；valuation 是估值子图的真实产物
    （键见 valuation 子图 State）。三者【只做排布，不猜数字】：估值不可用
    就如实写「不包含目标价」，绝不给出个默认区间。

    :param company_name: 公司名称；为 None 时用 company_code 占位。
    :param company_code: 股票代码，写进标题行。
    :param industry: 所属行业；空串则写「未标注」。
    :param rating: 综合评级。
    :param rating_note: 评级依据简述；空串则省略该行。
    :param dimensions: 四维结论 dict，形如 fetch_current_dimension_results() 的返回值，
        键为四大维度，值为含 score/data_sufficiency/conclusion/evidence 的 dict。
    :param has_reference: 是否检索到可比历史研报，决定历史参照一节的写法。
    :param valuation_available: 估值是否可用，决定估值一节是否给出区间。
    :param comparison_points: 历史参照要点列表；默认 None。
    :param valuation: 估值子图产物 dict（含 equity_value_low/high、per_share_low/high、
        method、assumptions、rationale 等键）；默认 None。
    :return: 拼装完成的研报正文 markdown 字符串。
    """
    lines: list[str] = []
    name = company_name or company_code

    lines.append(f"# {name}（{company_code}）投资研究报告")
    lines.append("")
    lines.append(f"所属行业：{industry or '未标注'}")
    lines.append(f"综合评级：{rating}")
    if rating_note:
        lines.append(f"评级说明：{rating_note}")
    lines.append("")

    lines.append("## 一、多维分析结论")
    lines.append("")
    if not dimensions:
        lines.append("本次未产出任何维度的分析结论。")
    for dim, label in DIMENSION_LABELS.items():
        item = dimensions.get(dim)
        lines.append(f"### {label}")
        if not item:
            lines.append("未产出（该维度本次没有可用结论）。")
            lines.append("")
            continue
        score = item.get("score")
        sufficiency = SUFFICIENCY_LABELS.get(item.get("data_sufficiency"), "未知")
        lines.append(f"评分：{score if score is not None else '未评分'}（数据充分性：{sufficiency}）")
        lines.append(f"结论：{item.get('conclusion') or '（无）'}")
        evidence = item.get("evidence") or []
        if evidence:
            lines.append("证据：")
            lines.extend(f"  - {e}" for e in evidence)
        lines.append("")

    lines.append("## 二、历史研报参照")
    lines.append("")
    if not has_reference:
        lines.append("本次未检索到可对比的历史研报。")
    else:
        lines.append("本次检索到以下可对比的历史研报要点：")
        lines.append("")
        for point in (comparison_points or []):
            lines.append(f"  - {point}")
    lines.append("")

    lines.append("## 三、估值")
    lines.append("")
    if not valuation_available or not valuation:
        lines.append("本次未产出可用估值，报告不包含目标价或估值区间 —— "
                     "估值不可用时绝不给出默认数字。")
    else:
        low, high = valuation["equity_value_low"], valuation["equity_value_high"]
        lines.append(f"股权价值区间：{low:.2f} ~ {high:.2f} 亿元"
                     f"（方法：{valuation.get('method')}）")
        per_low, per_high = valuation.get("per_share_low"), valuation.get("per_share_high")
        if per_low is not None and per_high is not None:
            lines.append(f"对应每股价值区间：{per_low:.2f} ~ {per_high:.2f} 元/股")
        else:
            # 总股本没提取到 —— 只给股权价值，不留一行空的每股区间。
            lines.append("每股价值：本次未取到总股本，不提供每股口径。")
        lines.append("")
        lines.append(f"估值理由：{valuation.get('rationale') or '（无）'}")
        lines.append("")
        lines.append("估值假设与来源：")
        for item in (valuation.get("assumptions") or []):
            lines.append(f"  - {item['name']}：{item['value']}（来源：{item['source']}）")
    lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def build_risk_disclosure(dimensions: dict, has_reference: bool,
                          valuation_available: bool) -> str:
    """风险揭示由事实生成，不是固定套话。

    哪几个维度数据不足、估值有没有、历史参照有没有 —— 这些都是审阅人必须一眼看到的
    局限，写成套话就等于没写。

    :param dimensions: 四维结论 dict，用于统计哪些维度未产出或数据充分性未达标。
    :param has_reference: 是否有历史参照，决定参照风险一句的措辞。
    :param valuation_available: 估值是否可用，决定估值风险一句的措辞。
    :return: 风险揭示段落字符串（多行、由换行连接）。
    """
    lines = ["本报告由自动化流程生成，结论仅供研究参考，不构成投资建议；"
             "最终发布以人工签字为准。", ""]

    weak = [DIMENSION_LABELS.get(d, d) for d, item in (dimensions or {}).items()
            if (item or {}).get("data_sufficiency") != "sufficient"]
    missing = [label for dim, label in DIMENSION_LABELS.items() if dim not in (dimensions or {})]

    if missing:
        lines.append(f"数据局限：{'、'.join(missing)} 维度本次未产出结论。")
    if weak:
        lines.append(f"数据局限：{'、'.join(weak)} 维度的数据充分性未达「充分」，结论稳定性有限。")
    if not missing and not weak:
        lines.append("数据局限：各维度数据充分性均达标。")

    lines.append("估值风险：本次未产出可用估值，报告不含目标价；"
                 "请勿据此推断隐含估值。"
                 if not valuation_available else
                 "估值风险：估值结论依赖模型假设，假设变化会显著改变结果。")
    lines.append("参照风险：本次无历史研报可比对，无法说明结论相对上期的变化。"
                 if not has_reference else
                 "参照风险：历史研报结论与本期可能存在口径差异。")

    return "\n".join(lines)
