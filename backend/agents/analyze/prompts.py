# backend/agents/analyze/prompts.py
# 四维分析的提示词。四个维度共用一套模板，只换角色描述与判断要点 ——
# 分成四份高度重复的长提示词，改一处就要改四处，是维护陷阱。

DIMENSION_LABELS: dict[str, str] = {
    "fundamental": "基本面",
    "technical":   "技术面",
    "sentiment":   "舆情面",
    "industry":    "行业面",
}

_DIMENSION_GUIDES: dict[str, str] = {
    "fundamental": "关注营收与利润增速、毛利率与净利率的变化趋势、经营性现金流与净利润的匹配度、"
                   "资产负债结构与有息负债水平。",
    "technical":   "关注价格趋势与成交量配合、关键区间的支撑与压力、资金流向变化。"
                   "只依据提供的材料，不要引用未给出的行情数据。",
    "sentiment":   "关注媒体与机构的关注度变化、舆论的正负面倾向、是否存在尚未落地的传闻。"
                   "对未经证实的传闻必须明确标注。",
    "industry":    "关注行业景气度、竞争格局与集中度变化、政策与监管动向、上下游议价能力。",
}

ANALYZE_PROMPT_TEMPLATE = """你是一名严谨的证券研究分析师，正在为「{company_code}」{industry}行业的标的撰写{dimension_label}分析。

{dimension_guide}

以下是本维度可用的全部材料（已按时效与可信度排序）：

{evidence}

请给出本维度的判断。输出形状由系统指定的 schema 约束：按该结构返回分数、结论与证据。
【只依据上面给出的材料打分】。材料不足以支撑判断时，也要给出你认为最合理的分数，
由系统根据材料数量另行判断是否采信；证据只引用材料中的具体事实（含数字），
【不要出现材料中不存在的数据】。

特别注意：本报告将用于投资研究参考。任何编造的数据都会导致读者做出错误判断，
因此宁可结论保守，也不要补充材料中没有的信息。
"""


def build_analyze_prompt(dimension: str, company_code: str, industry: str,
                         evidence_lines: list[str]) -> str:
    """把某一维度的证据材料拼成完整提示词。"""
    evidence = "\n".join(f"- {line}" for line in evidence_lines) or "（本维度暂无材料）"
    return ANALYZE_PROMPT_TEMPLATE.format(
        company_code=company_code,
        industry=industry or "未标注",
        dimension_label=DIMENSION_LABELS[dimension],
        dimension_guide=_DIMENSION_GUIDES[dimension],
        evidence=evidence,
    )
