# backend/agents/retrieve/prompts.py
"""研报检索的对比点提炼提示词。

⚠️ 输出契约 ComparisonExtraction 只有 points: list[str]，没有数值字段 ——
引用历史研报里的数字时，数字来自检索到的原文片段，不来自模型的计算。
"""


def build_extract_prompt(company_code: str, industry: str,
                         chunks: list[dict]) -> str:
    lines = [f"本次研究标的：{company_code}（{industry or '未标注行业'}）", ""]
    for i, chunk in enumerate(chunks, start=1):
        label = "横向·可比公司" if chunk["direction"] == "horizontal" else "纵向·本公司历史"
        lines.append(f"[片段 {i}｜{label}｜来源：{chunk.get('source') or '语料库'}]")
        lines.append(chunk["content"])
        lines.append("")
    lines.append("请提炼可比对要点。")
    return "\n".join(lines)