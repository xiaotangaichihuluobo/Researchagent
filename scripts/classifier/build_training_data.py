# scripts/classifier/build_training_data.py
"""重建出 Query 分类器的投研增强训练数据（general / specialized 二分类）。

背景：线上 classifier 把所有投研类查询（泸州老窖分析、舍得酒业股票分析等）误判成
general，根因是训练时 specialized 侧缺「个股/板块/财报/估值」类样本。本脚本用投研实体
（白酒股、各行各股、板块）× 分析意图词交叉展开成 specialized 正样本，配一组 diverse 的
general 负样本，生成一份 balanced 的 JSONL。

产出：scripts/classifier/training_data.jsonl，每行 {"text": ..., "label": "general|specialized"}。
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "training_data.jsonl"

# ── specialized 实体：投研类问题的主角 ───────────────────────────
SPEC_ENTITIES = [
    # 白酒股
    "泸州老窖", "舍得酒业", "贵州茅台", "山西汾酒", "五粮液", "古井贡酒",
    "洋河股份", "今世缘", "酒鬼酒", "水井坊", "口子窖", "迎驾贡酒",
    # 其他行业个股
    "比亚迪", "宁德时代", "腾讯控股", "阿里巴巴", "美的集团", "海天味业",
    "片仔癀", "招商银行", "中国平安", "中芯国际", "隆基绿能", "牧原股份",
    "万华化学", "恒瑞医药", "东方财富", "格力电器", "长城汽车", "海康威视",
    # 板块 / 宏观
    "白酒板块", "新能源车板块", "半导体板块", "医药板块", "A股市场", "港股",
]

# ── specialized 意图词：跟在实体后表示「要投研分析」 ──────────────
SPEC_PURPOSES = [
    "分析", "股票分析", "基本面分析", "财务分析", "估值", "市盈率", "年报",
    "财报", "营收和净利润", "毛利率", "资产负债率", "现金流", "股价走势",
    "技术分析", "研报", "行业地位", "投资价值", "未来前景", "风险提示",
    "目标价", "评级", "一季度业绩", "股息率", "成长性", "估值水平",
]

# ── 句式模板（轮流取用，控制模板化程度） ──────────────────────────
SPEC_TEMPLATES = [
    lambda e, p: f"{e}{p}",
    lambda e, p: f"{e}的{p}",
    lambda e, p: f"{e}{p}怎么样",
    lambda e, p: f"帮我分析一下{e}的{p}",
    lambda e, p: f"讲讲{e}的{p}",
    lambda e, p: f"{e}{p}怎么看",
]

# 关键锚点：线上误判样本用更多句式强化，确保训练后必为 specialized
ANCHOR_EXAMPLES = [
    "泸州老窖分析", "泸州老窖股票分析", "泸州老窖这家公司怎么样",
    "分析一下泸州老窖", "泸州老窖的基本面怎么样", "泸州老窖估值贵不贵",
    "舍得酒业股票分析", "舍得酒业分析", "舍得酒业还能买吗",
    "贵州茅台值得投资吗", "贵州茅台的市盈率", "山西汾酒的一季度业绩",
    "五粮液和泸州老窖哪个更有投资价值", "白酒板块现在值得入手吗",
    "比亚迪的估值水平", "宁德时代的三季报点评",
]

# ── general 负样本：闲聊 / 百科 / 生活，刻意不含投研信号 ──────────
# 普通（非投研）实体：物件/食品/学科/职业/城市/动物/活动，保证不含股票·财务·估值词
GENERAL_BASES = [
    "红烧肉", "咖啡", "吉他", "跑步", "宠物猫", "金毛犬", "洗衣机", "空调",
    "北京", "上海", "重庆", "巴黎", "考研", "面试", "驾照", "宝宝辅食",
    "电动车", "手机", "相机", "手表", "行李箱", "羽绒服", "面膜", "防晒霜",
    "微波炉", "空气炸锅", "水饺", "奶茶", "小龙虾", "火锅", "蛋糕", "徒步",
    "瑜伽", "游泳", "钢琴", "摄影", "园艺", "浇花", "java", "数据库",
]

# 非投研的百科 / 生活询问句式（避开「估值/评级/买入」等可撞投研语义的词）
GENERAL_QTEMPLATES = [
    lambda b: f"{b}是什么",
    lambda b: f"{b}怎么做",
    lambda b: f"{b}是怎么做的",
    lambda b: f"{b}有什么好处",
    lambda b: f"介绍下{b}",
    lambda b: f"{b}怎么挑选",
    lambda b: f"{b}和水饺哪个好吃",
]

# 手工高信号 general：闲聊 / 百科 / 生活，刻意不含投研措辞
GENERAL_EXAMPLES = [
    "你好", "嗨", "早上好", "谢谢", "再见", "在吗", "请问在吗",
    "今天天气怎么样", "明天会下雨吗", "推荐一部好看的电影", "有什么好听的歌",
    "怎么做红烧肉", "西红柿炒鸡蛋的做法", "推荐一本小说", "讲个笑话",
    "什么是Python", "Python怎么装", "Spring IOC是什么", "什么是多线程",
    "什么是RESTful API", "SQL索引怎么建", "怎么学英语", "怎么减肥",
    "什么是黑洞", "光速是多少", "地球有多少岁", "什么是二次元",
    "推荐一个好用的笔记软件", "怎么提高工作效率", "什么是幸福", "人生的意义是什么",
    "怎么和同事相处", "今天星期几", "你会写诗吗", "介绍一下你自己",
]


def build() -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()

    def add(text: str, label: str) -> None:
        t = text.strip()
        if t and t not in seen:
            seen.add(t)
            rows.append({"text": t, "label": label})

    # specialized：实体 × 意图 交叉，句式轮换
    for i, e in enumerate(SPEC_ENTITIES):
        for p in SPEC_PURPOSES:
            tmpl = SPEC_TEMPLATES[(i + len(p)) % len(SPEC_TEMPLATES)]
            add(tmpl(e, p), "specialized")
    # 手工锚点强化
    for ex in ANCHOR_EXAMPLES:
        add(ex, "specialized")

    # general：程序化百科/生活句式 + 手工闲聊
    for b in GENERAL_BASES:
        for t in GENERAL_QTEMPLATES:
            add(t(b), "general")
    for g in GENERAL_EXAMPLES:
        add(g, "general")

    return rows


def main() -> None:
    rows = build()
    n_spec = sum(1 for r in rows if r["label"] == "specialized")
    n_gen = sum(1 for r in rows if r["label"] == "general")
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"written {OUT} → specialized={n_spec}, general={n_gen}, total={len(rows)}")


if __name__ == "__main__":
    main()