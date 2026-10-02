# backend/core/research_rules.py
# 投研域业务规则：时效降权、来源可信度、证据充分性、综合评级。
#
# 为什么单独成一个模块：这四条规则是纯函数 —— 不碰 DB、不碰 LLM、不碰图。
# 纯函数才能被穷举测试（见 tests/test_research_rules.py），也才能在调参时
# 只改一处。把它们散进各 Agent 节点里，就再也没法验证「系数改了会怎样」。

from datetime import datetime, timezone

from backend.core.timeutil import CN_TZ, cn_now
from typing import Optional

from backend.config import get_settings
from backend.core.logger import get_logger

logger = get_logger(__name__)

# ── 四个分析维度 ────────────────────────────────────────────────
# 次序即权重表的次序；新增维度必须同时改这里的元组、DIMENSION_WEIGHTS
# 与数据库 dimension_analyses.dimension 的 CHECK 约束，三处必须一致。
DIMENSIONS: tuple[str, ...] = ("fundamental", "technical", "sentiment", "industry")

# ── 综合评级权重（设计文档 §5.2；属「待实测调优」项，改了不影响代码结构）──
DIMENSION_WEIGHTS: dict[str, float] = {
    "fundamental": 0.35,   # 基本面：投研的核心依据，权重最高
    "technical":   0.20,   # 技术面
    "sentiment":   0.20,   # 舆情面
    "industry":    0.25,   # 行业面
}

# ── 来源可信度分档：公告 > 财报 > 行业数据 > 新闻 ───────────────
# 公告是公司自己的正式披露，财报次之，行业数据为第三方统计，
# 新闻是二手转述（可能被加工、断章取义），可信度最低。
RELIABILITY_BY_SOURCE: dict[str, float] = {
    "announcement":     1.0,
    "financial_report": 0.9,
    "industry":         0.8,
    "news":             0.6,
}
_UNKNOWN_SOURCE_RELIABILITY = 0.6   # 不认识的来源按最低档处理，不给中间值

# ── 评级文案与阈值 ──────────────────────────────────────────────
RATING_BUY     = "买入"
RATING_OBSERVE = "观察"
RATING_SELL    = "卖出"


def compute_timeliness_weight(
    published_at: Optional[datetime],
    now: Optional[datetime] = None,
) -> float:
    """把数据发布时间换算成时效权重（0.3 ~ 1.0）。

    规则（设计文档 §5.1）：
        days ≤ 30          → 1.0                                  「最新」
        30 < days < 约286  → 1.0 - (days - 30) / 365              线性衰减
        更久               → 0.3                                  下限

    为什么有下限而不是衰减到 0：一份 5 年前的财报并非毫无价值 ——
    它仍是这家公司的真实历史，只是不足以单独支撑当下的判断。
    下限调到 0 会让「老数据」与「没有数据」变得无法区分，
    而这两者在业务上必须区分：前者算「证据偏旧」，后者算「数据不足」。

    :param published_at: 数据发布时间；缺失时按最低权重处理
    :param now: 基准时刻，默认当前 UTC 时间（测试时注入以获得确定性）
    :return: [floor, 1.0] 区间内的浮点数
    """
    settings = get_settings()
    if now is None:
        now = cn_now()

    # 没有发布时间 = 无法证明它新 → 直接给下限，走「证据偏旧」路径
    if published_at is None:
        logger.debug("rules.timeliness_missing_published_at")
        return settings.research_timeliness_floor

    # 统一到 aware datetime，避免 naive/aware 相减抛 TypeError
    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=timezone.utc)

    days = (now - published_at).total_seconds() / 86400.0

    # 未来时间（数据源时间戳错误）当作最新处理，不产生 > 1.0 的权重
    if days <= settings.research_timeliness_full_days:
        return 1.0

    decayed = 1.0 - (days - settings.research_timeliness_full_days) / settings.research_timeliness_decay_days
    # 不在这里 round：DB 列本就是 NUMERIC(3,2)，存库时自然会取整。
    # 规则函数里再取一次整只会削掉信息 —— judge_data_sufficiency 要拿这个值
    # 去和 0.5 比，先取整会让边界判定失真；而且 d=31 会 round 成恰好 1.0，
    # 把「衰减段」和「满权」混成同一个值。
    return max(settings.research_timeliness_floor, decayed)


def compute_reliability(source_type: str) -> float:
    """按来源类型返回可信度（公告 1.0 / 财报 0.9 / 行业 0.8 / 新闻 0.6）。

    :param source_type: 来源类型；不认识的类型按最低档 0.6 处理
    :return: 该来源类型的可信度浮点数 [0, 1]
    """
    return RELIABILITY_BY_SOURCE.get(source_type, _UNKNOWN_SOURCE_RELIABILITY)


def judge_data_sufficiency(weights: list[float], min_count: int) -> str:
    """判定某个维度的证据是否够用。

    返回三档，语义必须分清楚：
        insufficient —— 证据太少或全部过期，**不打分**（score 置 NULL，绝不补零）
        partial      —— 条数够但有效证据不足半数，照常打分，研报标注「证据偏旧」
        sufficient   —— 有效证据不少于半数，正常打分

    为什么把「全部过期」与「部分过期」分开：前者说明这个维度当下无据可依，
    给分就是编；后者只是数据不够新，结论仍然成立，但读者有权知道。

    :param weights: 该维度全部证据的时效权重
    :param min_count: 最少证据条数（来自配置）
    :return: 'insufficient' | 'partial' | 'sufficient' 三档之一
    """
    settings = get_settings()

    if len(weights) < min_count:
        return "insufficient"                     # 条数不够，与新旧无关

    fresh = sum(1 for w in weights if w >= settings.research_timeliness_valid)
    if fresh == 0:
        return "insufficient"                     # 一条有效证据都没有
    if fresh * 2 < len(weights):
        return "partial"                          # 有效证据不足半数
    return "sufficient"


def aggregate_rating(results: dict[str, dict]) -> tuple[str, str, bool]:
    """把四维结果聚合成综合评级。

    :param results: 四维结果字典，键为维度名，值为
        {"score": float | None, "data_sufficiency": str}
    :return: 三元组 (rating, note, has_buy_sell_advice)。has_buy_sell_advice=False
        表示本次只给「观察」，不得出具买入/卖出建议

    两条业务硬规则写在这里，而不是散在校验代码里：
      1. 任一维 insufficient → 综合评级强制「观察」，且不出具买卖建议。
         数据不足时给出投资建议在业务上是不允许的。
      2. insufficient 的维度从加权中【剔除并重归一化】，绝不当 0 分。
         当 0 分会让综合分凭空掉一截，产生一个虚假的低分结论。
    """
    settings = get_settings()

    insufficient_dims = [
        d for d, r in results.items()
        if not r or r.get("data_sufficiency") == "insufficient"
    ]
    if insufficient_dims:
        note = (
            f"以下维度数据不足，综合评级强制降级为「{RATING_OBSERVE}」，"
            f"本次不出具买入/卖出建议：{'、'.join(insufficient_dims)}"
        )
        # 硬规则 2 在这里也要落地：把 insufficient 的维度剔除后【重归一化】算一遍分数，
        # 只写进 note 留档复核，不用它定评级（评级已被硬规则 1 锁死为「观察」）。
        # 不这么做的话，「不补零」就只写在文档里、代码里根本没有可被验证的落点。
        _usable = [d for d, r in results.items()
                   if r and r.get("score") is not None and d in DIMENSION_WEIGHTS]
        if _usable:
            _weight_sum = sum(DIMENSION_WEIGHTS[d] for d in _usable)
            _score = sum(results[d]["score"] * DIMENSION_WEIGHTS[d] for d in _usable) / _weight_sum
            note += (
                f"（剔除数据不足维度后重归一化的综合分 {round(_score, 2)}，"
                f"仅留档复核，不构成建议）"
            )
        logger.info("rules.rating_forced_observe", insufficient=insufficient_dims)
        return RATING_OBSERVE, note, False

    # 只对「算得出分」的维度加权，权重重归一化 —— 这是「不补零」的落点
    usable = {
        d: r for d, r in results.items()
        if r and r.get("score") is not None and d in DIMENSION_WEIGHTS
    }
    if not usable:
        logger.warning("rules.rating_no_usable_dimension")
        return RATING_OBSERVE, "四个维度均无可用评分，综合评级为「观察」", False

    weight_sum = sum(DIMENSION_WEIGHTS[d] for d in usable)
    weighted = sum(r["score"] * DIMENSION_WEIGHTS[d] for d, r in usable.items()) / weight_sum
    weighted = round(weighted, 2)

    if weighted >= settings.research_rating_buy_threshold:
        rating, advice = RATING_BUY, True
    elif weighted < settings.research_rating_sell_threshold:
        rating, advice = RATING_SELL, True
    else:
        rating, advice = RATING_OBSERVE, True

    # note 里必须带出算出来的分数，否则评级结论无法被复核
    note = (
        f"综合分 {weighted}（参与加权维度：{'、'.join(sorted(usable))}），"
        f"评级为「{rating}」"
    )
    logger.info("rules.rating_done", rating=rating, weighted=weighted, dims=sorted(usable))
    return rating, note, advice


if __name__ == '__main__':
    # 手工核对分档：python -m backend.core.research_rules
    from datetime import timedelta

    base = datetime(2026, 9, 17, tzinfo=CN_TZ)
    for d in (0, 30, 31, 100, 200, 300, 400, 3650):
        w = compute_timeliness_weight(base - timedelta(days=d), now=base)
        print(f"{d:>5} 天前 → timeliness_weight={w}")
    print("---")
    print("评级（满分）:", aggregate_rating({
        d: {"score": 90.0, "data_sufficiency": "sufficient"} for d in DIMENSIONS
    }))
    print("评级（舆情维数据不足）:", aggregate_rating({
        "fundamental": {"score": 95.0, "data_sufficiency": "sufficient"},
        "technical":   {"score": 92.0, "data_sufficiency": "sufficient"},
        "sentiment":   {"score": None, "data_sufficiency": "insufficient"},
        "industry":    {"score": 90.0, "data_sufficiency": "sufficient"},
    }))
