# backend/agents/analyze/nodes.py
# 分析子图：证据准备 → 四维并发 → 聚合评级。
#
# 四个维度节点由同一个工厂生成而不是写四份：它们的差异只有「看哪个源的数据」
# 和「用哪段提示词」，逻辑完全一致。写四份的话，改一处边界条件就要改四处。

from typing import Callable, Optional

from pydantic import BaseModel, Field

from backend.agents.analyze.prompts import build_analyze_prompt
from backend.agents.analyze.state import AnalyzeState
from backend.config import get_settings
from backend.core import research_repo as repo
from backend.core.llm_factory import get_structured_llm
from backend.core.logger import get_logger
from backend.core.research_rules import DIMENSIONS, aggregate_rating, judge_data_sufficiency
from backend.core.retry import FallbackResult, with_retry

logger = get_logger(__name__)

# 维度 → 主要数据源。一个维度只吃一个源，边界清晰：
# 「技术面」的结论只由行情/交易类材料支撑，不会因为当天新闻多就变好看。
DIMENSION_SOURCE: dict[str, str] = {
    "fundamental": "financial_report",
    "technical":   "announcement",
    "sentiment":   "news",
    "industry":    "industry",
}


class DimensionAnalysis(BaseModel):
    """LLM 的结构化输出契约。字段与写库所需的字段一一对应。"""
    score: float = Field(..., ge=0, le=100, description="0-100 的评分，50 为中性")
    conclusion: str = Field(..., max_length=500, description="不超过 150 字的中文结论")
    evidence: list[str] = Field(default_factory=list, description="2-4 条引用材料中的具体事实")


async def prepare_evidence_node(state: AnalyzeState) -> dict:
    """把父图传进来的 redo_targets 翻译成子图内部用的 redo_dimensions。

    为什么不在父图里直接放 redo_dimensions：那是分析子图的内部概念，
    父图不该知道它。父图只说「重做 analyze 阶段的这几个维度」，
    子图自己决定怎么解释它。

    :param state: AnalyzeState，读 redo_targets/tenant_id/task_id。
    :return: dict，含 current_stage="analyze"、redo_dimensions、dimension_results={}。
    """
    redo_targets = state.get("redo_targets") or {}
    redo_dimensions: list[str] = []
    if redo_targets.get("stage") == "analyze":
        redo_dimensions = [d for d in redo_targets.get("dimensions", []) if d in DIMENSIONS]

    # 先写业务表、再返回 State（设计文档 §4.4 的顺序规则）。
    # current_stage 必须真落库：前端进度条读的是这一列，不是图内存里的 State。
    # 本节点是 analyze 子图的入口，所以写 "analyze"；回边重做时会再写一次，这是对的。
    await repo.update_task_status(state["tenant_id"], state["task_id"], current_stage="analyze")
    await repo.record_stage_event(state["tenant_id"], state["task_id"], "analyze", "started")

    return {"current_stage": "analyze", "redo_dimensions": redo_dimensions,
            "dimension_results": {}}


def _evidence_lines(items: list[dict], limit: int = 12) -> list[str]:
    """把入库的条目压成提示词里的若干行。

    limit 限制送进提示词的材料条数：每条已经截到 2000 字，
    再多送会超出上下文窗口，而且边际信息量递减。

    :param items: 入库条目列表，每项含 source_name/timeliness_weight/reliability/
        title/content 键。
    :param limit: 最多取用的前 N 条材料；默认 12。
    :return: 若干条压平的证据行字符串列表。
    """
    return [
        f"[{item['source_name']}｜时效 {item['timeliness_weight']}｜可信度 {item['reliability']}]"
        f" {item['title']}：{item['content'][:600]}"
        for item in items[:limit]
    ]


def make_dimension_node(dimension: str) -> Callable:
    """维度节点工厂。返回的函数可以直接 add_node 到图里。

    :param dimension: 维度名，须在 DIMENSIONS 中，否则抛 ValueError。
    :return: 一个异步维度节点函数 dimension_node(state)。返回的函数被赋名
        analyze_{dimension}_node。
    """
    if dimension not in DIMENSIONS:
        raise ValueError(f"未知维度：{dimension}")

    async def dimension_node(state: AnalyzeState) -> dict:
        """执行单个维度的分析：取数 → 判充分性 → LLM 打分（或落 NULL）。

        :param state: AnalyzeState，读 task_id/tenant_id/redo_dimensions/company_code/industry。
        :return: dict，含 dimension_results={维度: {...}}；被重做跳过的维度返回空 dict。
            未参与重做时返回空 dict 表示不改状态。
        """
        task_id, tenant_id = state["task_id"], state["tenant_id"]
        redo_dimensions = state.get("redo_dimensions") or []

        # 驳回重做时，没被点名的维度直接复用上一版 —— 既省一次 LLM 调用，
        # 也保证「没被质疑的结论不发生漂移」。返回空 dict 表示「本节点不改任何状态」。
        if redo_dimensions and dimension not in redo_dimensions:
            logger.info("analyze.dimension_skipped", dimension=dimension,
                        redo_dimensions=redo_dimensions)
            return {}

        items = await repo.fetch_data_items(tenant_id, task_id,
                                            source_type=DIMENSION_SOURCE[dimension])
        weights = [float(i["timeliness_weight"]) for i in items]
        sufficiency = judge_data_sufficiency(weights, get_settings().research_evidence_min_count)

        # 数据不足：写 NULL 分数，不调 LLM。
        # 【绝不补零】：0 分是一个真实的负面结论，而我们此刻的真实结论是「没有结论」。
        if sufficiency == "insufficient":
            await repo.upsert_dimension_analysis(
                tenant_id, task_id, dimension, score=None,
                conclusion=f"可用材料不足（{len(items)} 条），本维度不参与综合评级。",
                evidence=[], data_sufficiency=sufficiency,
            )
            logger.info("analyze.dimension_insufficient", dimension=dimension, items=len(items))
            return {"dimension_results": {dimension: {
                "score": None, "data_sufficiency": sufficiency,
                "conclusion": "可用材料不足，本维度不参与综合评级", "evidence": [],
            }}}

        prompt = build_analyze_prompt(dimension, state["company_code"],
                                      state.get("industry") or "", _evidence_lines(items))

        structured = get_structured_llm("analyze", DimensionAnalysis)

        @with_retry(agent_type="analyze")
        async def _invoke():
            """携带闭包 paged prompt 调 LLM；无参数。返回结构化解析结果。"""
            return await structured.ainvoke(prompt)

        try:
            parsed = await _invoke()
        except Exception as e:                       # noqa: BLE001 —— 降级也得留一条记录
            logger.error("analyze.dimension_failed", dimension=dimension, error=str(e))
            parsed = None

        # 降级契约：拿不到合法的结构化结果就落 NULL，绝不解析兜底结构里的内容。
        # 兜底结构里的文字是给用户看的提示语，不是分析结论 —— 把它当分数写进去就是编造。
        fallback_used = not isinstance(parsed, DimensionAnalysis)
        score = None if fallback_used else float(parsed.score)
        if not fallback_used:
            conclusion = parsed.conclusion
        elif isinstance(parsed, FallbackResult):
            # 用哨兵自带的说明，不在这里另写一句：note 区分了「该 Agent 的降级
            # 策略生效」与「连降级都没兜住」，这两件事对排查指向完全不同的地方，
            # 节点自己编文案就把这个区分丢了。
            conclusion = parsed.note
        else:
            # _invoke 抛了（不可重试 / 必须阻断的异常），连哨兵都没拿到。
            conclusion = "本维度分析服务暂时不可用，未参与综合评级。"
        evidence = [] if fallback_used else list(parsed.evidence)

        await repo.upsert_dimension_analysis(
            tenant_id, task_id, dimension, score=score, conclusion=conclusion,
            evidence=evidence,
            data_sufficiency="insufficient" if fallback_used else sufficiency,
        )
        return {"dimension_results": {dimension: {
            "score": score,
            "data_sufficiency": "insufficient" if fallback_used else sufficiency,
            "conclusion": conclusion, "evidence": evidence,
            "fallback_used": fallback_used,
        }}}

    dimension_node.__name__ = f"analyze_{dimension}_node"
    return dimension_node


async def aggregate_rating_node(state: AnalyzeState) -> dict:
    """聚合四维结果，算出综合评级。

    结果来源是「库里当前有效的 + 本轮新写的」的并集：
    驳回重做时只有被点名的维度出现在本轮 State 里，其余三个必须从库里补回来。

    :param state: AnalyzeState，读 task_id/tenant_id/dimension_results。
    :return: dict，含 dimension_results（并集）、rating、rating_note、has_buy_sell_advice。
    """
    task_id, tenant_id = state["task_id"], state["tenant_id"]

    db_results = await repo.fetch_current_dimension_results(tenant_id, task_id)
    # 「库里当前有效的 + 本轮新写的」的并集：驳回重做时只有被点名维度在本轮 State 里，
    # 其余三个必须从库里补回来。先拷库里的，再让本轮的结果覆盖进来（同 key 以本轮为准）。
    merged = dict(db_results)
    merged.update(state.get("dimension_results") or {})

    rating, note, has_advice = aggregate_rating(merged)

    await repo.record_stage_event(
        tenant_id, task_id, "analyze", "success",
        {"rating": rating, "dimensions": {
            d: merged.get(d, {}).get("score") for d in DIMENSIONS}},
    )
    logger.info("analyze.aggregated", task_id=task_id, rating=rating, advice=has_advice)

    return {
        "dimension_results": merged,
        "rating": rating,
        "rating_note": note,
        "has_buy_sell_advice": has_advice,
    }
