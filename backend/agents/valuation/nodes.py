# backend/agents/valuation/nodes.py
# 估值子图：装配输入 → 双路并发 → 合成。
#
# 硬约束，贯穿本文件：
#   ① 提数阶段的数字由 LLM 提取，但须过 facts.to_financial_facts 的交叉校验 + MISSING
#     判定才进计算（LLM 裸数字不可信，见 facts.py 头注）。
#   ② 合成阶段：LLM 只选路径 + 写理由，不经模型产生新数字（ValuationDecision 无数值
#     字段，是结构性事实）。
#   ③ 任何一段失败 ⇒ valuation_available=False、四个数值列全 NULL。绝不补零。
#
# 关于 with_retry 的挂载形态：只包【节点内部的闭包】，且闭包内无副作用。
# record_stage_event 是 append-only 的 INSERT，把重试范围放大到节点体会重复写审计流水。

from typing import Literal, Optional

from pydantic import BaseModel, Field

from backend.agents.valuation.facts import (
    MISSING_FIELDS, ExtractionResult, FinancialFacts, to_financial_facts,
)
from backend.agents.valuation.prompts import build_extraction_prompt, build_valuation_prompt
from backend.agents.valuation.state import ValuationState
from backend.config import get_settings
from backend.core import research_repo as repo
from backend.core.llm_factory import get_structured_llm
from backend.core.logger import get_logger
from backend.core.retry import FallbackResult, with_retry
from backend.core.valuation_model import (
    Range, per_share, run_comparable, run_dcf, select_range,
)

logger = get_logger(__name__)

VALID_METHODS = ("dcf", "comparable", "blended")


class ValuationDecision(BaseModel):
    """LLM 的输出契约。

    【刻意不含任何数值字段】。这样「不编造数字」不是一条要靠提示词被遵守的祈愿，
    而是结构性事实：数字根本不经过模型。tests/test_valuation_agent.py 里有一条
    断言直接钉住这一点 —— 往这个类里加一个 float 字段，那条测试就会红。
    """
    method: Literal["dcf", "comparable", "blended"] = Field(
        ..., description="估值路径。必须严格填 dcf / comparable / blended 三者之一的英文，不得写中文")
    rationale: str = Field(..., max_length=800, description="选择该路径的理由，不超过 300 字")


async def extract_facts_node(state: ValuationState) -> dict:
    """LLM 提数：从采集结果里提取财务事实，写回 state["financial_facts"]。

    提数走结构化 LLM（ExtractionResult），但裸数字不回直接进计算 —— 提取后由
    facts.to_financial_facts 做确定性交叉校验与 MISSING 判定，这两步挡在 LLM 与
    DCF 之间，防止「错得自然的数字」混进估值。

    纯 LLM、不保留正则兜底：LLM 失败/输出非法 ⇒ financial_facts=None，调用方据此降级。

    :param state: ValuationState，读 tenant_id/task_id/company_code。
    :return: dict，含 financial_facts（FinancialFacts 或 None）与 degrade_reason
        （None 表示成功提取）。
    """
    tenant_id, task_id = state["tenant_id"], state["task_id"]

    await repo.update_task_status(tenant_id, task_id, current_stage="valuation")
    await repo.record_stage_event(tenant_id, task_id, "valuation", "started")

    items = await repo.fetch_data_items(tenant_id, task_id)
    if not items:
        return {"financial_facts": None, "degrade_reason": "未找到任何采集数据"}

    prompt = build_extraction_prompt(state.get("company_code", ""), items)
    structured = get_structured_llm("valuation", ExtractionResult)

    @with_retry(agent_type="valuation")
    async def _invoke():
        """带闭包 prompt 调提取 LLM；无参数。返回 ExtractionResult。"""
        return await structured.ainvoke(prompt)

    try:
        parsed = await _invoke()
    except Exception as e:                       # noqa: BLE001 —— 降级也得能走出去
        logger.error("valuation.extract_failed", task_id=task_id, error=str(e))
        return {"financial_facts": None, "degrade_reason": f"财务提取失败：{e}"}

    if not isinstance(parsed, ExtractionResult):
        note = (parsed.note if isinstance(parsed, FallbackResult)
                else "财务提取服务不可用")
        return {"financial_facts": None, "degrade_reason": f"财务提取失败：{note}"}

    rows = parsed.docs
    if not rows:
        return {"financial_facts": None, "degrade_reason": "未找到任何年度报告"}

    # 出处要对齐到真实资料：LLM 给的 index → 那条 item 的标题。
    source_titles = {i + 1: (item.get("title") or "")
                     for i, item in enumerate(items)}
    # 交叉校验失败不整体作废：只废增速这一项，好让下游 rationale 说清缺的是什么
    facts = to_financial_facts(rows, source_titles, tolerate_cross_check_failure=True)
    return {"financial_facts": facts, "degrade_reason": None}


async def prepare_inputs_node(state: ValuationState) -> dict:
    """装配估值输入：读 extract_facts_node 的产物 + 组装配置参数。

    提取不到任何一项【必要】输入就快速失败、不进后续节点 —— 失败原因是「数据不够」
    而不是「算错了」，这两件事在排查时指向完全不同的地方。

    总股本不在必要清单里：它缺了只丢每股口径，股权价值照常给出。

    :param state: ValuationState，读 tenant_id/task_id/financial_facts/degrade_reason。
    :return: dict，含 inputs_ok、valuation_inputs、assumption_sources；必要输入缺失时
        这些键为缺省/空、valuation_available=False。
    """
    tenant_id, task_id = state["tenant_id"], state["task_id"]
    settings = get_settings()

    facts: FinancialFacts | None = state.get("financial_facts")
    missing = MISSING_FIELDS(facts)

    if facts is None or missing:
        reason = state.get("degrade_reason") or ("必要输入缺失：" + ", ".join(missing)
                                                 if missing else "未找到任何年报")
        logger.warning("valuation.inputs_missing", task_id=task_id, missing=missing)
        return {"current_stage": "valuation", "inputs_ok": False,
                "valuation": None, "valuation_available": False,
                "valuation_inputs": None, "assumption_sources": [],
                "dcf_result": None, "comparable_result": None,
                "degrade_reason": reason}

    inputs = {
        "base_revenue_yi": facts.revenue_yi.value,
        "growth_rate": facts.revenue_yoy_pct.value / 100.0,
        "net_profit_yi": facts.net_profit_yi.value,
        "base_cashflow_yi": facts.operating_cashflow_yi.value,
        "total_shares_wan": (facts.total_shares_wan.value
                             if facts.total_shares_wan is not None else None),
        "base_report_title": facts.base_report_title,
    }
    # 假设清单：每一条要么能溯源到数据、要么被标为配置参数，不存在第三种。
    # 这条硬性质由 tests/test_valuation_agent.py 逐条断言。
    sources = [
        {"name": "营业收入", "value": facts.revenue_yi.value,
         "source": "derived:research_data_items", "excerpt": facts.revenue_yi.excerpt},
        {"name": "营收同比增速", "value": facts.revenue_yoy_pct.value,
         "source": "derived:research_data_items",
         "excerpt": facts.revenue_yoy_pct.excerpt},
        {"name": "归母净利润", "value": facts.net_profit_yi.value,
         "source": "derived:research_data_items", "excerpt": facts.net_profit_yi.excerpt},
        {"name": "经营现金流净额", "value": facts.operating_cashflow_yi.value,
         "source": "derived:research_data_items",
         "excerpt": facts.operating_cashflow_yi.excerpt},
        {"name": "折现率（WACC）", "value": settings.valuation_discount_rate,
         "source": "config:valuation_discount_rate"},
        {"name": "永续增长率", "value": settings.valuation_terminal_growth,
         "source": "config:valuation_terminal_growth"},
        {"name": "预测年限", "value": settings.valuation_forecast_years,
         "source": "config:valuation_forecast_years"},
        {"name": "可比 PE 下限", "value": settings.valuation_comparable_pe_low,
         "source": "config:valuation_comparable_pe_low"},
        {"name": "可比 PE 上限", "value": settings.valuation_comparable_pe_high,
         "source": "config:valuation_comparable_pe_high"},
    ]
    if facts.total_shares_wan is not None:
        sources.append({"name": "总股本", "value": facts.total_shares_wan.value,
                        "source": "derived:research_data_items",
                        "excerpt": facts.total_shares_wan.excerpt})

    return {"current_stage": "valuation", "inputs_ok": True,
            "valuation_inputs": inputs, "assumption_sources": sources}


async def run_dcf_node(state: ValuationState) -> dict:
    """DCF 路径。inputs_ok 为 False 时不动任何状态（返回空 dict）。

    :param state: ValuationState，读 inputs_ok/valuation_inputs/task_id。
    :return: {"dcf_result": {"low", "high", "error": None}}；inputs_ok 为 False 返回空
        dict，纯函数拒绝输入组合时返回 {"dcf_result": None}。
    """
    if not state.get("inputs_ok"):
        return {}
    settings = get_settings()
    inputs = state["valuation_inputs"] or {}
    try:
        result = run_dcf(
            base_cashflow_yi=inputs["base_cashflow_yi"],
            growth_rate=inputs["growth_rate"],
            years=settings.valuation_forecast_years,
            discount_rate=settings.valuation_discount_rate,
            terminal_growth=settings.valuation_terminal_growth,
        )
    except ValueError as e:
        # 纯函数拒绝了这个输入组合（负现金流等）—— 记下来，让 synthesize 统一决定降级。
        logger.warning("valuation.dcf_rejected", task_id=state["task_id"], error=str(e))
        return {"dcf_result": None}
    return {"dcf_result": {"low": result.low, "high": result.high, "error": None}}


async def run_comparable_node(state: ValuationState) -> dict:
    """可比法路径。inputs_ok 为 False 时不动任何状态（返回空 dict）。

    :param state: ValuationState，读 inputs_ok/valuation_inputs/task_id。
    :return: {"comparable_result": {"low", "high", "error": None}}；inputs_ok 为 False
        返回空 dict，纯函数拒绝输入组合时返回 {"comparable_result": None}。
    """
    if not state.get("inputs_ok"):
        return {}
    settings = get_settings()
    inputs = state["valuation_inputs"] or {}
    try:
        result = run_comparable(
            net_profit_yi=inputs["net_profit_yi"],
            pe_low=settings.valuation_comparable_pe_low,
            pe_high=settings.valuation_comparable_pe_high,
        )
    except ValueError as e:
        logger.warning("valuation.comparable_rejected", task_id=state["task_id"], error=str(e))
        return {"comparable_result": None}
    return {"comparable_result": {"low": result.low, "high": result.high, "error": None}}


async def _degrade(tenant_id: str, task_id: str, reason: str) -> dict:
    """统一的降级出口：落一行全 NULL 的不可用记录 + 一条 failed 阶段事件。

    为什么降级也要落一行：valuation_results 是这个域里唯一记录「估值发生过什么」的表。
    一张在估值失败时零行的表，与「估值从未跑过」事后完全无法区分。

    :param tenant_id: 租户，数据隔离键。
    :param task_id: 任务唯一标识。
    :param reason: 降级原因，写入 rationale 与阶段事件。
    :return: dict，含 current_stage="valuation"、valuation=None、valuation_available=False。
    """
    await repo.upsert_valuation_result(
        tenant_id=tenant_id, task_id=task_id, method="unavailable",
        value_low=None, value_high=None, per_share_low=None, per_share_high=None,
        currency="CNY", assumptions=[], rationale=reason, is_available=False,
    )
    await repo.record_stage_event(tenant_id, task_id, "valuation", "failed",
                                 {"reason": reason})
    logger.warning("valuation.unavailable", task_id=task_id, reason=reason)
    return {"current_stage": "valuation", "valuation": None, "valuation_available": False}


async def synthesize_node(state: ValuationState) -> dict:
    """合成：LLM 只选路径 + 写理由，数字由代码取。

    这是本子图【唯一】的 LLM 调用点，也是 with_retry("valuation") 的挂载处。

    :param state: ValuationState，读 tenant_id/task_id/inputs_ok/degrade_reason/
        dcf_result/comparable_result/valuation_inputs/company_code/industry/
        assumption_sources。
    :return: dict，含 current_stage="valuation"；可用时带 valuation（method、
        equity_value_low/high、per_share_low/high 等键）与 valuation_available=True，
        否则走降级返回 valuation=None、valuation_available=False。
    """
    tenant_id, task_id = state["tenant_id"], state["task_id"]

    if not state.get("inputs_ok"):
        return await _degrade(tenant_id, task_id,
                              state.get("degrade_reason") or "必要输入缺失")

    dcf, comp = state.get("dcf_result"), state.get("comparable_result")
    if dcf is None or comp is None:
        failed = [name for name, value in (("dcf", dcf), ("comparable", comp))
                  if value is None]
        return await _degrade(tenant_id, task_id,
                              f"估值路径失败：{', '.join(failed)}")

    inputs = state["valuation_inputs"] or {}
    facts_lines = [f"基期报告：{inputs.get('base_report_title')}",
                   f"基期营收：{inputs['base_revenue_yi']:.2f} 亿元",
                   f"营收增速：{inputs['growth_rate'] * 100:.2f}%",
                   f"归母净利润：{inputs['net_profit_yi']:.2f} 亿元"]
    prompt = build_valuation_prompt(state.get("company_code", ""),
                                    state.get("industry") or "", facts_lines, dcf, comp)

    structured = get_structured_llm("valuation", ValuationDecision)

    @with_retry(agent_type="valuation")
    async def _invoke():
        """带闭包 prompt 调选路径 LLM；无参数。返回 ValuationDecision。"""
        return await structured.ainvoke(prompt)

    # 与 analyze/nodes.py:114 同一形态：闭包只包 LLM 调用（无副作用），
    # 节点自己判别哨兵。直接装饰节点函数是不行的 —— with_retry 的返回值顶替
    # 被装饰函数的返回值，而 FallbackResult 不是 dict，会被 LangGraph 拒绝合并。
    try:
        parsed = await _invoke()
    except Exception as e:                       # noqa: BLE001 —— 降级也得留一条记录
        logger.error("valuation.synthesize_failed", task_id=task_id, error=str(e))
        parsed = None

    if not isinstance(parsed, ValuationDecision):
        note = (parsed.note if isinstance(parsed, FallbackResult)
                else "估值服务暂时不可用")
        return await _degrade(tenant_id, task_id, f"估值合成失败：{note}")

    method = parsed.method.strip().lower()
    if method not in VALID_METHODS:
        # 模型给了一个非法方法名 —— 不猜它想说什么，直接降级。
        return await _degrade(tenant_id, task_id, f"估值方法不可识别：{parsed.method!r}")

    try:
        chosen = select_range(method,
                             Range(low=dcf["low"], high=dcf["high"]),
                             Range(low=comp["low"], high=comp["high"]))
    except ValueError as e:
        return await _degrade(tenant_id, task_id, f"估值区间合成失败：{e}")

    total_shares = inputs.get("total_shares_wan")
    low_ps = high_ps = None
    if total_shares:
        low_ps = per_share(chosen.low, total_shares)
        high_ps = per_share(chosen.high, total_shares)

    assumptions = state.get("assumption_sources") or []
    await repo.upsert_valuation_result(
        tenant_id=tenant_id, task_id=task_id, method=method,
        value_low=chosen.low, value_high=chosen.high,
        per_share_low=low_ps, per_share_high=high_ps, currency="CNY",
        assumptions=assumptions, rationale=parsed.rationale, is_available=True,
    )
    await repo.record_stage_event(tenant_id, task_id, "valuation", "success",
                                 {"method": method, "low": round(chosen.low, 2),
                                  "high": round(chosen.high, 2),
                                  "per_share_available": low_ps is not None})
    logger.info("valuation.available", task_id=task_id, method=method)

    return {"current_stage": "valuation", "valuation_available": True,
            "valuation": {
                "method": method,
                "equity_value_low": chosen.low, "equity_value_high": chosen.high,
                "per_share_low": low_ps, "per_share_high": high_ps,
                "currency": "CNY", "assumptions": assumptions,
                "rationale": parsed.rationale, "is_available": True,
            }}