# backend/agents/research/protocols.py
# 阶段间通信协议层（handoff contract）。
#
# 五子图的通信原本是「靠约定」而非「靠契约」：父 State 声明了哪些键、各子图 State 里
# 「继承/产出」哪些键，全靠注释和人记得写对 —— 漏一个就是运行期静默拿到 None。
# 这一层把每个阶段的【入/出契约】声明成可校验的数据，在父图的 handoff 边界强制校验，
# 并把「契约自身一致」也变成一条可自省的测试（见 tests/test_research_protocol.py）。
#
# 借鉴点：agent harness 的协议层（s09/s10）。注意我们【不需要】邮箱/认领那套并发通信 ——
# 投研流水线是固定顺序、同步驱动的，真正值得借的是「阶段之间有一个机器能校验的握手契约」。

from dataclasses import dataclass, field
from typing import Callable, Optional

from backend.core.exceptions import NonDegradableError

# 一条 handoff 的约定：进入阶段前必须满足的前置，与离开阶段后必须写回的产出。
# requires 表示「上游必须已产出、且此刻非 None」的父 State 键；
# produces 表示「本阶段必须写回父 State」的键。
# require_check 是可选的额外谓词：返回 None 表示满足，返回字符串表示失败原因（如
# 「dimension_results 非空」这种无法只靠键存在表达的约束必须用谓词，不能用语义造默认值）。
#
# risk 的 produces 只列闸门【之前】就产出的键（compliance_checklist）：risk_decision /
# redo_targets 是 human_gate 之后的产物，首跑会停在闸门上、那些键还不存在，纳入
# verify 会在暂停分支上误报「未产出」。它们由 test_protocol 的整图 happy-path 测试钉住。


@dataclass(frozen=True)
class StageContract:
    stage: str
    requires: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()
    require_check: Optional[Callable[[dict], Optional[str]]] = None


def _dimensions_nonempty(state: dict) -> Optional[str]:
    """analyze → retrieve 的实质约定：检索要拿四维结论拼查询，
    一个空的 dimension_results 等于「上游空分析」，不是「检索没结果」。

    :param state: 当前父 State，检查其 dimension_results 键。
    :return: None 表示满足（dimension_results 非空）；否则返回失败原因字符串。
    """
    if not state.get("dimension_results"):
        return "dimension_results 为空或缺失（上游未产出四维结论）"
    return None


# 父图初始态就携带的键（runner.py 的 initial_state + 任务行带上的）。
_INITIAL_STATE_KEYS = ("task_id", "tenant_id", "company_id", "company_code",
                       "industry", "redo_count")

# 流水线固定顺序。reachability 自省依赖这个顺序。
_PIPELINE_ORDER = ("collect", "analyze", "retrieve", "valuation", "risk")

CONTRACTS: dict[str, StageContract] = {
    "collect": StageContract(
        stage="collect",
        requires=("task_id", "tenant_id", "company_id", "company_code"),
        produces=("collected_ids", "source_stats"),
    ),
    "analyze": StageContract(
        stage="analyze",
        requires=("company_id", "collected_ids"),
        produces=("dimension_results", "rating", "has_buy_sell_advice"),
    ),
    "retrieve": StageContract(
        stage="retrieve",
        requires=("dimension_results",),
        require_check=_dimensions_nonempty,
        produces=("has_reference", "comparison_points"),
    ),
    "valuation": StageContract(
        stage="valuation",
        requires=("rating", "has_reference"),
        produces=("valuation_available",),
    ),
    "risk": StageContract(
        stage="risk",
        requires=("rating", "has_reference", "valuation_available"),
        produces=("compliance_checklist",),   # 闸门产物，见文件头注释
    ),
}


class HandoffContractError(NonDegradableError, RuntimeError):
    """handoff 校验失败：下游前置缺失或本阶段未产出。

    双继承同 CollectExhaustedError 的手法：给 with_retry 看到 NonDegradableError
    （不可重试、不可降级、立即阻断），给现有按 RuntimeError 捕获的调用方保留兼容。
    走到这里说明是状态/接线层面的结构性错误 —— 阻断比给出一个看似成功的结果更干净。
    """


def _contract(stage: str) -> StageContract:
    """按阶段名取已注册的握手契约。

    :param stage: 阶段名（collect/analyze/retrieve/valuation/risk）。
    :return: 该阶段的 StageContract；未知阶段抛 HandoffContractError。
    """
    try:
        return CONTRACTS[stage]
    except KeyError:
        raise HandoffContractError(f"未知阶段：{stage}（CONTRACTS 未声明该阶段的契约）")


def validate_requires(stage: str, state: dict) -> None:
    """进阶段前：上游必须已产出本阶段 requires 声明的键（且非 None）。

    必须在父图 handoff 边界调用：它把「下游静默拿到 None」变成「边上有响亮的错误」。

    :param stage: 目标阶段名，按其在 CONTRACTS 里的契约校验。
    :param state: 当前父 State；缺失或为 None 的 requires 键都会触发失败。
    :return: 无返回值；校验失败抛 HandoffContractError。
    """
    c = _contract(stage)
    missing = [k for k in c.requires
               if k not in state or state.get(k) is None]
    if missing:
        raise HandoffContractError(
            f"[{stage}] 前置缺失（上游未产出）：{[f'{k}(None)' for k in missing]}")
    if c.require_check is not None:
        reason = c.require_check(state)
        if reason:
            raise HandoffContractError(f"[{stage}] 前置不满足：{reason}")


def verify_produces(stage: str, state: dict) -> None:
    """出阶段后：本阶段 produces 声明的键必须已写回父 State（且非 None）。

    在父图 handoff 边界、子图节点之后调用，防止「该产出的键被静默丢弃 / 子图漏写」。

    :param stage: 目标阶段名，按其在 CONTRACTS 里的契约校验。
    :param state: 出段后的父 State；produces 声明的键缺失或为 None 都会触发失败。
    :return: 无返回值；校验失败抛 HandoffContractError。
    """
    c = _contract(stage)
    missing = [k for k in c.produces
               if k not in state or state.get(k) is None]
    if missing:
        raise HandoffContractError(
            f"[{stage}] 未产出下游所需键：{[f'{k}(None)' for k in missing]}")


def contract_integrity_issues() -> list[str]:
    """自省契约清单本身，返回所有不一致项（空列表 = 契约自洽）。

    复刻「注册表自省」手法（retry.describe_registry）：契约里的每个键必须真的存在于
    父 State 声明里，且每个 requires 必须能被初始态或上游 produces 可达 ——
    否则就是一条「写了但永远满足不了」的死契约。

    :return: 不一致项列表；每项是一条描述字符串，空列表表示契约清单自洽。
    """
    from backend.agents.research.state import ResearchState

    declared = set(ResearchState.__annotations__)
    issues: list[str] = []

    for stage, c in CONTRACTS.items():
        for k in (*c.requires, *c.produces):
            if k not in declared:
                issues.append(f"[{stage}] 键 {k} 未在父 State（ResearchState）声明")

    produced = set(_INITIAL_STATE_KEYS)
    for stage in _PIPELINE_ORDER:
        c = CONTRACTS[stage]
        miss = set(c.requires) - produced
        if miss:
            issues.append(
                f"[{stage}] requires 不可达：{sorted(miss)} 不在初始态、也不被上游 produces")
        produced |= set(c.produces)

    return issues