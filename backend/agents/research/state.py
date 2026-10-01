# backend/agents/research/state.py
# 投研域编排父图的状态模型。
#
# 一条硬规矩：业务数据的唯一可信来源是业务表，State 只放「引用」与「小数据」。
# 采集到的财报全文、新闻正文这类大对象存在 research_data_items 里，
# State 只留 ID 列表；用完即走的中间大对象（如 RAG 检索到的正文片段）
# 只声明在子图 State 里 —— 重构后由阶段机显式掌控并发与合并（见 steps.py），
# 不再依赖 LangGraph 的 reducer/合并语义，「大对象不进 State」的分工原样保留。

from typing import Optional

# 仅用 typing_extensions.TypedDict 声明字段形状 —— 纯类型标注，与 LangChain
# 无关。保留它而非普通 dict，是为了让 protocols.contract_integrity_issues 能通过
# ResearchState.__annotations__ 自省契约键是否声明（运行时反射，无框架依赖）。
from typing_extensions import TypedDict




# ── 阶段机控制常量 ─────────────────────────────────────────────
# PENDING_SIGN_OFF 是 risk 段返回的哨兵：引擎看到它就知道首跑停在人工闸门上
# （取代 LangGraph 的 interrupt）。其余是普通字符串，表驱动线性推进。
PENDING_SIGN_OFF = "|PENDING_SIGN_OFF|"

# 线性下一段映射。risk 之后没有线性下一段：由风控裁决 + routing 决定去向
# （publish / 回边到 analyze 或 valuation / mark_rejected / mark_failed）。
STAGE_ORDER = ("collect", "analyze", "retrieve", "valuation", "risk")
NEXT_STAGE = {
    "collect": "analyze",
    "analyze": "retrieve",
    "retrieve": "valuation",
    "valuation": "risk",
}


class ResearchState(TypedDict):
    """投研域父 State。

    这份字段清单是「跨阶段必需的字段」的并集：某个阶段写、后续阶段要读的，
    必须在这里声明；某阶段内部自用的中间变量，不要在这里出现。
    """

    # ── 请求上下文（每步都用，且都很小）──────────────────────────
    task_id: str                              # 任务唯一标识
    tenant_id: str                            # 租户，数据/事件隔离键
    company_id: str                           # 标的公司 ID
    company_code: str                         # 标的股票代码
    industry: str                             # 所属行业（检索/估值用到）
    current_stage: str                        # 条件路由依赖；由各阶段入口节点写入

    # ── ① 采集阶段产物（只放引用，正文在业务表）────────────────
    collected_ids: list[str]                  # research_data_items.id 列表，正文在业务表，State 只存引用
    source_stats: dict                        # 各源成功/失败统计
    errors: list[dict]                        # 部分失败记录（不跨阶段累加，见注释）

    # ── ② 分析阶段产物 ─────────────────────────────────────────
    dimension_results: dict                   # {维度: 结论}；检索段要拿它拼查询（空 dict ≠ 满足，见 protocols）
    rating: str                               # 综合评级（buy / sell / hold）
    rating_note: str                          # 评级依据简述
    has_buy_sell_advice: bool                 # 是否给出买卖建议；数据不足时必须为 False

    # ── ③ 研报 RAG 产物 ────────────────────────────────────────
    has_reference: bool                       # 本轮是否检索到可比标的，估值段依赖
    comparison_points: list[str]              # 可比对要点，最终写进研报

    # ── ④ 估值产物 ─────────────────────────────────────────────
    valuation: Optional[dict]                 # 估值结果；不可用时为 None，绝不给默认区间
    valuation_available: bool                 # 估值是否可用，风控结算依赖

    # ── ⑤ 风控产物 ─────────────────────────────────────────────
    risk_decision: Optional[str]              # 审核签字：approve / modify / reject
    risk_comments: Optional[str]              # 审核人签署意见
    redo_targets: Optional[dict]              # 驳回重做目标 {"stage": "analyze", "dimensions": [...]}
    compliance_checklist: dict                # 自动预检结果

    # ── 流程控制 ───────────────────────────────────────────────
    # redo_count 的权威值在 research_tasks.redo_count（DB），
    # 这里只是引擎从 DB 回填的一份副本，供路由判定用。
    # 落 DB 后由引擎回填，计数同时获得持久性（重启不丢）与可审计性（阶段事件可查）。
    redo_count: int                           # 驳回次数；权威值在 DB，这里是从 DB 回填的副本
    fallback_used: bool                       # 本任务是否触发过降级