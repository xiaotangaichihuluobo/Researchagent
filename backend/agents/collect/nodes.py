# backend/agents/collect/nodes.py
# 采集子图的四个节点：查标的 → 四源并发取数 → 统一格式化 → 落库。
#
# 节点只做编排：四源并发取数的细节（装配适配器 + asyncio.gather + 逐源判成败）
# 收在 adapters.fetch_all_sources，节点这边一行调用。不拆成四个图节点，是因为
# 四个源各取各的、合并是普通列表拼接，用不着图级 fan-out 和 reducer——
# 那套留给分析阶段真正需要并发写同一份 State 的四维并发。

from datetime import datetime, timezone

from backend.agents.collect.adapters import (
    CollectExhaustedError, RawDataItem, normalize_and_score,
)
from backend.agents.collect.state import CollectState
from backend.config import get_settings
from backend.core import research_repo as repo
from backend.core.logger import get_logger
from backend.mcp.client import call_mcp_tool

logger = get_logger(__name__)


def _raw_from_dict(d: dict) -> RawDataItem:
    """把 MCP 返回的 RawDataItem JSON 形态重建回 dataclass（published_at 反解析）。"""
    published = d.get("published_at")
    return RawDataItem(
        source_type=d.get("source_type", ""),
        source_name=d.get("source_name", ""),
        title=d.get("title", ""),
        content=d.get("content", ""),
        url=d.get("url"),
        published_at=(datetime.fromisoformat(published) if published else None),
        raw=d.get("raw") or {},
    )


async def resolve_company_node(state: CollectState) -> dict:
    """确认标的在库中存在，并把它的元信息补进 State。

    标的查不到是【配置错误】而不是运行时故障 —— 与其让它一路跑到分析阶段
    才发现没有公司名可用，不如在第一步就失败，错误信息也直指原因。
    """
    company = await repo.get_company_by_code(state["tenant_id"], state["company_code"])
    if not company:
        raise CollectExhaustedError(f"标的不存在：{state['company_code']}")

    # 先写业务表、再返回 State（设计文档 §4.4 的顺序规则）。
    # current_stage 必须真落库：前端进度条读的是这一列，不是图内存里的 State。
    await repo.update_task_status(state["tenant_id"], state["task_id"], current_stage="collect")
    await repo.record_stage_event(state["tenant_id"], state["task_id"],
                                  "collect", "started", {"company": state["company_code"]})
    return {
        "current_stage": "collect",
        "company_id": str(company["id"]),
        "industry": company.get("industry") or state.get("industry") or "",
        "errors": [],
    }


async def fetch_all_sources_node(state: CollectState) -> dict:
    """四个源并发取数（只做编排；取数细节收在 data_source MCP 的 fetch_all_sources）。"""
    company = {
        "code": state["company_code"],
        "name": state.get("company_name") or state["company_code"],
        "industry": state.get("industry") or "",
    }

    # 真走 MCP 协议调采集 MCP（/mcp/data-source 的 fetch_all_sources）。
    # 与 QA 检索、联网搜索一致：不在此静默切回 adapters 直连函数 —— MCP 失败即抛，
    # 由统错误处理落回「四源全败」归因，而不是绕过 MCP 假装取数成功。
    # timeout=180：四源并发里最慢的是 baostock 财报（登录 + 拉 6 份年报，在无 GPU 的
    # 云服务器上实测要 ~57s）。call_mcp_tool 默认 30s 会在此超时 —— 表现为 runner 在
    # 任务开始后恰 +30s 报 pipeline_failed 且 error=''（超时的 str() 为空），而 baostock
    # 稍后仍采完（fetch_done 晚到 27s）。放宽到 180s，给慢源留足余量，别把正常慢当失败。
    result = await call_mcp_tool(
        get_settings().data_source_mcp_server_url,
        "fetch_all_sources",
        {"company_code": company["code"], "company_name": company["name"],
         "industry": company["industry"]},
        timeout=180.0,
    ) or {}

    # call_mcp_tool 对【返回单个 dict】的工具会把结果包成 [dict]：它的归一假设是工具
    # 返回 list（search_knowledge_base / web_search 都返回 list），而 fetch_all_sources
    # 是唯一一个返回单 dict 的工具，于是 result 拿到 [{"raw_items": ...}] 而不是 dict。
    # 不解包的话，下面的 result.get("raw_items") 会抛 'list' object has no attribute
    # 'get' —— 正是不做这段兼容时采集阶段崩溃（collect 停在『进行中』、analyze 不启动）
    # 的根因。这里兼容两种形态：dict 直接用，[dict] 取下标，[] 兜成 {}。
    if isinstance(result, list):
        result = result[0] if result else {}

    raw_items = [_raw_from_dict(d) for d in (result.get("raw_items") or [])]
    source_stats = result.get("source_stats") or {}
    errors = result.get("errors") or []

    for e in errors:                                   # 失败源只记日志，不中断其它源
        logger.warning("collect.source_failed", source_type=e["source_type"], error=e["error"])

    logger.info("collect.fetch_done", task_id=state["task_id"],
                sources={k: v["count"] for k, v in source_stats.items()})
    return {"raw_items": raw_items, "source_stats": source_stats, "errors": errors}


async def normalize_and_score_node(state: CollectState) -> dict:
    """格式化：算时效权重与可信度、截断正文、按源限量。"""
    rows = normalize_and_score(state.get("raw_items") or [],
                               now=datetime.now(timezone.utc))
    logger.info("collect.normalized", task_id=state["task_id"], count=len(rows))
    return {"raw_items": rows}


async def persist_collected_node(state: CollectState) -> dict:
    """落库。

    四源一条数据都没取到时【主动抛异常】而不是让流程继续：
    继续下去会走到分析阶段，四个维度全部「数据不足」，最终产出一份
    「因数据不足无法评级」的研报 —— 这看起来像分析失败，实际是采集失败，
    错误归因错了，排查会绕远路。宁可在源头失败，错误信息也准确。
    """
    rows = state.get("raw_items") or []

    if not rows:
        source_stats = state.get("source_stats", {})
        errors = state.get("errors") or []
        # 走到这里有两种情形，处置相反，所以必须分开记：源全报错 → 去查数据源；
        # 源全返回空 → 是这个标的问题。此前一律写「四源均无数据」，等于把源故障
        # 误报成标的问题 —— 正是上面那段注释说要避免的错误归因。
        # 用机器可读的 cause 分，不靠人去解析文案。
        cause = "sources_failed" if errors else "no_data"
        reason = "四个数据源均取数失败" if errors else "四个数据源均无数据"
        await repo.record_stage_event(
            state["tenant_id"], state["task_id"], "collect", "failed",
            {"cause": cause, "reason": reason,
             "by_source": source_stats, "errors": errors},
        )
        raise CollectExhaustedError(
            f"{reason}（{state['company_code']}），"
            f"请确认该标的的数据源配置：{source_stats}"
        )

    collected_ids = await repo.insert_data_items(
        state["tenant_id"], state["task_id"], state["company_id"], rows,
    )

    await repo.record_stage_event(
        state["tenant_id"], state["task_id"], "collect", "success",
        # by_source 连 ok 一起落，不能只落 count：count==0 既可能是「这个源没数据」
        # （正常业务结果），也可能是「这个源挂了」（要排查），两者处置相反。只留条数
        # 的话，图内部还分得清、出了图就分不清了 —— 而事后审计恰恰在图外面。
        # errors 同理：失败原因此前只活在 State 里，下一阶段一覆盖就没了。
        {"items": len(collected_ids),
         "by_source": state.get("source_stats", {}),
         "errors": state.get("errors") or []},
    )
    return {"collected_ids": collected_ids, "raw_items": []}   # 落库后清空，State 只留引用
