# backend/mcp/data_source_server.py
# 采集 MCP（FastMCP streamable HTTP）：把四源取数收进一个跨进程工具。
#
# 与联网搜索、研报检索两条 MCP 同构：对外暴露一个 fetch_all_sources 工具，
# 内部复用 backend.agents.collect.adapters.fetch_all_sources —— 四源（财报 /
# 公告 / 新闻 / 行业）并发 gather + 失败归并 + fixture 兜底 的逻辑不重写，
# server 只是把它们包成工具层，客户端一次调用拿全量。
#
# 组合姿态：fetch_* 仍是"平铺普通异步函数"（见 05 讲义 5.2/5.3 的立场），
# 这个 MCP server 是包在它们外面的一层壳，不是替换那个抽象决策。

import asyncio
from datetime import datetime, timezone

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    name="ResearchAgent-DataSource",
    stateless_http=True,
    json_response=True,
)


def _raw_to_dict(item) -> dict:
    """把 RawDataItem 序列化成 JSON 可传输的 dict（published_at → iso str）。

    :param item: RawDataItem 实例（source_type/source_name/title/content/url/published_at/raw）。
    :return: dict，published_at 转 iso 字符串；raw 原样（通常是 dict/None）。
    """
    return {
        "source_type": item.source_type,
        "source_name": item.source_name,
        "title": item.title,
        "content": item.content,
        "url": item.url,
        "published_at": item.published_at.isoformat() if item.published_at else None,
        "raw": item.raw,
    }


@mcp.tool()
async def fetch_all_sources(
    company_code: str,
    company_name: str = "",
    industry: str = "",
) -> dict:
    """
    四源并发取数：财报 / 公告 / 新闻 / 行业（各自带 fixture 兜底与失败归并）。

    对应采集子图的 fetch_all_sources_node：节点只做编排，取数细节收在
    adapters.fetch_all_sources，本工具把它包成跨进程 MCP 能力。客户端一次调用
    拿全量，无需在客户端另起并发。

    Args:
        company_code: 股票代码（如 600519.SH）
        company_name: 公司名（新闻/财报真实源拼查询词用）
        industry:     行业（透传给各 fetch_*）

    Returns:
        dict，含 raw_items（RawDataItem 的 JSON 形态列表）、source_stats
        （{source_type: {"ok", "count"}}）、errors（失败源明细列表）。
        raw_items 为空（四源均无数据）不抛错，由调用方决定怎么处置。
    """
    from backend.agents.collect import adapters
    from backend.core.logger import get_logger

    logger = get_logger(__name__)

    company = {"code": company_code, "name": company_name or company_code,
               "industry": industry}
    try:
        raw_items, source_stats, errors = await adapters.fetch_all_sources(company)
    except Exception as e:                           # noqa: BLE001 —— 取数失败按空返回，调用方归因
        logger.error("data_source_mcp.fetch_failed", error=str(e))
        return {"raw_items": [], "source_stats": {}, "errors": [
            {"source_type": "all", "error": str(e)}]}

    logger.info("data_source_mcp.fetch_done",
                company=company_code,
                sources={k: v["count"] for k, v in source_stats.items()})
    return {
        "raw_items": [_raw_to_dict(it) for it in raw_items],
        "source_stats": source_stats,
        "errors": errors,
    }


if __name__ == '__main__':
    import uvicorn

    port = 8003
    print(f"DataSource MCP Server → http://localhost:{port}/mcp")
    uvicorn.run(mcp.streamable_http_app(), host="0.0.0.0", port=port)