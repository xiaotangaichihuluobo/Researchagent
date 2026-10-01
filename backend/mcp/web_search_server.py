import asyncio
import os
import sys

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    name="ResearchAgent-WebSearch",
    stateless_http=True,
    json_response=True,
)


async def _search_tavily(query: str, max_results: int, api_key: str) -> list[dict]:
    """Tavily 搜索（结构化结果，需 API key，https://tavily.com）"""
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            "https://api.tavily.com/search",
            json={
                "api_key":             api_key,
                "query":               query,
                "max_results":         max_results,
                "include_answer":      False,
                "include_raw_content": False,
            },
        )
        resp.raise_for_status()
    # print(f'resp.json()-->{resp.json()}')
    # print(f'results-->{len(resp.json().get("results", []))}')
    # print(f'results[0]-->{resp.json().get("results", [])[0]}')
    return [
        {
            "title":   r.get("title", ""),
            "url":     r.get("url", ""),
            "snippet": r.get("content", "")[:500],
            "content": r.get("content", ""),
        }
        for r in resp.json().get("results", [])
    ]
async def _search_duckduckgo(query: str, max_results: int) -> list[dict]:
    """DuckDuckGo 搜索（免费，无需 API key）

    duckduckgo-search 6.x+ 移除了 AsyncDDGS，统一使用同步 DDGS。
    用 asyncio.to_thread 包装避免阻塞事件循环（与 run_in_executor 效果相同，写法更简洁）。
    """
    from duckduckgo_search import DDGS

    def _sync_search() -> list[dict]:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title":   r.get("title", ""),
                    "url":     r.get("href", ""),
                    "snippet": r.get("body", "")[:500],
                    "content": r.get("body", ""),
                })
        return results

    return await asyncio.to_thread(_sync_search)
# ── MCP 工具 ──────────────────────────────────────────────────────

@mcp.tool()
async def web_search(
    query: str,
    max_results: int = 5,
) -> list[dict]:
    """
    Search the web for current information not available in the knowledge base.

    Automatically selects the best available backend:
    - Tavily (preferred): structured results; requires TAVILY_API_KEY in .env.local
    - DuckDuckGo (fallback): free, no API key required

    Args:
        query:       Search query string
        max_results: Maximum number of results to return (default: 5)

    Returns:
        List of search results, each with title / url / snippet / content
    """
    from backend.config import get_settings
    from backend.core.logger import get_logger

    logger = get_logger(__name__)
    settings = get_settings()

    # 优先 Tavily，失败后降级 DuckDuckGo
    if settings.tavily_api_key:
        try:
            results = await _search_tavily(query, max_results, settings.tavily_api_key)
            logger.info("web_search_mcp.tavily_done", hits=len(results))
            return results
        except Exception as e:
            logger.warning("web_search_mcp.tavily_failed", error=str(e))

    try:
        results = await _search_duckduckgo(query, max_results)
        logger.info("web_search_mcp.ddgs_done", hits=len(results))
        return results
    except Exception as e:
        logger.error("web_search_mcp.ddgs_failed", error=str(e))
        return []
if __name__ == '__main__':
    import uvicorn
    port = 8002
    print(f"Web Search MCP Server → http://localhost:{port}/mcp")
    uvicorn.run(mcp.streamable_http_app(), host="0.0.0.0", port=port)