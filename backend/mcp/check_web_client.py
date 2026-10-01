# scripts/test_mcp_search.py
# 前提：python backend/mcp/web_search_server.py 已在终端 1 运行
import asyncio, sys
from backend.mcp.client import call_mcp_tool

async def main():
    base = "http://localhost:8000/mcp/search"          # 独立模式：直接指向端口，无路径前缀

    results = await call_mcp_tool(
        server_url=base,
        tool_name="web_search",
        arguments={"query": "什么是牛奶", "max_results": 1},
    )
    print(f"搜索结果 {len(results)} 条")
    print(results)
    for i, r in enumerate(results, 1):
        print(f"[{i}] {r}")
        print(f"[{i}] {r['title']}")
        print(f"     {r['url']}")
        print(f"     {r['snippet'][:80]}...\n")

asyncio.run(main())