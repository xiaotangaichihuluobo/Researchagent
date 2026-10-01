import asyncio, sys
sys.path.insert(0, ".")

from backend.mcp.client import call_mcp_tool, list_mcp_tools

async def main():
    base = "http://localhost:8009"
    print("已注册工具：", [t["name"] for t in await list_mcp_tools(base)])
    result = await call_mcp_tool(base, "add", {"a": 3, "b": 5})
    print("add(3, 5) =", result)      # 预期：8

asyncio.run(main())