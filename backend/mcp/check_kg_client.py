# backend/mcp/check_kg_client.py
# 知识库 MCP 演示客户端：走 streamable HTTP 调新 tool search_knowledge_base，
# 检索研报语料 report_corpus。用法（先起后端后）：
#   python -m backend.mcp.check_kg_client
import asyncio
import json

import httpx

MCP_URL = "http://localhost:8000/mcp/kb"


async def _call_tool(session_id: str, name: str, args: dict) -> dict:
    """向 MCP streamable HTTP 端点发一次 tool call，返回 JSON。"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            MCP_URL,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": name, "arguments": args, "_meta": {"sessionId": session_id}},
            },
        )
        resp.raise_for_status()
        return resp.json()


async def main():
    # 先初始化会话拿到 sessionId
    async with httpx.AsyncClient(timeout=15.0) as client:
        init = await client.post(
            MCP_URL,
            json={
                "jsonrpc": "2.0",
                "id": 0,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "check-kg-client", "version": "0.1.0"},
                },
            },
        )
        init.raise_for_status()
        body = init.json()
        session_id = body["result"]["_meta"]["sessionId"]

    result = await _call_tool(
        session_id,
        "search_knowledge_base",
        {"query": "白酒行业的盈利模式", "tenant_id": "tenant_default", "top_k": 3},
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())