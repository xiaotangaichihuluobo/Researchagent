# backend/mcp/client.py

import json
from typing import Any

import httpx

from backend.core.logger import get_logger

logger = get_logger(__name__)


async def call_mcp_tool(
    server_url: str,
    tool_name: str,
    arguments: dict[str, Any],
    timeout: float = 120.0,
) -> Any:
    """
    调用 stateless MCP Server 的单个工具。

    stateless_http=True 的 Server 每次 POST 完全自包含，无需先发 initialize 握手。

    Args:
        server_url: MCP Server 基础 URL，例如 "http://localhost:8000/mcp/kb"
        tool_name:  工具名称，与 @mcp.tool() 注册名一致
        arguments:  工具参数字典
        timeout:    请求超时（秒）

    Returns:
        工具返回值（JSON 反序列化后的 Python 对象）

    Raises:
        httpx.HTTPStatusError:  Server 返回 4xx/5xx
        ValueError:             JSON-RPC 错误（工具内部异常）
        httpx.TimeoutException: 请求超时
    """
    payload = {
        "jsonrpc": "2.0",
        "id":      1,
        "method":  "tools/call",
        "params": {
            "name":      tool_name,
            "arguments": arguments,
        },
    }

    # json_response=True 的 Server 要求客户端声明 Accept: application/json，
    # 否则 Server 返回 -32600 "Not Acceptable" 错误
    headers = {
        "Content-Type": "application/json",
        "Accept":        "application/json",
    }

    # trust_env=False：禁止 httpx 读取 HTTP_PROXY / ALL_PROXY 等环境变量和系统代理。
    # macOS 系统代理或 PyCharm/Charles 等工具设置的代理不会排除 localhost，
    # 导致本地 MCP 调用被路由到代理后返回 502。对 localhost 的内部调用永远不需要代理。
    logger.info("mcp.call_started", server_url=server_url, tool=tool_name,
                args_keys=",".join(arguments.keys()))
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        resp = await client.post(f"{server_url}/mcp", json=payload, headers=headers)
        resp.raise_for_status()

    data = resp.json()
    # print(f"data:{data}")
    # JSON-RPC 错误信封
    if "error" in data:
        logger.warning("mcp.call_failed", server_url=server_url, tool=tool_name,
                       error=str(data["error"])[:300])
        raise ValueError(
            f"MCP tool '{tool_name}' error: {data['error'].get('message', str(data['error']))}"
        )

    content = data.get("result", {}).get("content", [])
    logger.info("mcp.call_succeeded", server_url=server_url, tool=tool_name,
                content=len(content))

    # FastMCP 对 list[dict] 的序列化行为：
    #   - 每个 dict 单独放进一个 TextContent 条目（最常见）
    #   - 或整个列表序列化成一个 TextContent 条目（部分版本）
    # 必须遍历所有 content 条目，不能只取 content[0]，否则多条结果只返回第一条。
    # print("*"*80)
    # print(f'content:{len(content)}')
    # print(f'content:{content[0]}')
    if not content:
        return []

    items = []
    for item in content:
        if not isinstance(item, dict):
            continue
        text = item.get("text", "")
        if not text:
            continue
        try:
            parsed = json.loads(text)
            # print(f'parsed:{parsed}')
            # print(f'parsed:{type(parsed)}')
            # break
        except (json.JSONDecodeError, TypeError):
            parsed = text
        if isinstance(parsed, list):
            # 整个列表在一个 TextContent 里，直接返回
            return parsed
        items.append(parsed)

    return items if items else content


async def list_mcp_tools(server_url: str, timeout: float = 10.0) -> list[dict]:
    """
    列出 MCP Server 提供的所有工具（调试 / 验证用）。

    Returns:
        工具列表，每项含 name / description / inputSchema
    """
    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        resp = await client.post(f"{server_url}/mcp", json=payload, headers=headers)
        resp.raise_for_status()
    print(f"resp.json()-->{resp.json()}")
    return resp.json().get("result", {}).get("tools", [])

