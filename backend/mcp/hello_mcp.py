# backend/mcp/hello_mcp.py —— 最小 MCP Server（演示，不依赖任何模型）

from mcp.server.fastmcp import FastMCP

# 创建一个 MCP Server 实例（三个参数含义见上面的 FastMCP 小节）
mcp = FastMCP(name="HelloServer", stateless_http=True, json_response=True)


@mcp.tool()                          # 就这一个装饰器，把下面的函数注册成了一个 MCP 工具
async def add(a: int, b: int) -> int:
    """把两个整数相加（演示 MCP 工具的最小形态）"""   # docstring 会成为工具描述
    return a + b                     # 返回值由 FastMCP 自动包成 JSON 回给调用方


if __name__ == "__main__":
    import uvicorn
    # streamable_http_app() 把 mcp 变成标准 ASGI 应用，uvicorn 直接跑；端点为 /mcp
    uvicorn.run(mcp.streamable_http_app(), host="0.0.0.0", port=8009)