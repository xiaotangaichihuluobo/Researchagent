# backend/main.py

import os
import sys

# Windows conda 专用：只在真的缺 liblzma.dll 时，才把 base 的 Library/bin 加入 DLL 搜索路径，
# 让 _lzma.pyd 能找到 liblzma.dll（非 Windows 跳过，不影响 Linux/Mac）
if sys.platform == "win32":
    _conda_base_lib_bin = os.path.normpath(
        os.path.join(os.path.dirname(sys.executable), "..", "..", "Library", "bin")
    )
    # 只有真的缺 liblzma.dll 时才动 DLL 搜索路径。
    # 无条件 add_dll_directory 会把 base 环境的 libssl 一起带进来，遮蔽本环境自己的 OpenSSL，
    # 导致 _ssl 加载失败 —— 而那会连锁打死 fastapi/httpx 的整条导入链。
    if os.path.isdir(_conda_base_lib_bin):
        try:
            import _lzma  # noqa: F401
        except ImportError:
            os.add_dll_directory(_conda_base_lib_bin)

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import get_settings                          # 配置（第 3 章）
from backend.core.logger import configure_logging, get_logger    # 日志（3.3）
from backend.api.router import api_router                        # 8.6.1 聚合的总路由

settings = get_settings()

# MCP Sub-Apps（第 5.11 章实现）在 lifespan 定义前创建，
# 这样 lifespan 与下面的 mount 共享同一实例
# 知识库 MCP（search_knowledge_base → 检索研报语料 report_corpus）；
# 联网搜索 MCP（轨道 B 低置信兜底用）。
from backend.mcp.web_search_server import mcp as ws_mcp          # 联网搜索 MCP（5.11）
from backend.mcp.knowledge_base_server import mcp as kb_mcp      # 知识库 MCP → 研报库
from backend.mcp.data_source_server import mcp as ds_mcp         # 采集 MCP → 四源取数

_ws_app = ws_mcp.streamable_http_app()
_kb_app = kb_mcp.streamable_http_app()
_ds_app = ds_mcp.streamable_http_app()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 生命周期钩子：启动时执行迁移/模型预热/MCP 子应用进入，关闭时清理缓存。

    :param app: FastAPI 应用实例。
    :return: 异步生成器；应用运行期间在此挂起，退出时执行关闭清理。
    """
    configure_logging()                                  # 初始化结构化日志（3.3）
    logger = get_logger(__name__)
    logger.info("app.starting | env=%s port=%s", settings.app_env, settings.app_port)

    # ① DB Schema 自动迁移（幂等，每次启动执行）—— 第 3 章的建表/迁移
    try:
        from backend.db.migrations import run_migrations
        await run_migrations()
    except Exception as e:
        logger.warning("app.migrations_failed | error=%s", e)   # 迁移失败只告警，不拦启动

    # ② 并行预热三个本地模型（首次加载慢，提前热好，避免首个请求卡顿）
    #    精排 + 嵌入 + 轨道 B 意图分类器（QueryClassifier 只在这里热，运行期懒单例）
    import asyncio
    try:
        from backend.core.reranker import BGEReranker            # 重排序模型（5.8）
        from backend.core.embedding import BGEMEmbedder              # BGE-M3 嵌入（5.4）
        from backend.core.query_classifier import QueryClassifier     # 轨道 B L2 意图分类

        loop = asyncio.get_running_loop()
        await asyncio.gather(                                    # 三个模型并行加载（各跑在线程池里）
            loop.run_in_executor(None, BGEReranker.get_instance),
            loop.run_in_executor(None, BGEMEmbedder.get_instance),
            loop.run_in_executor(None, QueryClassifier.get_instance),
        )
        logger.info("app.local_models_warmed_up")
    except Exception as e:
        logger.warning("app.local_models_warmup_failed | error=%s", e)  # 预热失败也不拦启动

    # ③ 驱动 MCP Server 的 lifespan（mount 不会自动调用子应用的 lifespan，需手动嵌套进入）
    async with _kb_app.router.lifespan_context(_kb_app):
        async with _ws_app.router.lifespan_context(_ws_app):
            async with _ds_app.router.lifespan_context(_ds_app):
                logger.info("app.started")
                yield                                        # ← 应用运行期间停在这里

                # ── 关闭时执行 ──
                logger.info("app.shutting_down")
                from backend.core.llm_factory import LLMFactory   # LLM 工厂（3.4）
                LLMFactory.clear_cache()                          # 清缓存
                logger.info("app.shutdown_complete")

app = FastAPI(
    title="ResearchAgent API",
    description="智能投研分析助手 API",
    version="1.0.0",
    docs_url="/docs",                                    # Swagger 文档
    redoc_url="/redoc",                                  # ReDoc 文档
    lifespan=lifespan,                                   # 挂上面的生命周期钩子
)

# CORS：允许前端开发端口跨域访问（3000/5173/8080）
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", "http://localhost:5173", "http://localhost:8080",
        "http://127.0.0.1:3000", "http://127.0.0.1:5173", "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")        # 挂总路由（8.6.1），所有业务接口在 /api/v1 下

app.mount("/mcp/kb", _kb_app)                            # 挂知识库 MCP 子应用（5.11）
app.mount("/mcp/web-search", _ws_app)                   # 挂联网搜索 MCP 子应用（5.11）
app.mount("/mcp/data-source", _ds_app)                  # 挂采集 MCP 子应用（四源取数）


@app.get("/health", tags=["系统"])                       # 健康检查（运维探活用）
async def health_check():
    """健康检查探活接口。

    :return: dict，结构 {"status": "ok", "env": str}。
    """
    return {"status": "ok", "env": settings.app_env}


# ── 启动入口 ────────────────────────────────────────────────
# 项目惯例是用 `uvicorn backend.main:app --port 8000` 命令行启动。
# 这个 __main__ 入口是【补充】：让 `python -m backend.main` 也能直接起服务，
# 方便没有装 uvicorn 脚手架、或习惯用 python -m 的开发者。
# 注意与命令行启动的行为等价：都走同一套 lifespan（迁移/checkpointer/模型预热）。
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.app_host,
        port=settings.app_port,
        # 不默认开 reload：见 04 章 —— Windows 上 --reload 会派生子进程，
        # 是之前占着 8000 端口僵尸进程的成因。要热重载请显式传 --reload 或
        # 改用 uvicorn 命令行 + 明确的 uvicorn.run(reload=True)。
    )


