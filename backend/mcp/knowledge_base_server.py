# backend/mcp/knowledge_base_server.py
# 知识库 MCP（FastMCP streamable HTTP）：检索**研报语料 report_corpus**。
#
# 与旧知识库（knowledge_domain）的差异：检索目标换成**研报库**，
# tool 复用研报读侧正规入口 search_published_reports（按 tenant 全库混合检索 +
# BGE-Reranker 精排），不再另起一套 retrieve 实现。

import asyncio

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    name="ResearchAgent-KnowledgeBase",
    stateless_http=True,
    json_response=True,
)


@mcp.tool()
async def search_knowledge_base(
    query: str,
    tenant_id: str = "tenant_default",
    top_k: int = 3,
    recall_top_k: int | None = None,
    rerank_top_k: int | None = None,
) -> list[dict]:
    """
    检索已发布的研报语料（report_corpus），返回最相关的研报切片。

    语义与旧知识库一致：query → docs。只把检索目标换成研报库：
    按 tenant 全库混合检索（BGE-M3 dense+sparse → WeightedRanker 融合 →
    BGE-Reranker 精排），不做公司/行业方向过滤。

    Args:
        query:        检索提问（中文自然语言）
        tenant_id:    租户 id，默认 tenant_default
        top_k:        返回条数（默认 3）、不回退时的兜底精排条数
        recall_top_k: 召回条数（可选）；传入则用调用方档位，缺省推导 max(top_k*3, 10)
        rerank_top_k: 精排返回条数（可选）；传入则用调用方档位，缺省 = top_k

    Returns:
        研报切片列表，每项含 content / score / source_name /
        company_code / industry / report_type / published_at。
        检索失败返回空 list（记日志，不抛）。
    """
    from backend.core.logger import get_logger
    from backend.core.reranker import search_published_reports

    logger = get_logger(__name__)

    # QA 检索按 query_type 分了三档 recall/rerank（PRECISE/VAGUE/BROAD），
    # 由调用方传入真实档位；缺省时回落本 server 原本的 top_k 推导，保持兼容。
    recall = recall_top_k if recall_top_k is not None else max(top_k * 3, 10)
    rerank = rerank_top_k if rerank_top_k is not None else top_k

    try:
        # search_published_reports 同步阻塞（BGE-M3 编码 + Milvus 检索 + 精排），
        # 用 to_thread 包，避免阻塞 MCP 的事件循环。
        docs, confidence = await asyncio.to_thread(
            search_published_reports,
            query=query,
            tenant_id=tenant_id,
            recall_top_k=recall,
            rerank_top_k=rerank,
        )
        logger.info("kb_mcp.search_done", hits=len(docs), confidence=confidence)
        return [
            {
                "content":      doc["content"],
                "score":        doc["score"],
                "source_name":  doc["metadata"].get("source_name") or "",
                "company_code": doc["metadata"].get("company_code") or "",
                "industry":     doc["metadata"].get("industry") or "",
                "report_type":  doc["metadata"].get("report_type") or "",
                "published_at": doc["metadata"].get("published_at"),
            }
            for doc in docs
        ]
    except Exception as e:
        logger.error("kb_mcp.search_failed", error=str(e))
        return []


if __name__ == '__main__':
    import uvicorn

    port = 8001
    print(f"KnowledgeBase MCP Server → http://localhost:{port}/mcp")
    uvicorn.run(mcp.streamable_http_app(), host="0.0.0.0", port=port)