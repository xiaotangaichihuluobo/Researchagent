# backend/core/knowledge_base.py
# Milvus 客户端（MilvusClient 版），服务研报语料 report_corpus。
#
# 单一职责：Milvus 连接单例 + 混合检索基元 _hybrid_search。嵌入模型
# （BGEMEmbedder）抽到 embedding.py，研报写入侧（chunk_text / ingest_report）
# 抽到 report_ingest.py —— 本文件只留这个客户端，读侧消费方（reranker.py 的
# _vector_search / search_published_reports）通过 collection_name / output_fields
# 显式选中研报集合。

from typing import Optional

from pymilvus import MilvusClient, AnnSearchRequest, WeightedRanker

from backend.config import get_settings
from backend.core.logger import get_logger, configure_logging
configure_logging()
logger = get_logger(__name__)


class KnowledgeBaseClient:
    """
    Milvus 客户端（MilvusClient 版），服务研报语料 report_corpus。

    _hybrid_search 的 collection_name / output_fields 由研报侧消费方
    （reranker.py 的 _vector_search / search_published_reports）显式传入。

    单例连接：_client 是类变量，整个进程只创建一次 MilvusClient 连接。
    """

    _client: Optional["MilvusClient"] = None
    _loaded: bool = False

    # HNSW 搜索时的候选集大小，精度/速度平衡点（_hybrid_search 使用）
    ANN_EF = 64

    def __init__(self):
        """获取 Milvus 客户端单例连接并加载集合。

        :return: 无返回值
        """
        if KnowledgeBaseClient._client is None:
            settings = get_settings()
            uri = f"http://{settings.milvus_host}:{settings.milvus_port}"
            KnowledgeBaseClient._client = MilvusClient(uri=uri)
            logger.info("milvus.connected", uri=uri)

        if not KnowledgeBaseClient._loaded:
            try:
                KnowledgeBaseClient._client.load_collection(
                    get_settings().report_collection_name)
            except Exception:
                pass   # init_milvus.py 尚未运行时忽略
            KnowledgeBaseClient._loaded = True

    # ── 检索配置 ─────────────────────────────────────────────

    VECTOR_TOP_K = 10  # Hybrid 召回的候选数量，传给 Reranker 精排

    def _hybrid_search(
            self,
            query_embedding: list[float],
            query_sparse: dict,
            top_k: int,
            filters: Optional[str] = None,
            collection_name: Optional[str] = None,
            output_fields: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        对 Milvus 集合做 Hybrid 检索（Dense + Sparse → WeightedRanker 融合）。

        两个 AnnSearchRequest 分别构造 Dense 和 Sparse 检索请求，
        由 Milvus 在服务端并行执行后，用 WeightedRanker 加权融合排序。

        :param query_embedding: Dense Query 向量（1024 维，来自 encode_query）
        :param query_sparse: Sparse Query 向量（{token_id: weight}，来自 encode_query）
        :param top_k: 每路召回数量（融合后同样取 top_k）
        :param filters: Milvus bool 表达式，如 'tenant_id == "xxx"'；None 则不过滤
        :param collection_name: 目标集合名；None 时取配置的报集合 report_collection_name
        :param output_fields: Milvus 返回的实体字段：content + company_code/industry/
            report_type/published_at；None 时用默认字段组合
        :return: 候选文档列表，每项含 "content" / "score" / "metadata"。
            metadata 是 output_fields 里除 content 外的字段原样字典；
            score 是 WeightedRanker 的加权排序信号，不是概率，交给 Reranker 做精细打分。
            异常时返回空列表
        """
        try:
            # ── Dense ANN 检索请求 ─────────────────────────────────────
            # COSINE 度量匹配 BGE-M3 dense 向量（L2 归一化后等价于余弦相似度）
            # ef=64：HNSW 搜索时的候选集大小，越大精度越高，64 是精度/速度平衡点
            dense_req = AnnSearchRequest(
                data=[query_embedding],
                anns_field="embedding",
                param={
                    "metric_type": "COSINE",
                    "params": {"ef": self.ANN_EF},
                },
                limit=top_k,
                expr=filters,
            )

            # ── Sparse 关键词检索请求 ──────────────────────────────────
            # IP（内积）是 BGE-M3 lexical_weights 的标准度量
            sparse_req = AnnSearchRequest(
                data=[query_sparse],
                anns_field="sparse_embedding",
                param={"metric_type": "IP"},
                limit=top_k,
                expr=filters,
            )

            if collection_name is None:
                collection_name = get_settings().report_collection_name
            if output_fields is None:
                output_fields = ["content", "company_code", "industry",
                                 "report_type", "published_at"]

            # ── WeightedRanker(0.7, 0.3) ──────────────────────────────
            # 第一个权重对应第一个请求（Dense），第二个对应第二个请求（Sparse）
            # 两路结果在 Milvus 服务端并行检索，融合后返回
            results = self._client.hybrid_search(
                collection_name=collection_name,
                reqs=[dense_req, sparse_req],
                ranker=WeightedRanker(0.7, 0.3),
                limit=top_k,
                output_fields=output_fields,
            )
            candidates = []
            for hit in results[0]:
                # metadata 是除 content 外的实体字段原样字典 —— 研报侧带
                # company_code/industry/report_type/published_at，由 reranker
                # 里的消费方取用。
                metadata = {
                    k: hit["entity"].get(k)
                    for k in output_fields if k != "content"
                }
                candidates.append({
                    "content": hit["entity"].get("content") or "",
                    "score":   hit.get("distance") or 0.0,
                    "metadata": metadata,
                })
            logger.info(
                "knowledge_base.hybrid_search_done",
                candidates=len(candidates),
            )
            return candidates
        except Exception as e:
            logger.error("knowledge_base.hybrid_search_failed", error=str(e))
            return []


if __name__ == '__main__':
    # 研报语料 smoke test：嵌入一条 query，在 report_corpus 上做混合检索。
    from backend.core.embedding import BGEMEmbedder
    model = BGEMEmbedder.get_instance()
    dense, sparse = model.encode_query(text="白酒行业的盈利模式")
    kb = KnowledgeBaseClient()
    results = kb._hybrid_search(
        dense, sparse, top_k=5,
        collection_name=get_settings().report_collection_name,
        output_fields=["content", "company_code", "industry",
                       "report_type", "published_at"],
    )
    print(f'results[0]: {results[0] if results else "(空)"}')