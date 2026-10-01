
import os
from dataclasses import dataclass
from typing import Optional

import torch
from sentence_transformers import CrossEncoder

from backend.config import get_settings
from backend.core.logger import get_logger

logger = get_logger(__name__)
backend_path = os.path.dirname(os.path.dirname(__file__))
RERANK_MAX_INPUT_CHARS = 512   # 截断过长文档，防止超出 CrossEncoder max_length=512


@dataclass
class RankedDocument:
    """精排后的单个文档结果"""
    content:        str    # 文档文本
    score:          float  # BGE-Reranker 输出的相关性概率 [0, 1]
    original_index: int    # 在原始召回列表中的位置（0 起）
    metadata:       dict   # 来源元数据（source_name / chunk_type / course_id 等）


class BGEReranker:
    """
    BGE-Reranker-v2-m3 精排服务（单例）。

    对 Hybrid 召回的候选文档做 CrossEncoder 精排，
    直接返回 [0, 1] 置信度，无需额外归一化。

    用法：
        reranker = BGEReranker.get_instance()
        docs, confidence = reranker.rerank_with_confidence(
            query="什么是 Spring IOC？",
            documents=candidates,
            top_k=3,
        )
    """

    _instance: Optional["BGEReranker"] = None

    def __init__(self):
        """加载 BGE-Reranker 模型（优先本地、否则回落到 HuggingFace）。

        :return: 无返回值
        """
        os.environ["ACCELERATE_USE_META_DEVICE"] = "0"
        settings = get_settings()
        model_path = os.path.join(backend_path, settings.reranker_model_path)

        use_local = (
            os.path.exists(model_path)
            and os.path.isdir(model_path)
            and any(f.endswith((".bin", ".safetensors", ".json")) for f in os.listdir(model_path))
        )
        # print(f'use_local: {use_local}')
        model_id = model_path if use_local else "BAAI/bge-reranker-v2-m3"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("reranker.loading", model_id=model_id, device=device)
        self._model = CrossEncoder(model_id, device=device, max_length=512)
        # print(f'self._model is {self._model}')
        logger.info("reranker.loaded", model_id=model_id)
    @classmethod
    def get_instance(cls) -> "BGEReranker":
        """获取单例，首次调用时加载模型。

        :return: BGEReranker 进程内单例实例
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def rerank_with_confidence(
            self,
            query: str,
            documents: list[dict],
            top_k: int = 3,
    ) -> tuple[list[RankedDocument], float]:
        """
        精排并返回置信度。

        :param query: 用户 Query
        :param documents: 候选文档列表，每项含 "content" 字段
        :param top_k: 返回文档数量，默认 3
        :return: 二元组 (ranked_docs, confidence)。
            ranked_docs 是按相关性降序排列的 RankedDocument，长度 <= top_k；
            confidence 是 Top-1 文档的 BGE 相关性概率 [0, 1]：
            ≥ 0.75 → 高置信度直接走 LLM 生成；< 0.75 → 低置信度触发 Web 兜底
        """
        if not documents:
            return [], 0.0

        # CrossEncoder 输入：(query, document) 对，截断过长文档
        pairs = [
            (query, (doc.get("content") or "")[:RERANK_MAX_INPUT_CHARS])
            for doc in documents
        ]
        # print(f'pairs[0]: {pairs[0]}')
        # print(f'len(pairs): {len(pairs)}')
        # CrossEncoder 默认 sigmoid 激活，predict() 直接输出 [0, 1] 概率
        scores: list[float] = self._model.predict(pairs).tolist()
        # print(f'scores: {scores}')
        ranked = sorted(
            [
                RankedDocument(
                    content=documents[i].get("content", ""),
                    score=scores[i],
                    original_index=i,
                    metadata=documents[i].get("metadata", {}),
                )
                for i in range(len(documents))
            ],
            key=lambda x: x.score,
            reverse=True,
        )
        # print(f'ranked: {ranked}')
        top_results = ranked[:top_k]
        confidence = top_results[0].score if top_results else 0.0

        logger.info(
            "reranker.done",
            candidates=len(documents),
            top_k=top_k,
            confidence=round(confidence, 4),
        )

        return top_results, confidence

# ──────────────────────────────────────────────────────────────
# 研报检索（P5）—— 沿用上面的 BGEMEmbedder / KnowledgeBaseClient /
# BGEReranker，只补研报领域自己的编排，不另起一份嵌入/检索/精排实现。
#
# 集合 = report_corpus。研报检索的关键在于「方向」：
#   横向（horizontal）= 可比公司（同行业、不同公司）
#   纵向（vertical）  = 本公司历史观点
# 两方向过滤条件互斥，所以各发一次查询、各自标注 direction，合并后一次精排，
# 再按方向配额取前 top_k —— 防止本公司历史研报把可比公司全部挤掉。
# 这里只放【读侧】。研报语料的写入（chunk_text / ingest_report /
# ingest_report_chunks）在 report_ingest.py；Milvus 连接/混合检索在
# knowledge_base.py；嵌入在 embedding.py。
# ──────────────────────────────────────────────────────────────

import asyncio

DIRECTIONS = ("horizontal", "vertical")


def build_filter(tenant_id: str, *, company_code: Optional[str] = None,
                 industry: Optional[str] = None,
                 exclude_company_code: Optional[str] = None) -> str:
    """拼 Milvus 的过滤表达式。租户条件永远在。

    :param tenant_id: 租户隔离键，恒出现在过滤表达式里
    :param company_code: 限定的公司代码；None（默认）则不限定
    :param industry: 限定的所属行业；None（默认）则不限定
    :param exclude_company_code: 需排除的公司代码（横向可比时排掉自己）；None（默认）不排除
    :return: 组装好的 Milvus 过滤表达式字符串
    """
    parts = [f'tenant_id == "{tenant_id}"']
    if company_code:
        parts.append(f'company_code == "{company_code}"')
    if exclude_company_code:
        # 横向必须排掉自己，否则「可比公司」里会混进本公司历史研报
        parts.append(f'company_code != "{exclude_company_code}"')
    if industry:
        parts.append(f'industry == "{industry}"')
    return " and ".join(parts)


@dataclass(frozen=True)
class DirectionSpec:
    """一次检索的方向及其过滤条件。"""
    direction: str
    filter_expr: str


def direction_filter(direction: str, *, tenant_id: str, company_code: str,
                     industry: str) -> DirectionSpec:
    """按方向给出过滤条件。

    未知方向必须响亮地抛 —— 悄悄降级成「无过滤」会把全库捞回来，
    那比报错更坏：结果看起来像是对的。

    :param direction: 检索方向，horizontal / vertical，未知则抛 ValueError
    :param tenant_id: 租户隔离键
    :param company_code: 公司代码，横向时排出自己、纵向时锁定本公司
    :param industry: 所属行业，横向可比检索的过滤字段
    :return: 对应的 DirectionSpec（含方向名与过滤表达式）
    """
    if direction == "horizontal":
        return DirectionSpec("horizontal",
                             build_filter(tenant_id, industry=industry,
                                          exclude_company_code=company_code))
    if direction == "vertical":
        return DirectionSpec("vertical", build_filter(tenant_id,
                                                      company_code=company_code))
    raise ValueError(f"未知检索方向：{direction!r}，只支持 {DIRECTIONS}")


def apply_direction_quota(ranked: list[dict], top_k: int) -> list[dict]:
    """按精排分数取前 top_k，但保证「本来有候选的方向」各至少留 1 条。

    为什么需要这条：同一个标的的历史研报（纵向）很容易把可比公司（横向）全部挤掉，
    而「两个方向」是设计目的。这条配额把「不挤占」从愿望变成可测的规则。

    做法：先给每个有候选的方向占住 1 个名额（取该方向分数最高的），
    再用全局分数补满剩余名额，最后按分数重排。空方向不占名额。

    :param ranked: 已按分数降序排列的候选文档列表，每项含 direction / score 键
    :param top_k: 需要返回的文档条数；<=0 返回空列表
    :return: 按配置额取前 top_k 的文档列表，按分数降序
    """
    if top_k <= 0 or not ranked:
        return []

    by_direction: dict[str, list[dict]] = {}
    for doc in ranked:                       # ranked 已按 score 降序，逐个按方向归堆
        direction = doc["direction"]
        if direction not in by_direction:    # 头一次见这个方向，先建个空表
            by_direction[direction] = []
        by_direction[direction].append(doc)  # 塞进它那堆

    # 第 1 步：每个方向先各占 1 个名额 —— 取该方向组内分数最高的那条（组内已降序，就是 docs[0]）
    chosen: list[dict] = []
    for docs in by_direction.values():
        if len(chosen) >= top_k:
            break
        chosen.append(docs[0])

    # 第 2 步：从全局降序依次补满剩余名额。
    # 用对象 id 记下「已被方向名额占掉」的（docs 是 dict 不可哈希，不能直接进 set）。
    occupied = {id(doc) for doc in chosen}
    for doc in ranked:
        if len(chosen) >= top_k:
            break
        if id(doc) in occupied:          # 这条已被方向名额占用，跳过
            continue
        chosen.append(doc)
        occupied.add(id(doc))

    # 最后按分数重排：横/纵分层抽出后顺序不是全局降序，得再排一次，取前 top_k。
    return sorted(chosen, key=lambda d: d["score"], reverse=True)[:top_k]


def _vector_search(spec: DirectionSpec, query: str) -> list[dict]:
    """一个方向的 Milvus **混合**检索（稠密 + 稀疏）。

    检索本体（AnnSearchRequest + hybrid_search + WeightedRanker 融合）由
    KnowledgeBaseClient._hybrid_search 提供 —— 这里只换过滤表达式和目标集合，
    并把命中结果标注上批 direction。**同步阻塞**，由调用方 to_thread 包。

    必须两路都发：BGE-M3 的稀疏向量承担关键词精确匹配（研报里「600519」「2025
    年报」这类词，稠密向量匹配得并不好）。只发稠密一路的话，稀疏索引建了就没人用。

    :param spec: 本次检索的方向规格（方向名 + 过滤表达式）
    :param query: 用户查询文本，用于编码查询向量
    :return: 命中列表，每项含 content / score / direction / source / report_type /
        published_at 键
    """
    from backend.core.embedding import BGEMEmbedder
    from backend.core.knowledge_base import KnowledgeBaseClient

    settings = get_settings()
    embedder = BGEMEmbedder.get_instance()
    dense, sparse = embedder.encode_query(query)

    kb = KnowledgeBaseClient()
    hits = kb._hybrid_search(
        query_embedding=dense, query_sparse=sparse,
        top_k=settings.report_recall_top_k,
        filters=spec.filter_expr,
        collection_name=settings.report_collection_name,
        output_fields=["content", "company_code", "industry",
                       "report_type", "published_at"],
    )
    return [{"content": hit["content"], "score": hit["score"],
             "direction": spec.direction,
             "source": hit["metadata"].get("source_name") or "",
             "report_type": hit["metadata"].get("report_type") or "",
             "published_at": hit["metadata"].get("published_at")}
            for hit in hits]


def _rerank(query: str, docs: list[dict], top_k: int) -> list[dict]:
    """BGE-Reranker 精排。**同步阻塞**，由调用方 to_thread 包。

    top_k 传 len(docs)：精排【全部】候选（两个方向合起来最多 2×recall_top_k），
    方向配额再过一道。只精排前 top_k 的话，被截掉的那条可能就是唯一能保住
    某个方向的候选 —— 配额会因此拿不到可换的东西。

    :param query: 用户查询文本，传给精排器打分
    :param docs: 待精排的候选文档列表（含 direction 等研究字段，无 score）
    :param top_k: 精排后返回的条数
    :return: 精排后按分数降序的前 top_k 条文档（每项带补上的 score 键）
    """
    if not docs:
        return []
    reranker = BGEReranker.get_instance()
    ranked, _ = reranker.rerank_with_confidence(query, docs, top_k=len(docs))

    # ranked 的每项只带 分数+原始下标；研究字段（direction/来源/特征等）还在原来的
    # docs dict 里 —— 所以按 original_index 把分数逐对贴回原 doc，再按分降序取前 top_k。
    score_by_index = {r.original_index: r.score for r in ranked}

    # 把重排分数贴回原 doc：逐个拷一份原 doc，再给它开一个 score 键存精排分
    enriched = []
    for i, doc in enumerate(docs):
        item = dict(doc)                              # 拷贝，别污染原 docs
        item["score"] = score_by_index.get(i, 0.0)    # original_index==i 那条的分数，贴上去
        enriched.append(item)

    # 按 score 字段从高到低排，取前 top_k
    enriched.sort(key=lambda d: d["score"], reverse=True)
    return enriched[:top_k]


async def search_reports(*, query: str, tenant_id: str, company_code: str,
                         industry: str, recall_top_k: int,
                         rerank_top_k: int) -> list[dict]:
    """两个方向各查一次 → 合并 → 一次精排 → 按配额取前 rerank_top_k。

    **两个方向各发一次查询**，不是一次查询兼两个方向：两者的过滤条件互斥
    （横向要求 company_code 不同、纵向要求相同），合成一次只能取并集，
    top_k 会在两个方向间互相挤占，且返回条目无法标注它属于哪个方向。
    代价是两倍往返；Milvus 在本地，这笔开销可忽略。

    :param query: 用户查询文本
    :param tenant_id: 租户隔离键
    :param company_code: 公司代码（本体，纵向检索锁定用）
    :param industry: 所属行业（横向可比检索的过滤字段）
    :param recall_top_k: 每路召回的候选条数
    :param rerank_top_k: 精排后按方向配额取回的条数
    :return: 精排并取配额后的文档列表（每项含 direction / score / content 等）
    """
    specs = [direction_filter(d, tenant_id=tenant_id, company_code=company_code,
                             industry=industry) for d in DIRECTIONS]

    merged: list[dict] = []
    for spec in specs:
        hits = await asyncio.to_thread(_vector_search, spec, query)
        merged.extend(hits)

    if not merged:
        logger.info("reranker.report_no_candidates", query_preview=query[:40])
        return []

    ranked = await asyncio.to_thread(_rerank, query, merged, len(merged))
    kept = apply_direction_quota(ranked, rerank_top_k)
    logger.info("reranker.report_searched", candidates=len(merged), kept=len(kept),
                directions=sorted({d["direction"] for d in kept}))
    return kept


# ──────────────────────────────────────────────────────────────
# 轨道 B：问已发布研报（全租户、股权稀疏的跨公司跟随式检索）
# ──────────────────────────────────────────────────────────────
# 与 search_reports 的分工：后者按 company_code/industry 做「方向」双向检索并带
# 方向配额，服务研报检索子图（绑定单一标的）。这里的「问已发布研报」面对多轮
# 跟随式问题 ——「那五粮液呢」会换公司，只能按 tenant_id 过滤 report_corpus 全库，
# 不能绑在某个标的上。复用 _hybrid_search + _rerank 既有基元，不另起并行模块。
#
# 注意：report_corpus 的 schema 没有 source_name 列（id 是 "{report_key}:{i}"），
# 来源标签用 company_code + report_type 拼一个可读、可去重的值（见 _report_source_label）。


def _report_source_label(company_code: str, report_type: str) -> str:
    """研报来源的可读标签（无 source_name 列，用公司+类型拼，便于前端展示与去重）。

    :param company_code: 公司代码；为空时返回通用标签「已发布研报」
    :param report_type: 研报类型；可选，非空时拼到标签尾部
    :return: 可读、可去重的来源标签字符串
    """
    if not company_code:
        return "已发布研报"
    return f"研报 · {company_code}" + (f" · {report_type}" if report_type else "")


def search_published_reports(*, query: str, tenant_id: str,
                             recall_top_k: int,
                             rerank_top_k: int) -> tuple[list[dict], float]:
    """全租户检索已发布研报（report_corpus），精排，返回 (docs, confidence)。

    - **同步阻塞**（BGE-M3 编码 + Milvus 检索 + 精排），调用方用 asyncio.to_thread 包，
      与 reranker.retrieve / _vector_search 一致。
    - docs: list[dict]，每项含 content / score / metadata（含 company_code、
      industry、report_type、published_at、source_name 可读标签）。

    :param query: 用户查询文本
    :param tenant_id: 租户隔离键（仅按租户过滤，不绑公司）
    :param recall_top_k: Hybrid 召回的候选条数
    :param rerank_top_k: 精排后返回的条数
    :return: 二元组 (docs, confidence)。docs 每项含 content / score / metadata；
        confidence 是 Top-1 精排概率 [0,1]
    """
    from backend.core.embedding import BGEMEmbedder
    from backend.core.knowledge_base import KnowledgeBaseClient

    settings = get_settings()
    embedder = BGEMEmbedder.get_instance()
    dense, sparse = embedder.encode_query(query)

    kb = KnowledgeBaseClient()
    hits = kb._hybrid_search(
        query_embedding=dense, query_sparse=sparse,
        top_k=recall_top_k,
        filters=build_filter(tenant_id),          # 仅租户：不绑公司
        collection_name=settings.report_collection_name,
        output_fields=["content", "company_code", "industry",
                       "report_type", "published_at"],
    )

    docs: list[dict] = []
    for hit in hits:
        meta = hit["metadata"]
        docs.append({
            "content": hit["content"],
            "score": hit["score"],
            "metadata": {
                **meta,
                "source_name": _report_source_label(
                    meta.get("company_code") or "",
                    meta.get("report_type") or "",
                ),
            },
        })

    ranked = _rerank(query, docs, rerank_top_k)
    confidence = ranked[0]["score"] if ranked else 0.0
    logger.info("reranker.report_published_searched", tenant_id=tenant_id,
                kept=len(ranked), confidence=round(confidence, 4))
    return ranked, confidence