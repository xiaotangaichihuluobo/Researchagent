# backend/core/report_ingest.py
# 研报语料写入侧（report_corpus）：把一个研报的正文切成切片、嵌入、upsert 进
# Milvus，并在 PG 里登记一行。函数平铺、不套类——每个函数只做一件事：
#   chunk_text        —— 按固定字符窗口切分（纯文本来源，中文无词边界）
#   chunk_ids         —— 切片主键 {report_key}:{index}
#   _embed_and_upsert —— 同步嵌入 + upsert（由调用方 to_thread 包）
#   ingest_report_chunks —— 入参是**已分好**的切片列表（md/pdf 智能切块后走这）
#   ingest_report     —— 入参是一段 content（内部用 chunk_text 切）
# 检索侧见 reranker.py 的 search_reports / search_published_reports。

import asyncio
from datetime import datetime, timezone
from typing import Optional

from backend.config import get_settings
from backend.core.embedding import BGEMEmbedder
from backend.core.knowledge_base import KnowledgeBaseClient
from backend.core.logger import get_logger

logger = get_logger(__name__)

REPORT_COLLECTION = get_settings().report_collection_name


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """按字符窗口切分，带重叠。

    中文按字符切是可接受的 —— 没有英文那样的词边界问题，而按标点切会让块长
    极不均匀。子图 State 要求精排后 <= 5 条且正文按 report_chunk_chars 截断：
    子图内部字段仍会被 checkpoint，量大一样会撑。

    :param text: 待切分的原始正文
    :param size: 每个切块的字符窗口大小，必须为正数
    :param overlap: 相邻切块重叠的字符数，须满足 0 <= overlap < size
    :return: 切好的非空文本切片列表
    """
    if size <= 0:
        raise ValueError("size 必须为正")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap 必须满足 0 <= overlap < size")

    chunks: list[str] = []
    step = size - overlap
    start = 0
    while start < len(text):
        piece = text[start:start + size].strip()
        if piece:
            chunks.append(piece)
        start += step
    return chunks


def chunk_ids(report_key: str, count: int) -> list[str]:
    """切片主键：{report_key}:{index}。Milvus 的 id 列是 VARCHAR(64)。

    :param report_key: 文档唯一键，作主键的前缀
    :param count: 要生成的切片数量
    :return: 形如 {report_key}:{index} 的主键字符串列表
    """
    return [f"{report_key}:{i}" for i in range(count)]


def _embed_and_upsert(rows: list[dict]) -> list[str]:
    """一个研报的切片嵌入 + upsert 进 Milvus。**同步阻塞**，由调用方 to_thread 包。

    单独抽成一个函数是为了让测试能整体桩掉它 —— 测试不得加载 BGE 模型。
    嵌入本体沿用 BGEMEmbedder，写入沿用 KnowledgeBaseClient 的连接单例。

    :param rows: 待写入的切片行列表，每项含 id/content/tenant_id/company_code/
        industry/report_type/published_at；嵌入结果会写回其 embedding / sparse_embedding
    :return: 写入成功的切片主键 id 列表
    """
    embedder = BGEMEmbedder.get_instance()
    contents = [row["content"] for row in rows]
    dense, sparse = embedder.encode(contents)          # 批量编码

    payload = []
    for i, row in enumerate(rows):
        item = dict(row)
        item["embedding"] = dense[i]
        item["sparse_embedding"] = sparse[i]
        payload.append(item)

    kb = KnowledgeBaseClient()
    kb._client.upsert(collection_name=REPORT_COLLECTION, data=payload)
    return [row["id"] for row in rows]


async def ingest_report_chunks(*, tenant_id: str, report_key: str, title: str,
                               chunks: list[str], company_code: str,
                               industry: str, report_type: str,
                               published_at: Optional[datetime],
                               company_id: Optional[str] = None) -> int:
    """把**已分好的**研报切片嵌入、写进 Milvus，并在 PG 里登记一行。返回切片数。

    :param tenant_id: 租户隔离键（检索过滤恒带它）
    :param report_key: 文档唯一键，切片主键前缀 {report_key}:{index}
    :param title: 研报标题，PG 登记行的标识之一
    :param chunks: 已切分好的正文切片列表（本函数不再切，接受的是成块的文本）
    :param company_code: 公司代码（横向/纵向检索的过滤字段）
    :param industry: 所属行业（横向可比检索的过滤字段）
    :param report_type: 研报类型（如 "深度报告"/"年报"）
    :param published_at: 发布日期，PG 登记行标识之一；缺省用当前时间戳打档
    :param company_id: PG 侧公司外键（可空，非必填的冗余关联）
    :return: 写入的切片数

    与 ingest_report 的唯一差别：入参不是一段 content，而是切分好的 chunk 文本
    列表。这样后台直接导入 .md/.pdf 时，能先用 md/pdf 智能切块（按标题/按页，
    见 scripts/build_knowledge_base.py），复用这一层「嵌入 + report_corpus 写入 +
    PG 登记」的底座，而不是只能走字符窗口。

    【本函数会抛】—— 「回灌绝不抛出」是 publish_report_node 的策略，不是这里的。
    把失败吞在这里会让调用方无法区分「没写」与「写了但报错」。

    幂等：同一个 report_key 重复回灌，Milvus 侧 upsert 覆盖同名主键，
    PG 侧按 (tenant_id, title, published_at) 复用已有登记行。
    """
    from backend.core import research_repo as repo

    if not chunks:
        logger.warning("report_ingest.empty_content", report_key=report_key)
        return 0

    ids = chunk_ids(report_key, len(chunks))
    stamp = int((published_at or datetime.now(timezone.utc)).timestamp())
    rows = [
        {"id": ids[i], "content": chunks[i], "tenant_id": tenant_id,
         "company_code": company_code, "industry": industry or "",
         "report_type": report_type, "published_at": stamp}
        for i in range(len(chunks))
    ]
    stored_ids = await asyncio.to_thread(_embed_and_upsert, rows)

    await repo.upsert_report_corpus(
        tenant_id=tenant_id, report_id=None, company_id=company_id,
        industry=industry, title=title, report_type=report_type,
        published_at=published_at, milvus_ids=stored_ids,
    )
    logger.info("report_ingest.ingested", report_key=report_key,
                chunks=len(stored_ids))
    return len(stored_ids)


async def ingest_report(*, tenant_id: str, report_key: str, title: str,
                        content: str, company_code: str, industry: str,
                        report_type: str, published_at: Optional[datetime],
                        company_id: Optional[str] = None) -> int:
    """把一份研报切分、嵌入、写进 Milvus，并在 PG 里登记一行。返回切片数。

    :param tenant_id: 租户隔离键（检索过滤恒带它）
    :param report_key: 文档唯一键，切片主键前缀 {report_key}:{index}
    :param title: 研报标题，PG 登记行的标识之一
    :param content: 整段正文（内部用 chunk_text 按字符窗切成 chunks；已在
        md/pdf 侧智能切块时请直接走 ingest_report_chunks）
    :param company_code: 公司代码（横向/纵向检索的过滤字段）
    :param industry: 所属行业（横向可比检索的过滤字段）
    :param report_type: 研报类型（如 "深度报告"/"年报"）
    :param published_at: 发布日期，PG 登记行标识之一；缺省用当前时间戳打档
    :param company_id: PG 侧公司外键（可空，非必填的冗余关联）
    :return: 写入的切片数

    对纯文本走固定字符窗口 chunk_text；若后台导入的是 .md/.pdf，请改用
    ingest_report_chunks，先用 md/pdf 智能切块再落库。

    【本函数会抛】—— 「回灌绝不抛出」是 publish_report_node 的策略，不是这里的。
    把失败吞在这里会让调用方无法区分「没写」与「写了但报错」。

    幂等：同一个 report_key 重复回灌，Milvus 侧 upsert 覆盖同名主键，
    PG 侧按 (tenant_id, title, published_at) 复用已有登记行。
    """
    settings = get_settings()
    chunks = chunk_text(content, settings.report_chunk_chars,
                        settings.report_chunk_overlap)
    return await ingest_report_chunks(
        tenant_id=tenant_id, report_key=report_key, title=title, chunks=chunks,
        company_code=company_code, industry=industry, report_type=report_type,
        published_at=published_at, company_id=company_id,
    )


def _delete_vectors_from_milvus(ids: list[str]) -> int:
    """一批切块向量从 Milvus 按主键精确删除。**同步阻塞**，由调用方 to_thread 包。

    用主键 `ids` 删、而不是 `id like "report_key:%"` 前缀删 —— ids 来自 PG 登记行
    的 milvus_ids，本身就是精确集合，不依赖集合里额外的标量索引。单独抽成函数
    是为了让测试能整体桩掉它（与 _embed_and_upsert 同构，测试不得直连 Milvus）。

    :param ids: 要删除的切片主键列表（可空，空则直接返回 0）
    :return: 实际请求删除的主键数量
    """
    if not ids:
        return 0
    kb = KnowledgeBaseClient()
    kb._client.delete(collection_name=REPORT_COLLECTION, ids=ids)
    return len(ids)


async def delete_report_vectors(*, tenant_id: str, report_key: str) -> int:
    """把一篇文档在语料库里的全部切块向量删除，并同步清掉对应的 PG 登记行。返回删掉的切片数。

    与 ingest_report / ingest_report_chunks 对称：用同一个 report_key 标识一篇文档
    （切块主键 {report_key}:{index}），删除就是回灌的反向操作。report_key 不独立
    落列，靠 PG 登记行的 milvus_ids 前缀反查 —— 见 research_repo.list_report_milvus_ids_by_key。

    【顺序关键】先删 Milvus 向量、后删 PG 登记行 —— 若 Milvus 删除抛异常则整函数上抛，
    PG 行保留，不留下「向量没了、孤儿登记行还挂在 list_report_corpus 里」的半成品。
    反过来先删登记行、再删向量失败，就会留下一批检索得到的幽灵切块，更糟。

    【本函数会抛】—— 与 ingest 一致，不把失败吞掉，让调用方分清「删干净了」与「删失败」。

    幂等：report_key 下无登记行/无切块 → 返回 0，不报错。删除「已不存在的东西」是合法的。

    :param tenant_id: 租户隔离键
    :param report_key: 文档唯一键，用于定位并删除该文档的切块与登记行
    :return: 删除的切片数；0 表示该文档无登记行/无切块
    """
    from backend.core import research_repo as repo

    ids = await repo.list_report_milvus_ids_by_key(tenant_id, report_key)
    if not ids:
        logger.info("report_ingest.delete_noop", tenant_id=tenant_id,
                    report_key=report_key)
        return 0
    await asyncio.to_thread(_delete_vectors_from_milvus, ids)
    await repo.delete_report_corpus(tenant_id, report_key)
    logger.info("report_ingest.deleted", tenant_id=tenant_id,
                report_key=report_key, chunks=len(ids))
    return len(ids)