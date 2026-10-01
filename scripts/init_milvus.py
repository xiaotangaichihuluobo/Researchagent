# scripts/init_milvus.py
# 执行：python scripts/init_milvus.py
#
# 沿用「一个业务域一个集合、靠 tenant_id 过滤隔离」的写法，建研报语料集合并灌样例：
#   report_corpus —— 投研研报：company_code / industry / report_type / published_at
# 默认顺带把研报样例语料灌入 report_corpus（--no-seed-reports 跳过）。
# 旧知识库 knowledge_domain 已移除（见 backend/core/knowledge_base.py）。
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# 脚本在 scripts/ 下直接跑，把项目根塞进 sys.path 才能 import backend。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pymilvus import MilvusClient, DataType
from backend.config import get_settings


MILVUS_URI = f"http://{get_settings().milvus_host}:{get_settings().milvus_port}"
VECTOR_DIM = 1024                  # BGE-M3 稠密向量维度

# ── report_corpus：研报语料──────────────────────────────────
REPORT_COLLECTION = get_settings().report_collection_name
REPORT_SCALAR_FIELDS: list[tuple] = [
    ("content",      DataType.VARCHAR, 4096),
    ("tenant_id",    DataType.VARCHAR, 64),
    ("company_code", DataType.VARCHAR, 32),
    ("industry",     DataType.VARCHAR, 64),
    ("report_type",  DataType.VARCHAR, 16),
    ("published_at", DataType.INT64, None),
]
REPORT_INVERTED_FIELDS = ("tenant_id", "company_code", "industry", "report_type")


def build_report_schema(client: MilvusClient):
    """研报语料集合 schema：双向量 + 研报道义标量字段（沿用 REPORT_SCALAR_FIELDS）。"""
    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=VECTOR_DIM)
    schema.add_field("sparse_embedding", DataType.SPARSE_FLOAT_VECTOR)
    for name, dtype, max_length in REPORT_SCALAR_FIELDS:
        if max_length is None:
            schema.add_field(name, dtype)
        else:
            schema.add_field(name, dtype, max_length=max_length)
    return schema


def build_report_index_params(client: MilvusClient):
    """研报语料集合索引：稠密 HNSW、稀疏 SPARSE_INVERTED、标量 INVERTED。"""
    ip = client.prepare_index_params()
    ip.add_index(field_name="embedding", index_type="HNSW", metric_type="COSINE",
                 params={"M": 16, "efConstruction": 256})
    ip.add_index(field_name="sparse_embedding", index_type="SPARSE_INVERTED_INDEX",
                 metric_type="IP", params={"drop_ratio_build": 0.2})
    for name in REPORT_INVERTED_FIELDS:
        ip.add_index(field_name=name, index_type="INVERTED")
    return ip


def _create(
    client: MilvusClient,
    name: str,
    schema_builder,
    index_builder,
) -> None:
    """建集合（可重跑）：已存在先删后建，清空旧数据。"""
    if client.has_collection(name):
        print(f"🗑️  删除旧集合 '{name}'...")
        client.drop_collection(name)
    # create_collection 传 index_params 会一并建索引并加载
    client.create_collection(
        collection_name=name,
        schema=schema_builder(client),
        index_params=index_builder(client),
    )
    print(f"✅ 集合 '{name}' 创建完成（含索引，已加载）")


FIXTURES = Path(__file__).resolve().parent.parent / "backend" / "agents" / "retrieve" / "fixtures"


async def seed_report_corpus() -> None:
    """灌入研报样例语料（幂等可重跑）。切分/嵌入/Milvus 写入/PG 登记全部
    沿用 backend.core.report_ingest.ingest_report，不在初始化脚本里重写一遍。"""
    from backend.core.report_ingest import ingest_report   # noqa: E402
    from backend.core import research_repo as repo          # noqa: E402
    tenant_id = "tenant_default"
    total = 0

    for path in sorted(FIXTURES.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        industry = data.get("industry") or ""
        for item in data["items"]:
            company_code = item.get("company_code") or data.get("company_code") or ""
            published = item.get("published_at")
            published_at = (datetime.fromisoformat(published) if published else None)
            # 切片主键前缀。优先用 item 显式的 key（published_reports.json 等种子
            # 已预置唯一值，避免标题 hash 碰撞导致后篇覆盖前篇）；缺省再回退标题 hash。
            key = f"sample-{item.get('key') or (path.stem + '-' + str(abs(hash(item['title'])) % 10**8))}"
            count = await ingest_report(
                tenant_id=tenant_id, report_key=key, title=item["title"],
                content=item["content"], company_code=company_code,
                industry=item.get("industry") or industry,
                report_type=item["report_type"], published_at=published_at,
            )
            total += count
            print(f"  ✅ {item['title']}（{count} 片）")

    print(f"\n共灌入 {total} 个切片，集合 '{REPORT_COLLECTION}'")
    print("📋 PG 登记行：")
    for row in await repo.list_report_corpus(tenant_id):
        print(f"  - {row['title']}｜{row['industry']}｜{len(row['milvus_ids'])} 片")


def main():
    print(f"连接 Milvus：{MILVUS_URI}")
    client = MilvusClient(uri=MILVUS_URI)

    _create(client, REPORT_COLLECTION, build_report_schema, build_report_index_params)

    asyncio.run(seed_report_corpus())
    print("⚠️  集合已重建，原有数据已清空。研报语料库 report_corpus 已重建。")


if __name__ == "__main__":
    main()