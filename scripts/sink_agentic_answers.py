# scripts/sink_agentic_answers.py
# 消费 agentic 高质量答案沉淀桶：把 qa_sediment_queue 里 status='pending' 的
# 现成问答双落成可复用知识——qa_faq(PG 可审计) + 可选回灌 report_corpus(Milvus 可检索)，
# 再把对应桶行标为 consumed —— 闭环。
#
# 与 scripts/consolidate_pending_faq.py 的区别：那边消费的是「没答案的低置信问题」，
# 靠 LLM 聚类现场生成答案；这里消费的是「已有现成答案」的 agentic 终局，直接落、不做聚类。
#
# 用法：
#   python scripts/sink_agentic_answers.py                      # 双落 qa_faq（PG 必做）
#   python scripts/sink_agentic_answers.py --tenant tenant_x   # 指定租户
#   python scripts/sink_agentic_answers.py --ingest            # 额外回灌 Milvus 可被 RAG 检索
#
# 说明：--ingest 需要 BGE 嵌入 + Milvus 就绪（等价跑一次研报回灌）；手动/定时走同一入口。

import argparse
import asyncio
import json
import os
import sys

# 脚本在 scripts/ 下直接跑，把项目根塞进 sys.path 才能 import backend。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.stdout.reconfigure(encoding="utf-8")

from backend.core import qa_repo
from backend.agents.qa.consolidate import _ingest_faqs_to_corpus


async def _main() -> int:
    parser = argparse.ArgumentParser(description="agentic 高质量答案双落成可检索知识")
    parser.add_argument("--tenant", default="tenant_default", help="目标租户（默认 tenant_default）")
    parser.add_argument("--ingest", action="store_true",
                        help="回灌 report_corpus（report_type=faq）使答案可被检索")
    args = parser.parse_args()

    pending = await qa_repo.list_sediment_pending(args.tenant)
    if not pending:
        print(json.dumps({"pending_seen": 0, "faqs_added": 0, "ingested": 0,
                          "consumed": 0, "left_pending": 0}, ensure_ascii=False))
        return 0

    # 1) PG 双落：qa_faq 是主存、可审计（幂等由 (tenant_id, question) 保证）。
    for p in pending:
        await qa_repo.add_faq(args.tenant, p["question"], p["answer"], source_count=1)
    # 2) 可选回灌 Milvus：复用 consolidate 的现成回灌（逐条 try、失败只记日志不中断）。
    ingested = await _ingest_faqs_to_corpus(args.tenant, pending) if args.ingest else 0
    # 3) 已双落的桶行标为 consumed（回写，闭环）。
    consumed = await qa_repo.mark_sediment_consumed(
        args.tenant, [p["id"] for p in pending])

    print(json.dumps({"pending_seen": len(pending), "faqs_added": len(pending),
                      "ingested": ingested, "consumed": consumed,
                      "left_pending": len(pending) - consumed}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))