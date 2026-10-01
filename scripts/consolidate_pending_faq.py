# scripts/consolidate_pending_faq.py
# 消费某租户待补料队列：把 status='pending' 的低置信问题按语义聚类成 FAQ，
# 落 qa_faq 表、可选回灌 report_corpus，再把已覆盖的 pending 行标为 consumed。
#
# 用法：
#   python scripts/consolidate_pending_faq.py                      # 只清点到 qa_faq
#   python scripts/consolidate_pending_faq.py --tenant tenant_x   # 指定租户
#   python scripts/consolidate_pending_faq.py --ingest            # 额外回灌 Milvus 可被 RAG 检索
#
# 说明：--ingest 需要 BGE 嵌入 + Milvus 就绪（等价跑一次研发研报回灌）；
# 定时触发（cron / 后台任务）走同一入口。

import argparse
import asyncio
import json
import os
import sys

# 脚本在 scripts/ 下直接跑，把项目根塞进 sys.path 才能 import backend。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.stdout.reconfigure(encoding="utf-8")

from backend.agents.qa.consolidate import consolidate_pending_faq


async def _main() -> int:
    parser = argparse.ArgumentParser(description="待补料问题闭环成 FAQ")
    parser.add_argument("--tenant", default="tenant_default", help="目标租户（默认 tenant_default）")
    parser.add_argument("--ingest", action="store_true",
                        help="回灌 report_corpus（report_type=faq）使 FAQ 可被检索")
    args = parser.parse_args()

    result = await consolidate_pending_faq(args.tenant, ingest=args.ingest)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))