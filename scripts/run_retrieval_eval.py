# scripts/run_retrieval_eval.py
# 轨道 B「问已发布研报」的离线检索/回答质量评估。
#
# 用法（需要真实 Milvus + report_corpus；选题/计时器，不进 pytest）：
#   python scripts/run_retrieval_eval.py --tenant tenant_default
#   python scripts/run_retrieval_eval.py --no-answer --recall-k 3
#
# 对每条 golden 样本：
#   · 检索  → 判断 expected_companies 是否出现在精排 Top-K 的 metadata.company_code → Recall@K
#   · 生成  → 用真实 LLM 走 build_rag_prompt 生成 → LLM-as-judge 给 1-5 分（answer 质量）
# 输出逐条明细 + 聚合 Recall@K 与平均分。期望无命中样本按「无期望 → 无召回为满分、有误召回为 0」。
#
# 复用既有基建，不另起模块：reranker.search_published_reports（同步，asyncio.to_thread 包）、
# agents.qa.prompts.build_rag_prompt、llm_factory.get_llm。

import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

JUDGE_PROMPT = (
    "你是 RAG 回答质量评审。基于下面给出的【参考内容】评估回答是否：1) 忠于参考、不编造；"
    "2) 直接回应了提问；3) 所需信息都已覆盖。\n"
    "【提问】{query}\n【参考内容】{context}\n【回答】{answer}\n\n"
    "只输出一个 1 到 5 的整数（1=错/编造，5=准确且完整覆盖）。"
)


def load_golden(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


async def _search(query: str, tenant: str, recall_k: int):
    from backend.core.reranker import search_published_reports

    docs, conf = await asyncio.to_thread(
        lambda: search_published_reports(
            query=query, tenant_id=tenant,
            recall_top_k=max(8, recall_k * 2), rerank_top_k=recall_k),
    )
    codes = [d.get("metadata", {}).get("company_code", "")
             for d in docs if d.get("metadata", {}).get("company_code")]
    return docs, codes, conf


async def _answer_and_judge(query: str, docs: list[dict]) -> tuple[str, float | None]:
    from backend.agents.qa.prompts import (
        build_rag_prompt, build_system_content,
    )
    from backend.core.llm_factory import get_llm

    context = "\n\n".join(
        f"【参考{i + 1}】\n{d['content']}" for i, d in enumerate(docs)) or "（无命中）"
    prompt = build_rag_prompt(
        system=build_system_content(None), history="（无历史对话）",
        context=context, query=query)
    llm = get_llm("qa_reports", temperature=0)
    answer = (await llm.ainvoke(prompt)).strip()
    try:
        raw = (await llm.ainvoke(
            JUDGE_PROMPT.format(query=query, context=context, answer=answer))).strip()
        score = float(raw.split()[0])
    except Exception:
        score = None
    return answer, score


async def run_one(row: dict, recall_k: int, do_answer: bool) -> dict:
    docs, codes, conf = await _search(row["question"], row["tenant"], recall_k)
    expected = row.get("expected_companies", []) or []
    hits = [c for c in expected if c in codes]
    recall = (len(hits) / len(expected)) if expected else (0.0 if codes else 1.0)

    answer, score = None, None
    if do_answer:
        answer, score = await _answer_and_judge(row["question"], docs)

    return {
        "question": row["question"],
        "expected": expected,
        "top_companies": codes,
        "recall": recall,
        "answer": answer,
        "judge": score,
    }


def main() -> int:
    # Windows 控制台默认 GBK，印 ✓/△/✗ 会 UnicodeEncodeError；强制 UTF-8 输出
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="轨道 B 检索/回答质量评估")
    parser.add_argument("--golden", default=os.path.join(PROJECT_ROOT, "samples",
                                                         "golden_retrieval.json"))
    parser.add_argument("--recall-k", type=int, default=3)
    parser.add_argument("--no-answer", action="store_true",
                        help="跳过 LLM 生成+判题，只跑检索 Recall")
    parser.add_argument("--tenant", default=None)
    args = parser.parse_args()

    golden = load_golden(args.golden)
    if args.tenant:
        for row in golden:
            row["tenant"] = args.tenant

    # 先同步预热嵌入与精排单例（与 backend.main lifespan 同构）。
    # 不预热就并行 gather 会让 get_instance() 撞非线程安全单例的竞态，
    # 输的那一份构建到一半落成 meta tensor → encode 时 "Cannot copy out of meta tensor"。
    from backend.core.embedding import BGEMEmbedder
    from backend.core.reranker import BGEReranker
    BGEMEmbedder.get_instance()
    BGEReranker.get_instance()

    async def _run_all():
        return await asyncio.gather(
            *(run_one(r, args.recall_k, do_answer=not args.no_answer) for r in golden))

    results = asyncio.run(_run_all())

    scores = [r["judge"] for r in results if r["judge"] is not None]
    print("\n===== 检索评估 Report (Recall@%d) =====" % args.recall_k)
    for r in results:
        mark = "✓" if r["recall"] == 1.0 else ("△" if r["recall"] > 0 else "✗")
        j = "" if r["judge"] is None else f" judge={r['judge']:.1f}"
        print(f"  {mark} {r['question'][:34]:36} | recall={r['recall']:.2f}"
              f" | top={r['top_companies']} | exp={r['expected']}{j}")
    n = len(results)
    recall_mean = sum(r["recall"] for r in results) / n if n else 0.0
    judge_mean = sum(scores) / len(scores) if scores else float("nan")
    print(f"  n={n}  Recall@{args.recall_k} 均值={recall_mean:.2f}"
          f"  judge 均值={judge_mean:.2f}" + ("" if scores else "  (未生成/无判分)"))
    return 0 if recall_mean >= 0.6 else 1


if __name__ == "__main__":
    raise SystemExit(main())