# scripts/classifier/eval_classifier.py
"""训练后对关键查询做冒烟验证。期望：投研类 → specialized，闲聊/百科 → general。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.core.query_classifier import QueryClassifier


def main() -> None:
    qc = QueryClassifier.get_instance()
    cases: list[tuple[str, str]] = [
        # (query, 期望标签)
        ("泸州老窖分析", "specialized"),
        ("泸州老窖股票分析", "specialized"),
        ("舍得酒业股票分析", "specialized"),
        ("舍得酒业分析", "specialized"),
        ("贵州茅台估值", "specialized"),
        ("山西汾酒一季度业绩", "specialized"),
        ("比亚迪的估值水平", "specialized"),
        ("白酒板块值得入手吗", "specialized"),
        ("宁德时代三季报点评", "specialized"),
        # general 须保持正确
        ("你好", "general"),
        ("今天天气怎么样", "general"),
        ("推荐一部电影", "general"),
        ("怎么做红烧肉", "general"),
        ("Python是什么", "general"),
    ]
    bad = 0
    for q, want in cases:
        label, conf = qc.classify(q)
        ok = label == want
        bad += (not ok)
        print(f"{'OK ' if ok else 'XX '} [{want:>11}] got {label:<11} conf={conf:.3f}  {q}")
    print(f"\npassed {len(cases)-bad}/{len(cases)}")
    if bad:
        raise SystemExit(f"{bad} case(s) mismatched")


if __name__ == "__main__":
    main()