# scripts/classifier/train_classifier.py
"""用上方 build_training_data.py 产出的 JSONL 微调 Query 二分类器。

过程：QueryClassifier.train() 从基座 all-MiniLM-L6-v2 加载，随机初始化 2 类分类头，
用投研增强数据微调 8 epochs（early stopping, 按 f1_macro 选最优），产物覆盖
config.finetuned_classifier_path（= backend/models/classifier/query-classifier-finetuned）。

usage: python scripts/classifier/train_classifier.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# 脚本可被从任意 cwd 调用：确保仓库根在 import 路径上
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.core.query_classifier import QueryClassifier
from backend.config import get_settings
from backend.core.logger import get_logger

logger = get_logger("train_classifier")

DATA = ROOT / "scripts" / "classifier" / "training_data.jsonl"


def main() -> None:
    settings = get_settings()
    out = (ROOT / "backend" / settings.finetuned_classifier_path.strip("./")).resolve()
    logger.info("train.start", data=DATA, output=out)

    qc = QueryClassifier()          # 加载基座，分类头随机初始化
    qc.train(
        data_path=str(DATA),
        output_dir=str(out),
        epochs=8,
        batch_size=64,
        lr=2e-5,
        max_length=128,
        val_ratio=0.1,
        test_ratio=0.1,
    )
    logger.info("train.done", output=out)


if __name__ == "__main__":
    main()