# backend/core/query_classifier.py
# QA Query 二分类器：把模糊 query 判成 general / specialized（三层分类的 L2）。
#
# 单一职责：对一条 query 做二分类，决定它该进「通用答」还是下钻给 L3 LLM 细分策略。
# 模型加载抽成懒加载单例（get_query_classifier）——首次调用才载入 ~90MB 权重，
# 避免 metadata/检索等无关路径被拖上模型。上游消费方是 agents/qa/nodes.py 的
# classify_query_node，只调用 classify(text)，不碰训练基建。

import json
import os
import random
from pathlib import Path
from typing import Optional

import torch

from backend.config import get_settings
from backend.core.logger import get_logger

backend_path = os.path.dirname(os.path.dirname(__file__))
logger = get_logger(__name__)

LABEL2ID = {"general": 0, "specialized": 1}
ID2LABEL = {0: "general", 1: "specialized"}

# general 侧置信阈值偏高（0.85）：专业问题被误判成通用问题的代价更高 ——
# LLM 会用自身知识回答，可能与研报内容矛盾；宁可多走一次 RAG，不放过研报相关问题。
GENERAL_CONFIDENCE_THRESHOLD = 0.85


class QueryClassifier:
    """
    QA Query 二分类器：general / specialized（微调 all-MiniLM-L6-v2）。

    训练阶段：
        qc = QueryClassifier()
        qc.train("backend/training_data.jsonl", output_dir="models/classifier")

    推理阶段：
        qc = QueryClassifier("models/classifier")     # 显式加载该模型
        qc = QueryClassifier()                         # 默认加载微调模型
        label, conf = qc.classify("什么是 Spring IOC？")
    """

    _instance: Optional["QueryClassifier"] = None

    def __init__(self, model_path: Optional[str] = None):
        """加载分类器模型并构造推理 pipeline。

        :param model_path: 模型加载路径；传入路径时加载该路径的模型（基座或任意微调
            结果），None（默认）时加载微调好的模型（finetuned_classifier_path）
        :return: 无返回值
        """
        settings = get_settings()
        model_id = model_path if model_path else os.path.join(
            backend_path, settings.finetuned_classifier_path)

        device = 0 if torch.cuda.is_available() else -1
        from transformers import pipeline as hf_pipeline
        self._pipeline = hf_pipeline(task="text-classification",
                                     model=model_id,
                                     device=device,
                                     top_k=None,
                                     truncation=True,
                                     max_length=128)
        logger.info("query_classifier.loaded", model_id=model_id)

    @classmethod
    def get_instance(cls) -> "QueryClassifier":
        """获取单例（首次调用时懒加载）。

        :return: QueryClassifier 进程内单例实例
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── 训练 ─────────────────────────────────────────────────

    def train(
        self,
        data_path: str,
        output_dir: str,
        epochs: int = 8,
        batch_size: int = 64,
        lr: float = 2e-5,
        max_length: int = 128,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> None:
        """
        微调当前加载的基座模型，训练完自动保存到 output_dir。

        :param data_path: 训练数据路径（JSONL，每行 {"text": ..., "label": "general/specialized"}）
        :param output_dir: 微调模型保存目录（训练完可直接用此路径初始化新实例）
        :param epochs: 训练轮数，默认 8
        :param batch_size: 训练批大小，默认 64
        :param lr: 学习率，默认 2e-5
        :param max_length: Token 最大长度，默认 128（Query 分类用不到长文本）
        :param val_ratio: 验证集比例，默认 0.1
        :param test_ratio: 测试集比例，默认 0.1
        :param seed: 随机种子，默认 42
        :return: 无返回值
        """
        # 训练库仅在此方法内 import，推理路径不加载这些依赖
        import numpy as np
        from datasets import Dataset
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
            set_seed,
        )

        set_seed(seed)

        # ── 加载数据 ──────────────────────────────────────────
        rows = self._load_jsonl(data_path)
        train_rows, val_rows, test_rows = self._stratified_split(
            rows, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)
        logger.info("query_classifier.split",
                    train=len(train_rows), val=len(val_rows), test=len(test_rows))

        # ── Tokenizer + Model ─────────────────────────────────
        # 从当前 pipeline 取 model_id，保持一致
        model_id = self._pipeline.model.config._name_or_path
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=2,
            label2id=LABEL2ID,
            id2label=ID2LABEL,
            ignore_mismatched_sizes=True,   # 分类头从2类随机初始化
        )

        # ── 数据集 ────────────────────────────────────────────
        def to_dataset(rows_: list[dict]) -> Dataset:
            """把样本行列表转成 HuggingFace Dataset。

            :param rows_: 每项含 text/label 的样本行列表
            :return: 含 text/label 两列的 Dataset
            """
            return Dataset.from_dict({
                "text":  [r["text"]              for r in rows_],
                "label": [LABEL2ID[r["label"]]   for r in rows_],
            })

        def tokenize(batch):
            """对一条批量文本做分词。

            :param batch: 含 text 字段的批量样本字典
            :return: 分词结果（含 input_ids 等）
            """
            return tokenizer(batch["text"], truncation=True, max_length=max_length)

        train_ds = to_dataset(train_rows).map(tokenize, batched=True, remove_columns=["text"])
        val_ds   = to_dataset(val_rows).map(tokenize,   batched=True, remove_columns=["text"])
        test_ds  = to_dataset(test_rows).map(tokenize,  batched=True, remove_columns=["text"])

        # ── 评估指标 ──────────────────────────────────────────
        def compute_metrics(eval_pred):
            """训练评估指标：accuracy 与 macro F1。

            :param eval_pred: 模型预测的 (logits, labels) 二元组
            :return: {"accuracy": float, "f1_macro": float} 指标字典
            """
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            acc = accuracy_score(labels, preds)
            _, _, f1, _ = precision_recall_fscore_support(
                labels, preds, average="macro", zero_division=0)
            return {"accuracy": float(acc), "f1_macro": float(f1)}

        # ── TrainingArguments ─────────────────────────────────
        use_cuda = torch.cuda.is_available()
        checkpoint_dir = str(Path(output_dir) / "_checkpoints")

        train_args = TrainingArguments(
            output_dir=checkpoint_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,     # 训练结束自动加载最优 checkpoint
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            learning_rate=lr,
            warmup_ratio=0.1,
            weight_decay=0.01,
            save_total_limit=1,
            logging_steps=20,
            fp16=use_cuda,
            report_to="none",
            seed=seed,
        )

        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        # ── 训练 + 测试集评估 ──────────────────────────────────
        trainer.train()
        test_metrics = trainer.evaluate(test_ds)
        logger.info("query_classifier.test_metrics", **{
            k: round(v, 4) for k, v in test_metrics.items() if k.startswith("eval_")})

        # ── 保存 ──────────────────────────────────────────────
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info("query_classifier.saved", output_dir=output_dir)

    # ── 推理 ─────────────────────────────────────────────────

    def classify(self, text: str) -> tuple[str, float]:
        """
        对 query 做 general / specialized 二分类。

        :param text: 待分类的 query 文本
        :return: 二元组 (label, confidence)。label 是 "general" 或 "specialized"；
            confidence 是对应标签的置信度 [0, 1]。
            规则：P(general) >= 0.85 → ("general", P(general))；
            否则 ("specialized", 1-P(general))。兜底：标签名不匹配时保守返回
            ("specialized", 0.5)
        """
        raw_outputs: list[dict] = self._pipeline(text)[0]

        # 找 general 标签的分数（兼容大小写和 LABEL_0 格式）
        general_score: Optional[float] = None
        for item in raw_outputs:
            lbl = item["label"].lower()
            if lbl in ("general", "label_0"):
                general_score = item["score"]
                break

        if general_score is None:
            logger.warning("query_classifier.unexpected_labels",
                           labels=[x["label"] for x in raw_outputs])
            return "specialized", 0.5

        if general_score >= GENERAL_CONFIDENCE_THRESHOLD:
            label, confidence = "general", general_score
        else:
            label, confidence = "specialized", 1.0 - general_score

        logger.info("query_classifier.result",
                    text_preview=text[:50], label=label,
                    confidence=round(confidence, 4))
        return label, confidence

    # ── 私有工具方法 ─────────────────────────────────────────

    @staticmethod
    def _load_jsonl(path: str) -> list[dict]:
        """读取并校验 JSONL 训练数据。

        :param path: JSONL 文件路径，每行 {"text": ..., "label": ...}
        :return: [{text, label}] 行列表；空行跳过，text 空或 label 非法抛 ValueError
        """
        rows: list[dict] = []
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = (obj.get("text") or "").strip()
                label = obj.get("label")
                if not text:
                    raise ValueError(f"第 {line_no} 行 text 为空")
                if label not in LABEL2ID:
                    raise ValueError(f"第 {line_no} 行 label 非法: {label!r}")
                rows.append({"text": text, "label": label})
        if not rows:
            raise ValueError(f"训练数据为空：{path}")
        return rows

    @staticmethod
    def _stratified_split(rows: list[dict], val_ratio: float, test_ratio: float,
                          seed: int) -> tuple[list[dict], list[dict], list[dict]]:
        """按标签分层切分，保证训练/验证/测试集的类别比例一致。

        :param rows: 待切分的样本行列表
        :param val_ratio: 验证集比例
        :param test_ratio: 测试集比例
        :param seed: 随机种子
        :return: 三元组 (train_rows, val_rows, test_rows)，各类别在各集内比例保持一致
        """
        random.seed(seed)

        buckets: dict[str, list[dict]] = {}
        for row in rows:
            buckets.setdefault(row["label"], []).append(row)

        train_rows, val_rows, test_rows = [], [], []
        for label, group in buckets.items():
            random.shuffle(group)
            n = len(group)
            n_test = max(1, int(n * test_ratio))
            n_val = max(1, int(n * val_ratio))
            test_rows.extend(group[:n_test])
            val_rows.extend(group[n_test:n_test + n_val])
            train_rows.extend(group[n_test + n_val:])
        random.shuffle(train_rows)
        return train_rows, val_rows, test_rows


# ── 模块级便捷函数：agents 内 `from …query_classifier import get_query_classifier` 直接调 ──
def get_query_classifier() -> QueryClassifier:
    """QueryClassifier.get_instance 的省键入（懒加载单例）。

    :return: QueryClassifier 单例实例
    """
    return QueryClassifier.get_instance()