# backend/core/embedding.py
# BGE-M3 本地嵌入模型（进程内单例，dense + sparse 双输出）。
# 从 knowledge_base.py 拆出：嵌入模型是读/写两侧（reranker 检索、研报写入）共用的
# 通用底座，与 Milvus 客户端、研报写入编排解耦，单独成文件。

import os
from typing import Optional

backend_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from backend.config import get_settings
from backend.core.logger import get_logger


logger = get_logger(__name__)

# 本地缺模型时，从这里联网把权重下载到配置路径。与本地目录内容一致（README 为 BGE-M3）。
BGE_M3_REPO_ID = "BAAI/bge-m3"


class BGEMEmbedder:
    """
    BGE-M3 本地嵌入模型单例。

    一次推理同时输出：
      - dense 向量（1024 维浮点数组，用于语义相似度检索）
      - sparse 向量（{token_id: weight} 字典，用于关键词精确检索）

    进程内单例：首次调用 get_instance() 时加载模型（约5-15秒），
    后续调用直接返回同一实例，不重复加载。

    用法：
        embedder = BGEMEmbedder.get_instance()
        dense, sparse = embedder.encode_query("什么是 Spring IOC？")
    """

    _instance: Optional["BGEMEmbedder"] = None   # 单例持有

    def __init__(self, model_path: str):
        """加载本机 BGE-M3 模型实例（含兼容性补丁）。

        :param model_path: BGE-M3 本地模型目录路径
        :return: 无返回值
        """
        # ── 兼容性补丁：FlagEmbedding 1.3.x 依赖 transformers 内部函数 ──
        # transformers>=5.0 移除了 is_torch_fx_available，
        # 但当前锁定 transformers==4.51.0 不受影响。
        # 此补丁作为保险，避免未来升级时报 ImportError。
        import importlib.util as _ilu
        from transformers.utils import import_utils as _tf_iu
        if not hasattr(_tf_iu, "is_torch_fx_available"):
            _tf_iu.is_torch_fx_available = (
                lambda: _ilu.find_spec("torch.fx") is not None
            )

        import torch
        from FlagEmbedding import BGEM3FlagModel

        logger.info("bge_m3.loading", model_path=model_path)

        # ── fp16 仅在 CUDA 上启用，MPS（Apple M系列）不启用 ──
        # MPS 在 BGE-M3 attention 矩阵乘法上会触发 LLVM ERROR，
        # CPU 模式下用 fp32，速度稍慢但稳定。
        # _device = "cuda:0" if torch.cuda.is_available() else "cpu"
        _use_fp16 = torch.cuda.is_available()

        # ── 兼容性补丁 #2：FlagEmbedding<=1.4.2 把 torch_dtype 以裸 `dtype=`
        # 传给 AutoModel.from_pretrained（见其 finetune runner 的 get_model），
        # 而 transformers>=4.5x 已不再把 `dtype` 映射为 `torch_dtype`（只读
        # `torch_dtype`），于是裸 `dtype` 留在 model_kwargs 里转发给
        # XLMRobertaModel.__init__ → "unexpected keyword argument 'dtype'"。
        #
        # 实测证据：transformers 4.51.0 的 modeling_utils 第 1994 行只有
        # `kwargs.pop("torch_dtype", ...)`，没有任何把裸 `dtype` 转
        # `torch_dtype` 的分支 —— 崩溃正是这条路。
        #
        # 修法：只在 BGEM3FlagModel 加载期间给 AutoModel.from_pretrained 加一层
        # 薄垫片，把裸 `dtype` 翻译成它认的 `torch_dtype`，加载完立即还原。
        # 不动 transformers 的 pin（整仓构建在 4.51.0 上，降版风险更大），
        # 也不改 FlagEmbedding 的 site-packages（不可复现）。
        # 与上方 is_torch_fx_available 补丁同一模式：局部、尽力而为、留痕。
        import transformers
        _orig_from_pretrained = transformers.AutoModel.from_pretrained

        def _dtype_shim(model_name_or_path, *args, **kwargs):
            """把裸 dtype 翻译成 AutoModel 认的 torch_dtype 的垫片。

            顺带强制 low_cpu_mem_usage=False：
            FlagEmbedding 的 EncoderOnlyEmbedderRunner.get_model 调 AutoModel.
            from_pretrained 时不传 low_cpu_mem_usage，transformers 在低内存
            （本进程同模型堆 BGE-M3 2.2G 后再加载）且带了 torch_dtype 时，会走
            accelerate 的 meta 分片加载 —— 权重留在 meta 占位，编码时
            self.model.to(device) 就抛 “Cannot copy out of meta tensor”。
            这里把开关钉成 False，堵死 meta 路径。

            :param model_name_or_path: 传给 from_pretrained 的模型名称或路径
            :param args: 透传给 from_pretrained 的位置参数
            :param kwargs: 透传的关键字参数；含裸 dtype 时转成 torch_dtype
            :return: from_pretrained 的返回对象
            """
            if "dtype" in kwargs and "torch_dtype" not in kwargs:
                kwargs["torch_dtype"] = kwargs.pop("dtype")
            # 钉死 False：不信任 transformers 在低内存下的默认自动加速分片判断
            kwargs["low_cpu_mem_usage"] = False
            return _orig_from_pretrained(model_name_or_path, *args, **kwargs)

        transformers.AutoModel.from_pretrained = _dtype_shim
        try:
            self._model = BGEM3FlagModel(
                model_name_or_path=model_path,
                use_fp16=_use_fp16,
                # device=_device,
            )
        finally:
            transformers.AutoModel.from_pretrained = _orig_from_pretrained
        # 防御：加载完扫一遍，确认没有残留 meta 张量（真实权重若停在 meta，
        # encode 时 .to(device) 必炸；低内存下 accelerate 会偷偷把它落下）。
        _meta = [n for n, p in self._model.model.named_parameters() if p.is_meta]
        if _meta:
            logger.error("bge_m3.meta_params_present", count=len(_meta), sample=_meta[:3])
        logger.info("bge_m3.loaded", use_fp16=_use_fp16)

    @classmethod
    def get_instance(cls) -> "BGEMEmbedder":
        """获取单例（首次调用时加载模型，后续复用）。

        :return: BGEMEmbedder 进程内单例实例
        """
        if cls._instance is None:
            bge_m3_model_path = os.path.join(backend_path, get_settings().bge_m3_model_path)
            # 本地缺模型或目录不完整 → 从 HF 仓库把权重下载到【配置路径】，不落 ~/.cache
            if not (
                os.path.exists(bge_m3_model_path)
                and os.path.isdir(bge_m3_model_path)
                and os.path.isfile(os.path.join(bge_m3_model_path, "config.json"))
                and any(f.endswith((".bin", ".safetensors")) for f in os.listdir(bge_m3_model_path))
            ):
                from huggingface_hub import snapshot_download
                logger.info("bge_m3.downloading", repo_id=BGE_M3_REPO_ID, local_dir=bge_m3_model_path)
                snapshot_download(BGE_M3_REPO_ID, local_dir=str(bge_m3_model_path))
            cls._instance = BGEMEmbedder(bge_m3_model_path)
        return cls._instance

    def encode(
        self,
        texts: list[str],
        batch_size: int = 12,
    ) -> tuple[list[list[float]], list[dict]]:
        """
        批量编码文本，同时返回 dense 和 sparse 两种向量。

        :param texts: 待编码的文本列表
        :param batch_size: 单次推理批大小，越大速度越快但显存占用越多；
            12 是 16GB 显存 / 统一内存下的经验值
        :return: 二元组 (dense_vecs, sparse_vecs)。
            dense_vecs 是 list of 1024-dim float 向量，每项对应 texts[i]；
            sparse_vecs 是 list of {token_id: weight} 字典，每项对应 texts[i]
        """
        output = self._model.encode(
            texts,
            batch_size=batch_size,
            max_length=8192,            # BGE-M3 支持最长 8192 token，覆盖大多数 chunk
            return_dense=True,          # 输出稠密语义向量
            return_sparse=True,         # 输出稀疏关键词向量
            return_colbert_vecs=False,  # ColBERT 多向量表示，本项目不用
        )
        dense_vecs = output["dense_vecs"].tolist()   # numpy → Python list
        # sparse: numpy.float16 → Python float
        # 必须转换！LangGraph MemorySaver 用 msgpack 序列化 State，
        # msgpack 不支持 numpy.float16，会在运行时抛 TypeError。
        sparse_vecs = []
        for d in output["lexical_weights"]:
            vec = {}
            for k, v in d.items():
                vec[int(k)] = float(v)
            sparse_vecs.append(vec)
        return dense_vecs, sparse_vecs

    def encode_query(self, text: str) -> tuple[list[float], dict]:
        """
        编码单条查询，返回 (dense_vec, sparse_vec)。

        查询时调用此方法（而非 encode），batch_size=1 避免不必要的 padding。

        :param text: 单条查询文本
        :return: 二元组 (dense_vec, sparse_vec)。
            dense_vec 是 1024-dim float 列表，sparse_vec 是 {token_id: weight} 字典
        """
        dense_list, sparse_list = self.encode([text], batch_size=1)
        return dense_list[0], sparse_list[0]