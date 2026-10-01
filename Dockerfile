# ResearchAgent 后端镜像 —— 不含模型权重（运行时从 HuggingFace 按需下载到 /models 挂载卷）
# 构建：docker build -t researchagent-backend:latest .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# torch / pymupdf / sentence-transformers 等均发布 manylinux wheel，无需 gcc 编译；
# ca-certificates 供模型下载 TLS、libgomp1 供 OpenMP 运行时。
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# 先拷贝依赖声明，利用 Docker 层缓存：requirements 不变则不复装
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 业务代码
COPY backend ./backend
COPY scripts ./scripts

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]