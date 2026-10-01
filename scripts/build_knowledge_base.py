# scripts/build_knowledge_base.py
# 研报语料导入：把后台直接上传的 .md / .pdf 文档切成可检索切片，写入 report_corpus。
#
# 旧知识库（knowledge_domain / build_pipeline / DocumentChunk）已移除，
# 本脚本只保留研报这条链路。切块复用下面三件套：
#   load_document        —— 统一加载（PDF 用 PyPDFLoader / Markdown 用 TextLoader）
#   split_markdown_documents —— MarkdownHeaderTextSplitter（按标题分节）+ MarkdownTextSplitter
#   split_pdf_documents  —— 过滤空页 + RecursiveCharacterTextSplitter（按页切）
# 切好后收成切片文本，走 backend.core.report_ingest.ingest_report_chunks 的
# 「嵌入 + report_corpus 写入 + PG 登记」底座。
#
# 本脚本是库模块（无 CLI）：研报导入请调用 build_report_pipeline(file_path, ...)。

import sys
from pathlib import Path
from datetime import datetime

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
)

# 把项目根目录加入 Python 路径，使得本脚本能 import backend.*
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core.report_ingest import ingest_report_chunks  # noqa: E402

# ── 模块级分块器单例 ──────────────────────────────────────────
_MD_HEADER_SPLITTER = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#",   "H1"),
        ("##",  "H2"),
        ("###", "H3"),
        ("####", "H4"),
    ],
    strip_headers=False,
)

_CHAR_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=100,
    separators=["\n\n", "\n", "。", "，", " ", ""],
)


# ── 文档加载 ──────────────────────────────────────────────────

def load_document(file_path: str) -> list[Document]:
    """统一文档加载入口，根据扩展名选择 Loader（.pdf / .md / .markdown）。"""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{file_path}")
    ext = path.suffix.lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        print(f"  [PDF] 加载完成：{len(pages)} 页 ← {path.name}")
        return pages
    elif ext in (".md", ".markdown"):
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
        print(f"  [MD]  加载完成：{len(docs[0].page_content)} 字符 ← {path.name}")
        return docs
    else:
        raise ValueError(
            f"不支持的文件类型：{ext}\n"
            f"当前支持：.pdf / .md / .markdown\n"
            f"提示：可用 markitdown 将 Word/PPT 转换为 .md 后再导入"
        )


# ── PDF 分块 ──────────────────────────────────────────────────

def split_pdf_documents(pages: list[Document]) -> list[Document]:
    """PDF 文档分块：过滤空页 + RecursiveCharacterTextSplitter。"""
    non_empty_pages = [p for p in pages if len(p.page_content.strip()) > 20]
    skipped = len(pages) - len(non_empty_pages)
    if skipped > 0:
        print(f"  过滤空页：{skipped} 页（图片/扫描件页）")

    chunks = _CHAR_SPLITTER.split_documents(non_empty_pages)

    for chunk in chunks:
        filename = Path(chunk.metadata.get("source", "未知文件")).stem
        page_num = chunk.metadata.get("page", 0) + 1
        chunk.metadata["source_name"] = f"{filename} 第{page_num}页"

    print(f"  [PDF] 分块完成：{len(non_empty_pages)} 页 → {len(chunks)} 个 chunk")
    return chunks


# ── Markdown 分块 ─────────────────────────────────────────────

def split_markdown_documents(
    docs: list[Document],
    chunk_size: int = 512,      # 代码类内容默认 1200，纯文字可调低到 600~800
    chunk_overlap: int = 100,
) -> list[Document]:
    """Markdown 文档分块：MarkdownHeaderTextSplitter + MarkdownTextSplitter 两阶段。"""
    splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    header_chunks: list[Document] = []
    for doc in docs:
        sections = _MD_HEADER_SPLITTER.split_text(doc.page_content)
        source_path = doc.metadata.get("source", "")
        for section in sections:
            section.metadata["source"] = source_path
        header_chunks.extend(sections)

    final_chunks = splitter.split_documents(header_chunks)

    for chunk in final_chunks:
        source_path = chunk.metadata.get("source", "")
        filename    = Path(source_path).stem if source_path else "未知文件"
        parts = [
            chunk.metadata.get("H1", ""),
            chunk.metadata.get("H2", ""),
            chunk.metadata.get("H3", ""),
            chunk.metadata.get("H4", ""),
        ]
        parts = [p for p in parts if p]
        chunk.metadata["source_name"] = (
            f"{filename} > {' > '.join(parts)}" if parts else filename
        )

    print(f"  [MD]  分块完成：{len(docs)} 个文件 → {len(final_chunks)} 个 chunk")
    return final_chunks


# ── 统一分块入口 ──────────────────────────────────────────────

def split_documents(docs: list[Document], file_path: str) -> list[Document]:
    """统一分块入口，根据文件类型自动选择分块策略。"""
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return split_pdf_documents(docs)
    elif ext in (".md", ".markdown"):
        return split_markdown_documents(docs)
    else:
        raise ValueError(f"不支持的文件类型：{ext}")


# ── 研报语料流水线（后台直接导入 .md/.pdf → report_corpus）────────────────

REPORT_TYPE = "research"   # 与 admin 后台「导入研报」一致；可按需改为 "summary" 等


async def build_report_pipeline(
    file_path: str,
    *,
    tenant_id: str = "tenant_default",
    company_code: str = "",
    industry: str = "",
    report_type: str = REPORT_TYPE,
    published_at: datetime | None = None,
    title: str | None = None,
) -> int:
    """
    把一个 .md/.pdf 研报文档导入 report_corpus 的完整流水线。

      Step 1  读取文档（load_document：PyPDFLoader / TextLoader）
      Step 2  智能分块（split_documents：Markdown 按标题 / PDF 按页+recursive）
      Step 3  收成切片文本，复用 ingest_report_chunks 嵌入 + 写 report_corpus + PG 登记

    返回切片数。report_key 用文件名 stem（幂等：同名重导入覆盖同主键）。
    """
    print(f"\n{'='*55}")
    print(f" 研报语料导入")
    print(f" 文件      ：{file_path}")
    print(f" 公司       ：{company_code or '（未填）'}")
    print(f" 行业       ：{industry or '（未填）'}")
    print(f" 报告类型   ：{report_type}")
    print(f"{'='*55}\n")

    # Step 1：读取
    print("📖 Step 1/3  读取文档…")
    docs = load_document(file_path)

    # Step 2：分块（md/pdf 智能切块）
    print("\n✂️  Step 2/3  智能分块…")
    chunks = split_documents(docs, file_path)
    texts = [c.page_content for c in chunks]

    # Step 3：嵌入 + 写 report_corpus + PG 登记
    print("\n🔢 Step 3/3  嵌入并写入 report_corpus…")
    report_key = Path(file_path).stem
    if not title:
        title = Path(file_path).stem
    if published_at is None:
        published_at = datetime.now()
    count = await ingest_report_chunks(
        tenant_id=tenant_id,
        report_key=report_key,
        title=title,
        chunks=texts,
        company_code=company_code,
        industry=industry,
        report_type=report_type,
        published_at=published_at,
    )

    print(f"\n🎉 完成！共写 {count} 个切片 → report_corpus")
    print(f"   report_key = {report_key}")
    print("   ⚠️  更新此研报时请保留 report_key（同名重导入即覆盖）")
    return count


# 无 CLI 入口：`build_report_pipeline` 是供上层导入 .md/.pdf 研报的库函数
# （样例文件 samples/sample2.md 已移除，原 `--report` 样例行入口一并删掉）。
if __name__ == '__main__':
    print("本脚本是库模块，无独立 CLI。研报导入请调用 build_report_pipeline(file_path, ...)")