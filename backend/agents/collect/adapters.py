# backend/agents/collect/adapters.py
# 四个数据源取数：财报 / 公告 / 新闻 / 行业。
#
# 刻意不用「适配器类 + 工厂」那套抽象：四个源就是四个普通异步函数，要真实源时
# （mixed 模式）内部切一下、失败自动回落本地 fixture，对外始终返回统一结构。
# 演示定位下，平铺函数比「抽象基类 + 子类 + 装配工厂 + 并发配对」好读得多。

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Optional

import httpx

from backend.config import get_settings
from backend.core.exceptions import NonDegradableError
from backend.core.logger import get_logger
from backend.core.research_rules import compute_reliability, compute_timeliness_weight

logger = get_logger(__name__)

# 每条/每源的上限，在采集时就截断：不截断会撑爆 checkpoint 与 DB 单行。
MAX_ITEMS_PER_SOURCE = 20
MAX_CONTENT_CHARS = 2000


class CollectExhaustedError(NonDegradableError, RuntimeError):
    """四个源全部取数失败，任务无据可依。继承 NonDegradableError → 不可降级。"""


@dataclass
class RawDataItem:
    """一条取回的数据，四个源统一出口。"""
    source_type: str
    source_name: str
    title: str
    content: str
    url: Optional[str] = None
    published_at: Optional[datetime] = None
    raw: dict = field(default_factory=dict)


def _parse_published_date(value: Any) -> Optional[datetime]:
    """把日期字符串解析成带 UTC 时区的 datetime；解析不出返回 None（不抛）。

    Tavily 给的是 RFC 1123（'Wed, 09 Sep 2026 08:00:00 GMT'）。naive 补成 UTC 必须：
    下游 compute_timeliness_weight 会拿它和「现在」相减，naive 减 aware 会抛。

    :param value: 原始日期值（可为 str 或可为空）。
    :return: 带 UTC 时区的 datetime，解析失败返回 None。
    """
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(str(value))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


async def _fixture(source_type: str, company: dict) -> list[RawDataItem]:
    """读本地 fixture JSON（默认数据源）。文件没有 = 该源无数据，返回 []，不抛。

    :param source_type: 数据源类型（financial_report/announcement/news/industry）。
    :param company: 公司 dict，读其 code 键拼 fixture 文件名。
    :return: 从 fixture 还原出的 RawDataItem 列表；文件缺失返回空列表。
    """
    path = Path(get_settings().research_fixture_dir) / f"{company['code']}_{source_type}.json"
    if not path.exists():
        return []
    return [
        RawDataItem(source_type=source_type, source_name=f"fixture:{source_type}",
                    title=e["title"], content=e["content"], url=e.get("url"),
                    published_at=(datetime.fromisoformat(e["published_at"])
                                  if e.get("published_at") else None),
                    raw=e)
        for e in json.loads(path.read_text(encoding="utf-8")).get("items", [])
    ]


# ── 真实数据源（mixed 模式才启用，失败自动回落同名 fixture）──────────

async def _tavily(company: dict, api_key: str) -> list[RawDataItem]:
    """Tavily 真实新闻。失败/取空都自动回落 fixture（不中断采集），并记日志。

    :param company: 公司 dict，读 name/code 拼查询词。
    :param api_key: Tavily API 密钥。
    :return: news 类型 RawDataItem 列表；失败或取空时回落到同名 fixture。
    """
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post("https://api.tavily.com/search", json={
                "api_key": api_key,
                "query": f"{company['name']} {company['code']} 最新消息",
                "max_results": 8, "search_depth": "basic",
                # topic 必须 news：通用检索的响应里没有 published_date、多是行情导航页
                "topic": "news",
            })
            resp.raise_for_status()
            results = resp.json().get("results", [])
    except Exception as e:                       # 新闻源不可用不该让整个采集失败
        logger.warning("collect.tavily_failed", error=str(e))
    else:
        items = [RawDataItem(source_type="news", source_name="tavily",
                             title=r.get("title", "")[:200],
                             content=r.get("content") or r.get("snippet") or "",
                             url=r.get("url"),
                             published_at=_parse_published_date(r.get("published_date")),
                             raw=r) for r in results]
        if items:
            logger.info("collect.tavily_loaded", company=company["code"], count=len(items))
            return items
    return await _fixture("news", company)


# ── 四源：每个返回 (条目, 错误)。错误只在「取数 + fixture 兜底都失败」时非空 ──
# 真实源失败已有 fixture 兜底 → 不算错误（有素材就行）；fixture 空 = 该源确实没数据。

async def fetch_financial(company) -> tuple[list, Optional[str]]:
    """财报：默认 fixture；mixed 且装了 baostock 时走真实源（失败自动回落 fixture）。

    :param company: 公司 dict（含 code/name 等键）。
    :return: (条目列表, 错误)；错误仅在取数 + fixture 兜底都失败时才非空（为 None 正常）。
    """
    if get_settings().research_data_source_mode == "mixed":
        try:
            from backend.agents.collect.baostock_adapter import fetch_financial_real
            return await fetch_financial_real(company), None
        except ImportError:
            logger.warning("collect.baostock_not_installed, fall back to fixture")
    try:
        return await _fixture("financial_report", company), None
    except Exception as e:
        return [], str(e)


async def fetch_announcement(company) -> tuple[list, Optional[str]]:
    """公告：只读 fixture（真实源要爬 GPL 页面，不划算）。

    :param company: 公司 dict（含 code 键）。
    :return: (条目列表, 错误)；错误仅在取数失败时才非空。
    """
    try:
        return await _fixture("announcement", company), None
    except Exception as e:
        return [], str(e)


async def fetch_news(company) -> tuple[list, Optional[str]]:
    """新闻：默认 fixture；mixed 且配了 key 时先试 Tavily。

    :param company: 公司 dict（含 code/name 键）。
    :return: (条目列表, 错误)；错误仅在取数 + 兜底都失败时才非空。
    """
    settings = get_settings()
    try:
        if settings.research_data_source_mode == "mixed" and settings.tavily_api_key:
            return await _tavily(company, settings.tavily_api_key), None
        return await _fixture("news", company), None
    except Exception as e:
        return [], str(e)


async def fetch_industry(company) -> tuple[list, Optional[str]]:
    """行业景气：只读 fixture（纯付费数据，接回来不划算）。

    :param company: 公司 dict（含 code 键）。
    :return: (条目列表, 错误)；错误仅在取数失败时才非空。
    """
    try:
        return await _fixture("industry", company), None
    except Exception as e:
        return [], str(e)


async def fetch_all_sources(company: dict):
    """并发取四源，返回 (全部条目, 各源统计, 失败明细)。

    每个源是独立函数、内部已兜底不抛；这里只用源名跟其结果一一汇总。

    :param company: 公司 dict（含 code/name 等键，透传给各 fetch_*）。
    :return: (raw_items, source_stats, errors) 三元组 —— raw_items 为全部取回条目列表，
        source_stats 为 {source_type: {"ok", "count"}}，errors 为失败明细列表。
    """
    results = await asyncio.gather(
        fetch_financial(company), fetch_announcement(company),
        fetch_news(company), fetch_industry(company),
    )
    source_types = ("financial_report", "announcement", "news", "industry")

    raw_items, source_stats, errors = [], {}, []
    for (items, error), source_type in zip(results, source_types):
        source_stats[source_type] = {"ok": not bool(error), "count": len(items)}
        raw_items.extend(items)
        if error:
            errors.append({"source_type": source_type, "error": error})
    return raw_items, source_stats, errors


def normalize_and_score(items: list[RawDataItem],
                        now: Optional[datetime] = None) -> list[dict]:
    """把取回的条目统一成可直接入库的 dict：算权重、截正文、按源限量。

    时效权重在这里算而非查询时算：它记录「采集这一刻这份数据有多新」，是审计快照；
    查询时算会让同一份研报的历史可比性随查看时刻漂移。

    :param items: 取回的 RawDataItem 列表。
    :param now: 计算时效权重的基准时刻；默认 None 时取当前 UTC 时间。
    :return: 可直接入库的 dict 列表（含 source_type/source_name/title/content/url/
        published_at/timeliness_weight/reliability/raw）；按源限量并截断正文。
    """
    if now is None:
        now = datetime.now(timezone.utc)

    # 按来源分桶后再限每条量：不能全局限量，否则条数多的源会把其它源挤没。
    by_source: dict[str, list[RawDataItem]] = {}
    for item in items:
        by_source.setdefault(item.source_type, []).append(item)

    rows: list[dict] = []
    for source_type, group in by_source.items():
        for item in group[:MAX_ITEMS_PER_SOURCE]:
            rows.append({
                "source_type": source_type,
                "source_name": item.source_name,
                "title": item.title[:500],
                "content": item.content[:MAX_CONTENT_CHARS],
                "url": item.url,
                "published_at": item.published_at,
                "timeliness_weight": compute_timeliness_weight(item.published_at, now=now),
                "reliability": compute_reliability(source_type),
                "raw": item.raw,
            })
    return rows