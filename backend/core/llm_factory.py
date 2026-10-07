# backend/core/llm_factory.py
# LLM Factory：统一封装大模型调用，按 Agent 类型路由。所有 Agent 必须经此模块取模型。
#
# 运输层用官方 openai SDK（AsyncOpenAI）打 Qwen 的 OpenAI 兼容接口：
#   · 文本输出   → await handle.ainvoke(prompt) 回收 content 字符串
#   · 流式输出   → async for chunk in handle.astream(prompt) 逐 token 正文
#   · 结构化输出 → get_structured_llm 绑 Pydantic 模型；ainvoke 回收模型实例
# 结构化输出直接把 Pydantic 类交给 chat.completions.parse()（response_format），由 SDK
# 生成 strict json_schema 并本地校验，不再手写 schema。重试/降级不在这里 —— 那是
# retry.py 的事（with_retry 包住 handle.ainvoke）。这个模块只负责「一次调用 + 校验」，
# 失败一律抛 LLMAPIError（可重试）或 AuthenticationError（不可重试）。

from typing import AsyncIterator, Any, Callable, Type

import time
import httpx
from openai import (
    APIError,
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError as _OpenAIAuthError,
    BadRequestError,
)
from pydantic import BaseModel

from backend.config import get_settings
from backend.core.exceptions import AuthenticationError, LLMAPIError
from backend.core.logger import get_logger
from backend.core import usage_ctx

logger = get_logger(__name__)

# Agent 类型 → (供应商, 模型名) 的路由表。想给某类业务换模型/供应商只改这里一行
# （供应商必须在 _VENDOR_ENDPOINTS 登记过端点）。键与 retry.py 的 fallback_map、
# 各子图的 with_retry 参数三处必须一致。
_AGENT_MODEL_ROUTING: dict[str, tuple[str, str]] = {
    "collect":     ("qwen", "qwen-turbo"),   # 数据采集：结构化抽取来源信息
    "analyze":     ("qwen", "qwen-turbo"),   # 多维分析：基本面/技术面/舆情面/行业面评分
    "retrieve":    ("qwen", "qwen-turbo"),   # 研报 RAG：横向对比点提炼
    "valuation":   ("qwen", "qwen-turbo"),   # 估值建模：结构化输出估值区间
    "risk":        ("qwen", "qwen-turbo"),   # 风控复核：合规预检
    "qa_reports":  ("qwen", "qwen-turbo"),   # 轨道 B：问已发布研报（分类/改写/生成共用）
}

# 供应商端点注册表：vendor_key → 从 settings 取 (base_url, api_key) 的工厂。
# 2026-09 已切到 通义千问 qwen-turbo：百炼兼容接口实测支持 json_schema(strict)。
_VENDOR_ENDPOINTS: dict[str, Callable[[], tuple[str, str]]] = {
    "qwen":     lambda: (get_settings().qwen_base_url.rstrip("/"),
                         get_settings().qwen_api_key),
    "deepseek": lambda: (get_settings().deepseek_base_url.rstrip("/"),
                         get_settings().deepseek_api_key),
}

# 传输客户端按 (base_url, api_key) 分键缓存 —— 同一网关多个 model 名共享一个客户端。
#   · max_retries=0：SDK 自带重试，但本项目的重试唯一权威是 retry.py 的 with_retry，
#     关掉内置重试避免叠出一堆重复请求。
#   · trust_env=False：绕 Windows 系统代理，否则请求经代理 TLS 握手失败。
#   · 超时：总 120s、建连 15s，一次调用最坏 2 分钟够用。
_client_cache: dict[tuple[str, str], AsyncOpenAI] = {}


def _async_client(vendor: str) -> AsyncOpenAI:
    """按 (base_url, api_key) 取共享客户端；未建则懒加载一个入缓存。"""
    try:
        endpoint_fn = _VENDOR_ENDPOINTS[vendor]
    except KeyError:
        raise ValueError(
            f"未知供应商: '{vendor}'，可用：{list(_VENDOR_ENDPOINTS)}")
    base_url, api_key = endpoint_fn()
    key = (base_url, api_key)
    client = _client_cache.get(key)
    if client is None:
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=0,
            http_client=httpx.AsyncClient(
                trust_env=False,
                timeout=httpx.Timeout(120.0, connect=15.0),
            ),
        )
        _client_cache[key] = client
    return client


class LLMFactory:
    """统一封装大模型调用，按 Agent 类型路由（运输层用官方 AsyncOpenAI）。

    对外只暴露 ainvoke / astream：
      ainvoke(prompt)  → 文本时回收 content 字符串；绑了 schema 时回收模型实例
      astream(prompt)  → 逐 token 产出正文（SSE 问答出口用）
    get_llm 返回按 (模型, 温度, 是否流式) 缓存的单例句柄；get_structured_llm 在
    get_llm 实例上绑一个 schema（缓存键刻意不含 schema：schema 多，不各留一份）。
    """

    # 缓存键 = (模型, 温度, 是否流式)；刻意不含 schema。
    _handle_cache: dict[tuple[str, float, bool], "LLMFactory"] = {}

    def __init__(self, agent_type: str, temperature: float, streaming: bool):
        """构造一个按 Agent 类型路由的模型句柄。"""
        if agent_type not in _AGENT_MODEL_ROUTING:
            raise ValueError(
                f"未知 agent_type: '{agent_type}'，可用类型：{list(_AGENT_MODEL_ROUTING.keys())}")
        vendor, model = _AGENT_MODEL_ROUTING[agent_type]
        self.agent_type = agent_type
        self.vendor = vendor
        self.model = model
        self.temperature = temperature
        self.streaming = streaming
        self._schema: Type[BaseModel] | None = None
        self._client = _async_client(vendor)

    # ── 真实内核：一次调用 → 响应对象 ───────────────────────────────────
    # 只承载「一次调用」，成败都返回/抛出，不做重试 —— 交给 with_retry。
    async def _complete(self, prompt: str) -> Any:
        """一次非流式调用；返回 ChatCompletion（绑 schema 时为 ParsedChatCompletion）。"""
        kwargs: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            # 显式关思考：投研结构化输出稳定（qwen 默认开思考会带回推理段落）。
            "extra_body": {"thinking": {"type": "disabled"}},
        }
        if self._schema is not None:
            # 直接把 pydantic 模型交给 parse()，由 SDK 生成 strict schema 并校验。
            kwargs["response_format"] = self._schema

        # 预算闸门：研报 token 累计超限在此抛 TaskBudgetExceeded，不发请求。
        # 检查在发请求前，用的是已落库的历史累计 —— 超限最多超一个调用。
        await usage_ctx.enforce_budget()

        _t0 = time.perf_counter()
        try:
            if self._schema is not None:
                resp = await self._client.chat.completions.parse(**kwargs)
            else:
                resp = await self._client.chat.completions.create(**kwargs)
        except BadRequestError as e:                        # 400：schema/参数被网关拒
            raise LLMAPIError(f"模型返回错误请求：{e}") from e
        except _OpenAIAuthError as e:                       # 401：不可重试
            raise AuthenticationError(f"API Key 无效或已过期：{e}") from e
        except (APIConnectionError, APITimeoutError, APIError) as e:  # 网络/超时：可重试
            raise LLMAPIError(f"模型请求失败：{e}") from e

        _latency_ms = int((time.perf_counter() - _t0) * 1000)
        usage = getattr(resp, "usage", None)
        await usage_ctx.record_llm_usage(
            agent_type=self.agent_type, vendor=self.vendor, model=self.model,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
            latency_ms=_latency_ms, streaming=False,
        )
        return resp

    async def _stream(self, prompt: str) -> AsyncIterator[str]:
        """流式调用：逐 token 产出正文。

        usage 观测：开 stream_options.include_usage，网关在最后一个 chunk 带 usage；
        若网关不理会（usage 恒 None），tokens 落 NULL、不阻塞。
        """
        _t0 = time.perf_counter()
        _usage = None
        await usage_ctx.enforce_budget()      # 预算闸门（研报）—— 问答不受限，压测 no-op
        try:
            stream = await self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                stream=True,
                stream_options={"include_usage": True},
                extra_body={"thinking": {"type": "disabled"}},
            )
            async for chunk in stream:
                if getattr(chunk, "usage", None):
                    _usage = chunk.usage          # 流式 usage 只出现在最后一个 chunk
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    yield delta.content
        except _OpenAIAuthError as e:                       # 401：不可重试
            raise AuthenticationError(f"API Key 无效或已过期：{e}") from e
        except (APIConnectionError, APITimeoutError, APIError) as e:  # 网络/超时：可重试
            raise LLMAPIError(f"模型流式请求失败：{e}") from e
        finally:
            # 流结束/中断都落一条（幂等：同一次调用只记一次）。观测不抛、不回传。
            _latency_ms = int((time.perf_counter() - _t0) * 1000)
            await usage_ctx.record_llm_usage(
                agent_type=self.agent_type, vendor=self.vendor, model=self.model,
                prompt_tokens=getattr(_usage, "prompt_tokens", None),
                completion_tokens=getattr(_usage, "completion_tokens", None),
                total_tokens=getattr(_usage, "total_tokens", None),
                latency_ms=_latency_ms, streaming=True,
            )

    # ── 对外调用面：文本 / 结构化 / 流式 ──────────────────────────────────

    async def ainvoke(self, prompt: str):
        """回收 content 字符串；若本实例绑了 schema，则回收 schema 校验出的模型实例。"""
        resp = await self._complete(prompt)
        if self._schema is not None:
            parsed = resp.choices[0].message.parsed
            if parsed is None:      # 空正文 / 被拒：交重试层处置
                raise LLMAPIError("结构化输出解析为空")
            return parsed
        return resp.choices[0].message.content or ""

    async def astream(self, prompt: str) -> AsyncIterator[str]:
        """流式调用：逐 token 产出正文（SSE 问答出口用）。"""
        async for chunk in self._stream(prompt):
            yield chunk

    # ── 取模型入口：缓存单例 + 结构化（绑 schema）────────────────────────

    @classmethod
    def get_llm(cls, agent_type: str, temperature: float = 0,
                streaming: bool = False) -> "LLMFactory":
        """按 Agent 类型取文本模型句柄（缓存单例，同配置复用同一个）。"""
        _, model = _AGENT_MODEL_ROUTING[agent_type]
        key = (model, temperature, streaming)
        handle = cls._handle_cache.get(key)
        if handle is None:
            handle = cls(agent_type, temperature, streaming)
            cls._handle_cache[key] = handle
            logger.info("llm_factory.model_initialized",
                        agent_type=agent_type, vendor=handle.vendor,
                        model_key=key[0], temperature=temperature)
        return handle

    @classmethod
    def get_structured_llm(cls, agent_type: str, output_schema: Type[BaseModel],
                           temperature: float = 0) -> "LLMFactory":
        """绑定了结构化 Schema 的句柄；`await ainvoke()` 回收 output_schema 实例。

        每次调用新建一个实例（非缓存）、共享底层配置、只多绑一个 schema —— 语义同旧版
        with_structured_output。缓存的是文本单例，schema 不进缓存键。
        """
        base = cls.get_llm(agent_type, temperature=temperature)   # streaming 固定 False
        structured = cls(base.agent_type, base.temperature, streaming=False)
        structured._schema = output_schema
        return structured

    @classmethod
    def clear_cache(cls) -> None:
        """清空句柄缓存与传输客户端缓存（测试/关停时用）。"""
        cls._handle_cache.clear()
        _client_cache.clear()
        logger.info("llm_factory.cache_cleared")


# ── 模块级便捷函数：Agent 内 `from …llm_factory import get_llm` 直接调 ──
def get_llm(agent_type: str, temperature: float = 0, streaming: bool = False) -> LLMFactory:
    """LLMFactory.get_llm 的省键入。"""
    return LLMFactory.get_llm(agent_type, temperature=temperature, streaming=streaming)


def get_structured_llm(agent_type: str, output_schema: Type[BaseModel],
                       temperature: float = 0) -> LLMFactory:
    """LLMFactory.get_structured_llm 的省键入。"""
    return LLMFactory.get_structured_llm(agent_type, output_schema, temperature=temperature)