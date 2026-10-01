# backend/core/llm_factory.py
# LLM Factory：统一封装大模型调用，按 Agent 类型路由。所有 Agent 必须经此模块取模型。
#
# 运输层用官方 openai SDK（AsyncOpenAI）打 DeepSeek/Qwen 的 OpenAI 兼容接口：
#   · 文本输出   → await handle.ainvoke(prompt) 回收 content 字符串
#   · 流式输出   → async for chunk in handle.astream(prompt) 逐 token 正文
#   · 结构化输出 → get_structured_llm 绑 schema；ainvoke 回收 Schema 实例
# 重试/降级不在这里 —— 那是 retry.py 的事（with_retry 包住 handle.ainvoke）。这个模块只
# 负责「一次调用 + schema 校验」，失败一律抛 LLMAPIError（可重试）或 AuthenticationError
# （不可重试），供 with_retry 按 agent_type 处置。

from typing import AsyncIterator, Callable, Type, Optional

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

# Agent 类型 → (供应商, 模型名) 的路由表。想给某类业务【换模型 / 换供应商】，
# 只改这里一行（供应商必须在 _VENDOR_ENDPOINTS 登记过端点）。
# 键与 retry.py 的 fallback_map、各子图的 with_retry 参数三处必须一致。
_AGENT_MODEL_ROUTING: dict[str, tuple[str, str]] = {
    "collect":   ("qwen", "qwen-turbo"),   # ① 数据采集：结构化抽取来源信息
    "analyze":   ("qwen", "qwen-turbo"),   # ② 多维分析：基本面/技术面/舆情面/行业面评分
    "retrieve":  ("qwen", "qwen-turbo"),   # ③ 研报 RAG：横向对比点提炼
    "valuation": ("qwen", "qwen-turbo"),   # ④ 估值建模：结构化输出估值区间
    "risk":      ("qwen", "qwen-turbo"),   # ⑤ 风控复核：合规预检
    "qa_reports": ("qwen", "qwen-turbo"),  # ⑥ 轨道 B：问已发布研报（分类/改写/生成共用）
}

# 供应商端点注册表：vendor_key → 从 settings 取 (base_url, api_key) 的工厂。
# 2026-09 已切到 通义千问 qwen-turbo：百炼兼容接口实测支持 json_schema(strict)，
# 结构化输出不再被网关 400 挡掉（DeepSeek 网关 json_schema 不识别）。
# 直连某个供应商 / 换 key，只改这里 + _AGENT_MODEL_ROUTING，不动闸门逻辑。
_VENDOR_ENDPOINTS: dict[str, Callable[[], tuple[str, str]]] = {
    "qwen":     lambda: (get_settings().qwen_base_url.rstrip("/"),
                         get_settings().qwen_api_key),
    "deepseek": lambda: (get_settings().deepseek_base_url.rstrip("/"),
                         get_settings().deepseek_api_key),
}

# 传输客户端按 (base_url, api_key) 分键缓存 —— 同一网关多个 model 名共享一个
# 客户端（复用连接），换供应商 / 换 key 自动落新客户端；不再有「全局唯一客户端」。
#   · max_retries=0：SDK 自带重试，但本项目的重试唯一权威是 retry.py 的 with_retry ——
#     关掉 SDK 内置重试，避免与业务重试时序叠出一堆重复请求。
#   · trust_env=False：绕 Windows 系统代理。否则请求经代理 TLS 握手失败。
#   · 超时：总 120s、建连 15s，一次调用最坏 2 分钟够用。
_client_cache: dict[tuple[str, str], AsyncOpenAI] = {}


def _async_client(vendor: str) -> AsyncOpenAI:
    """按 (base_url, api_key) 取共享客户端；未建则懒加载一个入缓存。

    :param vendor: 供应商键名，必须在 _VENDOR_ENDPOINTS 注册过，否则抛 ValueError
    :return: 对应 (base_url, api_key) 的 AsyncOpenAI 客户端单例
    """
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


def _strict_json_schema(model: Type[BaseModel]) -> dict:
    """把 pydantic 模型转成 OpenAI「strict 模式」兼容的 json_schema。

    作用：结构化输出不再靠 prompt 里描述字段 —— schema 序列化直接发给模型，作为
    唯一出处（改输出契约只改 pydantic 类）。strict 模式对 schema 形状有硬约束，这里
    负责把这些约束铲平：
      - 无 `$ref`/`$defs`（内联掉）；无 `anyOf`/`oneOf`
      - 每个 `type: object` 节点都要 `additionalProperties: false`
      - `required` 覆盖全部 properties（**含带默认值的** —— strict 不允许可选字段）
      - 剔除 `title` / `default` / `$schema` / `$id`

    当前四个结构化 schema（DimensionAnalysis / ValuationDecision / ComparisonExtraction /
    WordingCompliance）都是扁平对象，本函数递归也覆盖嵌套对象与 $defs，以便日后加字段。

    :param model: 目标输出结构的 pydantic BaseModel 子类
    :return: 满足 OpenAI strict 模式约束的 json_schema 字典
    """
    base = model.model_json_schema()
    defs = base.pop("$defs", {})

    def _resolve(node: dict) -> dict:
        """递归展开节点里的 $ref 引用。

        :param node: 待展开的 schema 节点
        :return: 展开 $ref 后的节点字典
        """
        if "$ref" in node:
            name = node["$ref"].split("/")[-1]
            return _resolve(defs[name])
        return node

    def _walk(node):
        """递归清理 schema：剔除 title/default/$schema 等，处理 object 节点。

        :param node: 任意 schema 值（dict/list/标量）
        :return: 清理并内联后的值；object 节点补全 required 与 additionalProperties
        """
        if isinstance(node, list):
            return [_walk(v) if isinstance(v, (dict, list)) else v for v in node]
        if not isinstance(node, dict):
            return node
        node = _resolve(node)
        out: dict = {}
        for key, val in node.items():
            if key in ("title", "default", "$schema", "$id", "$defs"):
                continue
            out[key] = _walk(val) if isinstance(val, (dict, list)) else val
        if node.get("type") == "object":
            props = node.get("properties") or {}
            out["required"] = sorted(props.keys())          # 全部必填（含默认值字段）
            out["additionalProperties"] = False
        return out

    return _walk(base)


# 供应商已切到 通义千问（百炼）：key / endpoint 从 qwen 配置取，换供应商只改
# config 三行 + llm_factory 的 _AGENT_MODEL_ROUTING/_VENDOR_ENDPOINTS。结构化模型
# Qwen 实测支持 json_schema(strict)。
#
# 结构化响应格式：schema 序列化直发模型作唯一出处；不用 llm 自己猜层结构
# （那段由 model_validate_json 在出参兜底重试）。
def _response_format(schema: Type[BaseModel]) -> dict:
    """组装 OpenAI 的 json_schema 响应格式参数。

    :param schema: 目标输出结构的 pydantic BaseModel 子类
    :return: 内含 name/description/strict/schema 的 response_format 字典
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name":        schema.__name__,
            "description": schema.__doc__ or "",
            "strict":      True,
            "schema":      _strict_json_schema(schema),
        },
    }


class LLMFactory:
    """统一封装大模型调用，按 Agent 类型路由（运输层用官方 AsyncOpenAI）。

    对外只暴露 ainvoke / astream：
      ainvoke(prompt)  → 文本时回收 content 字符串；绑了 schema 时回收 Schema 实例
      astream(prompt)  → 逐 token 产出正文（SSE 问答出口用）
    get_llm 返回的是按 (模型, 温度, 是否流式) 缓存的单例句柄；结构化是
    get_structured_llm 在该实例上绑一个 schema（缓存键刻意不含 schema：schema 多，不各留一份）。
    """

    # 缓存键 = (模型, 温度, 是否流式)，刻意【不含 schema】：schema 多，不各留一份。
    _handle_cache: dict[tuple[str, float, bool], "LLMFactory"] = {}

    def __init__(self, agent_type: str, temperature: float, streaming: bool):
        """构造一个按 Agent 类型路由的模型句柄。

        :param agent_type: Agent 类型，须在 _AGENT_MODEL_ROUTING 中注册，否则抛 ValueError
        :param temperature: 采样温度
        :param streaming: 是否启用流式输出
        :return: 无返回值
        """
        if agent_type not in _AGENT_MODEL_ROUTING:
            raise ValueError(
                f"未知 agent_type: '{agent_type}'，可用类型：{list(_AGENT_MODEL_ROUTING.keys())}")
        vendor, model = _AGENT_MODEL_ROUTING[agent_type]
        self.agent_type = agent_type
        self.vendor = vendor                       # 走哪个供应商端点（日志/排查看）
        self.model = model
        self.temperature = temperature
        self.streaming = streaming
        self._schema: Type[BaseModel] | None = None
        self._client = _async_client(vendor)

    # ── 真实内核：一次调用 → content 字符串 ────────────────────────────────
    # 只承载「一次调用」，成败都返回/抛出，不做重试 —— 交给 with_retry。
    async def _complete(self, prompt: str) -> str:
        """一次非流式调用，取出 content；异常按 retry.py 的分类上抛。

        :param prompt: 发送给模型的用户提示词
        :return: 模型返回的 content 字符串（可能为空串）
        """
        kwargs: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "stream": False,
            # 显式关思考，保证投研结构化输出稳定（default 开思考会带回推理段落）。
            "extra_body": {"thinking": {"type": "disabled"}},
        }
        if self._schema is not None:
            kwargs["response_format"] = _response_format(self._schema)

        # 预算闸门：研报任务累计 token 超限时在此抛 TaskBudgetExceeded，不发请求。
        # 检查在发请求前，用的是已落库的历史累计 —— 超限最多超一个调用。
        await usage_ctx.enforce_budget()

        _t0 = time.perf_counter()
        try:
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
        return resp.choices[0].message.content or ""

    async def _stream(self, prompt: str) -> AsyncIterator[str]:
        """流式调用：逐 token 产出正文。

        usage 观测：开 stream_options.include_usage，网关会在最后一个 chunk 带上 usage；
        若网关不理会该参数（usage 恒 None），tokens 落 NULL、不阻塞 —— 见 usage_ctx。

        :param prompt: 发送给模型的用户提示词
        :return: 异步迭代器，逐个产出正文 token 字符串
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

    # ── 对外调用面：文本 / 结构化 / 流式 ──────────────────────────────────────

    async def ainvoke(self, prompt: str):
        """回收 content 字符串；若本实例绑了 schema，则回收 Schema 实例。

        :param prompt: 发送给模型的用户提示词
        :return: 未绑 schema 时返回内容字符串；绑了 schema 时返回由该 schema
            解析出的 pydantic 实例
        """
        content = await self._complete(prompt)
        if self._schema is not None:
            return self._schema.model_validate_json(content)
        return content

    async def astream(self, prompt: str) -> AsyncIterator[str]:
        """流式调用：逐 token 产出正文（SSE 问答出口用）。

        :param prompt: 发送给模型的用户提示词
        :return: 异步迭代器，逐个产出正文 token 字符串
        """
        async for chunk in self._stream(prompt):
            yield chunk

    # ── 取模型入口：缓存单例 + 结构化（绑 schema）────────────────────────────

    @classmethod
    def get_llm(cls, agent_type: str, temperature: float = 0,
                streaming: bool = False) -> "LLMFactory":
        """按 Agent 类型取文本模型句柄（缓存单例，同配置复用同一个）。

        :param agent_type: Agent 类型，用于路由模型
        :param temperature: 采样温度，默认 0；同配置复用同一实例
        :param streaming: 是否流式输出，默认 False；作为缓存键组成部分
        :return: 文本模型句柄（LLMFactory 实例），同配置返回同一个缓存实例
        """
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

        每调新建一个（非缓存）实例、共享底层配置、只多绑一个 schema —— 语义同旧版
        with_structured_output。缓存的是文本单例，schema 不进缓存键。

        :param agent_type: Agent 类型，用于路由模型
        :param output_schema: 目标输出结构的 pydantic BaseModel 子类
        :param temperature: 采样温度，默认 0；透传给底层句柄
        :return: 绑定该 schema 的 LLMFactory 实例（非缓存、每次新建）
        """
        base = cls.get_llm(agent_type, temperature=temperature)   # streaming 固定 False
        structured = cls(base.agent_type, base.temperature, streaming=False)
        structured._schema = output_schema
        return structured

    @classmethod
    def clear_cache(cls) -> None:
        """清空句柄缓存与传输客户端缓存（测试/关停时用）。

        :return: 无返回值
        """
        cls._handle_cache.clear()
        _client_cache.clear()
        logger.info("llm_factory.cache_cleared")


# ── 模块级便捷函数：Agent 内 `from …llm_factory import get_llm` 直接调 ──
def get_llm(agent_type: str, temperature: float = 0, streaming: bool = False) -> LLMFactory:
    """LLMFactory.get_llm 的省键入。

    :param agent_type: Agent 类型，用于路由模型
    :param temperature: 采样温度，默认 0
    :param streaming: 是否流式输出，默认 False
    :return: 文本模型句柄（LLMFactory 实例）
    """
    return LLMFactory.get_llm(agent_type, temperature=temperature, streaming=streaming)


def get_structured_llm(agent_type: str, output_schema: Type[BaseModel],
                       temperature: float = 0) -> LLMFactory:
    """LLMFactory.get_structured_llm 的省键入。

    :param agent_type: Agent 类型，用于路由模型
    :param output_schema: 目标输出结构的 pydantic BaseModel 子类
    :param temperature: 采样温度，默认 0；透传给底层句柄
    :return: 绑定该 schema 的 LLMFactory 实例
    """
    return LLMFactory.get_structured_llm(agent_type, output_schema, temperature=temperature)


# ── 演示脚本：python -m backend.core.llm_factory ──────────────────
# 本模块常被问「BASE_MODE 进去 / get_structured_llm 进去，最终请求到底长什么样」。
# 这里用拦截 create 的方式，把两种句柄【真实会发给网关的 kwargs】完整打出来对比，
# 不真打网络：create 被替换成假实现，只记录 body、返回一个合法 JSON 让 ainvoke 走完。
# 为什么在 core 内演示凸定义 schema：本文件属 core，铁律是 core 不 import agents，
# 而真实的结构化 schema 都躺在 agents/ 下 —— 脚本在本地定义一个等效小模型即可说明数据流。
if __name__ == "__main__":
    import sys

    # 控制台 Windows gbk，打印中文 schema 前先把标准输出切成 utf-8，避免 UnicodeEncodeError。
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from pydantic import BaseModel, Field

    class AnswerOut(BaseModel):
        """对问题的结论。"""
        company: str = Field(description="公司/标的名称")
        revenue: float | None = Field(default=None, description="营收（亿元），未知则缺省")
        trend: str = Field(default="", description="趋势一句话")

    import json

    captured: dict = {}

    class _Choices:
        def __init__(self, content: str):
            """构造一个伪造的 choices 项。

            :param content: 充当模型返回正文的字符串
            :return: 无返回值
            """
            self.message = type("_M", (), {"content": content})()

    class _FakeCompletions:                       # 记下 body + 返回合法 JSON，不真发
        async def create(self, **kwargs):
            """伪造 create：只记录请求体并返回合法响应，不真发网络。

            :param kwargs: 真实会发给网关的请求参数
            :return: 含单一合法 JSON choices 的伪响应对象
            """
            captured.update(kwargs)
            return type("_R", (), {"choices": [_Choices(json.dumps(
                {"company": "贵州茅台", "revenue": 1234, "trend": "稳健上行"}))]})()

    class _FakeChat:
        completions = _FakeCompletions()

    async def main():
        """演示主流程：对比有无 schema 时真正发给网关的请求体。

        :return: 无返回值
        """
        # ① BASE_MODE：get_llm 无 schema —— 只有 messages/温度/stream，没有 response_format
        wrap = get_llm("qa_reports", temperature=0)
        wrap._client = type("_C", (), {"chat": _FakeChat()})()
        await wrap.ainvoke("贵州茅台 2025 年营收如何")
        print("── [1] get_llm（无 schema）真实 kwargs ──")
        print(json.dumps(captured, ensure_ascii=False, indent=2))

        # ② get_structured_llm 绑 schema —— 多出 response_format，st=> 返回 AnswerOut 实例
        captured.clear()
        wrap = get_structured_llm("qa_reports", AnswerOut)
        wrap._client = type("_C", (), {"chat": _FakeChat()})()
        out = await wrap.ainvoke("贵州茅台 2025 年营收如何")
        print("\n── [2] get_structured_llm（绑 schema）真实 kwargs ──")
        print(json.dumps(captured, ensure_ascii=False, indent=2))
        print("\n── [2] ainvoke 回收到的实例 ──")
        print(f"类型={type(out).__name__}  company={out.company}  revenue={out.revenue}  trend={out.trend!r}")

    import asyncio
    asyncio.run(main())