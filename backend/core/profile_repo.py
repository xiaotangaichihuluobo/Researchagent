# backend/core/profile_repo.py
# 轨道 B 的跨会话用户画像仓储（user_profiles 表）。与 qa_repo 的会话记忆正交：
#   会话记忆键 (tenant, user, session)、会话一关即清；画像键 (tenant, user)、跨会话、
#   存稳定偏好。
# 写入走「显式接口 / 显式信号」—— 结构化字段不读改写，不做每轮 LLM 提炼
# （低成本诉求 + 避免把流水账混进「用户是谁」，见 docs/跨会话记忆设计.md §3）。
#
# 设计约束：core 不 import agents。本模块自包含，LLM 概不参与。

import json
from typing import Optional

from sqlalchemy import text

from backend.core.logger import get_logger
from backend.dependencies import AsyncSessionLocal

logger = get_logger(__name__)


def _as_list(val) -> list:
    """JSONB 列可能以歧义回成 list 或 str，统一压成 list。"""
    if val is None:
        return []
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:                       # noqa: BLE001 —— 防御坏数据，只记不炸
            return []
    return list(val)


def _merge(cur: list, incoming: list | None) -> list:
    """合并去重：保留已有顺序，追加新项，不整表覆盖。incoming 缺席时不动旧值。"""
    if incoming is None:
        return list(cur)
    seen = set(cur)
    out = list(cur)
    for item in incoming:
        s = str(item)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


async def get_profile(tenant_id: str, user_id: str) -> Optional[dict]:
    """读一条用户画像；没有则返回 None。"""
    async with AsyncSessionLocal() as session:
        row = (await session.execute(
            text("SELECT preferred_style, interests, watchlist, facts"
                 " FROM user_profiles WHERE tenant_id = :t AND user_id = :u"),
            {"t": tenant_id, "u": user_id},
        )).fetchone()
    if not row:
        return None
    return {
        "preferred_style": row[0],
        "interests": _as_list(row[1]),
        "watchlist": _as_list(row[2]),
        "facts": _as_list(row[3]),
    }


async def upsert_profile(tenant_id: str, user_id: str, *,
                         preferred_style: Optional[str] = None,
                         interests: Optional[list] = None,
                         watchlist: Optional[list] = None) -> None:
    """幂等写入用户画像：同 (tenant, user) 只一行，ON CONFLICT 更新。

    列表字段合并去重（先读旧值 → 并新 → 整体写回），不覆盖用户在接口外攒下的关注。
    preferred_style 有值才更新（显式给的值才落盘）。
    """
    cur = await get_profile(tenant_id, user_id)
    merged_i = _merge((cur or {}).get("interests", []), interests)
    merged_w = _merge((cur or {}).get("watchlist", []), watchlist)
    style = preferred_style if preferred_style is not None else (cur or {}).get("preferred_style")

    async with AsyncSessionLocal() as session:
        await session.execute(
            text("""
                INSERT INTO user_profiles
                    (tenant_id, user_id, preferred_style, interests, watchlist)
                VALUES (:t, :u, :style, :i, :w)
                ON CONFLICT (tenant_id, user_id) DO UPDATE
                    SET preferred_style = COALESCE(EXCLUDED.preferred_style, user_profiles.preferred_style),
                        interests       = EXCLUDED.interests,
                        watchlist       = EXCLUDED.watchlist,
                        updated_at      = NOW()
            """),
            {"t": tenant_id, "u": user_id, "style": style,
             "i": json.dumps(merged_i, ensure_ascii=False),
             "w": json.dumps(merged_w, ensure_ascii=False)},
        )
        await session.commit()


def build_profile_text(profile: dict | None) -> str:
    """把画像渲染成一行的可读文本（注入 system 用）；无内容返回空串。

    只拼「有内容」的维度，返回体不含「【用户画像】」头 —— 头由 prompts 侧包。
    """
    if not profile:
        return ""
    parts: list[str] = []
    if profile.get("preferred_style"):
        parts.append(f"回答风格：{profile['preferred_style']}")
    if profile.get("interests"):
        parts.append("关注话题：" + "、".join(str(x) for x in profile["interests"]))
    if profile.get("watchlist"):
        parts.append("关注标的：" + "、".join(str(x) for x in profile["watchlist"]))
    return "；".join(parts)