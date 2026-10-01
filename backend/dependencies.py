# backend/dependencies.py（数据库部分）
from typing import AsyncGenerator
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials  # 解析 Authorization: Bearer 头
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from jose import JWTError, jwt                       # python-jose：JWT 的编解码库
import os,sys
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_root)
from backend.config import get_settings
from backend.core.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)

# ── 创建异步引擎（连接池）──
engine = create_async_engine(
    settings.database_url,        # 来自 config.py，最终来自 .env.local
    pool_size=10,                 # 连接池基础大小
    max_overflow=20,              # 高峰时最多再额外开 20 个连接
    echo=False,                   # True 会打印所有 SQL，调试时可临时打开
)

# ── 会话工厂 ──
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI 依赖：获取异步数据库会话，自动提交 / 回滚 / 关闭。

    :return: 异步生成器 AsyncSession；接口正常结束自动 commit，异常自动 rollback 后重抛。
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session            # 把会话交给接口使用（回顾 2.1.6 / 2.5）
            await session.commit()   # 接口正常结束 → 自动提交
        except Exception:
            await session.rollback() # 出错 → 自动回滚
            raise

# ── JWT 鉴权 ───────────────────────────────────────────────────
bearer_scheme = HTTPBearer()   # FastAPI 安全方案：自动从请求头解析 "Authorization: Bearer <token>"


async def get_current_user(
    # Depends(bearer_scheme)：FastAPI 自动取出 Bearer Token；没带或格式错会直接 401
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> dict:
    """FastAPI 依赖：验证 JWT Token，返回当前用户信息。
    返回 {"user_id": str, "role": str, "tenant_id": str}；
    sub、role 与 tenant_id 都是必需声明，缺任一个即视为无效凭证并抛 401。

    :param credentials: HTTP Bearer 凭证（由 Depends(bearer_scheme) 从 Authorization 头解析）；缺省或格式错直接 401。
    :return: dict，含 user_id/role/tenant_id；Token 无效或缺少声明时抛 401。
    """
    # 预先准备好「401 凭证无效」异常，多处复用
    # print(f'credentials: {credentials}')
    # print(f'credentials.credentials: {credentials.credentials}')
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无效的认证凭证",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        # 用密钥和算法解码 Token；若签名不对/过期，会抛 JWTError
        payload = jwt.decode(
            credentials.credentials,            # 实际的 token 字符串
            settings.jwt_secret_key,            # 验签密钥（和签发时同一个）
            algorithms=[settings.jwt_algorithm],
        )
        user_id: str = payload.get("sub")                                  # 标准字段 sub = 用户ID
        role: str = payload.get("role")                                    # 角色（必需声明，不给默认值）
        tenant_id: str = payload.get("tenant_id")                          # 租户（必需声明，不给默认值）

        if not user_id or not role or not tenant_id:  # Token 里缺用户ID、角色或租户，视为无效
            raise credentials_exception

    except JWTError:                            # 解码失败（签名错/过期等）
        raise credentials_exception

    return {"user_id": user_id, "role": role, "tenant_id": tenant_id}

# ── 角色鉴权 ───────────────────────────────────────────────────
def require_role(*roles: str):
    """FastAPI 依赖工厂：要求当前用户属于指定角色之一，否则 403。

    为什么必须有这个函数：JWT 里签了 role，但如果没有任何地方用它做授权判断，
    那 role 就只是个装饰 —— 任何登录用户都能调任何接口。
    前端路由守卫只算 UX，不能当安全边界；真正的边界在这里。

    用法：
        @router.post("/x", dependencies=[Depends(require_role("admin"))])
        async def x(): ...

    :param roles: 允许访问的角色名（可变参数，如 "admin"、"researcher"）。
    :return: 依赖函数 _check；校验当前用户的 role，不在 roles 内抛 403，通过则返回当前用户 dict。
    """
    async def _check(current_user: dict = Depends(get_current_user)) -> dict:
        """角色校验依赖：注入当前用户并校验其角色。

        :param current_user: 当前用户信息 dict（Depends(get_current_user) 注入）。
        :return: dict，角色通过时原样返回 current_user。
        """
        if current_user["role"] not in roles:
            logger.warning("auth.role_denied", role=current_user["role"], required=list(roles))
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="权限不足")
        return current_user
    return _check
if __name__ == '__main__':
    ...
    # test_get_db.py（项目根目录）
    import asyncio
    from sqlalchemy import text
    async def main():
        # 模拟 FastAPI 调用依赖：迭代一次拿到 session，跑一条查询
        async for db in get_db():
            r = await db.execute(text(
                "SELECT current_database(), "
                "count(*) FROM information_schema.tables WHERE table_schema = :s"
            ), {"s": "public"})
            row = r.fetchone()
            print("get_db 连到库:", row[0])
            print("public 下表数量:", row[1])


    asyncio.run(main())
    print("get_db 测试通过 ✅")
