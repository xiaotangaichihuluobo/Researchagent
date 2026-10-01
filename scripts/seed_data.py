# scripts/seed_data.py
# 执行：python scripts/seed_data.py
# 用途：灌入投研域本地开发数据 —— 三个角色的测试账号 + 两个研究标的

import asyncio
import uuid
import os
import asyncpg                       # PostgreSQL 异步驱动（脚本直接用它，简单直接）
from passlib.context import CryptContext
from backend.config import get_settings
# 兼容性补丁：同 auth.py，让 passlib 能读到 bcrypt 版本
import bcrypt as _b, types as _t
if not hasattr(_b, "__about__"):
    _b.__about__ = _t.SimpleNamespace(__version__=getattr(_b, "__version__", "4.x"))

s = get_settings()
# 用环境变量拼出 asyncpg 的连接串（注意 asyncpg 用的是 postgresql:// 而非 +asyncpg）
DB_DSN = (
    f"postgresql://{s.db_user}:{s.db_password}"
    f"@{s.db_host}:{s.db_port}"
    f"/{s.db_name}"
)

# 只打印不含密码的部分：原样打印 DSN 会把数据库密码写进终端历史与日志文件
print(f'DB_DSN: postgresql://{s.db_user}@{s.db_host}:{s.db_port}/{s.db_name}')
# print(DB_DSN)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
TENANT_ID = "tenant_default"


async def seed_users():
    """灌入 3 个测试账号（覆盖投研域三个角色，已存在则跳过）。"""
    conn = await asyncpg.connect(DB_DSN)             # 连接数据库
    print("✅ 数据库连接成功，开始灌入测试账号...")
    try:
        users = [
            {"username": "admin",      "email": "admin@research.local",      "pwd": "Admin@123456",    "role": "admin"},
            {"username": "researcher", "email": "researcher@research.local", "pwd": "Research@123456", "role": "researcher"},
            {"username": "risk",       "email": "risk@research.local",       "pwd": "Risk@123456",     "role": "risk_control"},
        ]
        for u in users:
            await conn.execute(
                """
                INSERT INTO users (id, tenant_id, username, email, password_hash, role)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (tenant_id, email) DO NOTHING
                """,
                str(uuid.uuid4()), TENANT_ID, u["username"], u["email"],
                pwd_context.hash(u["pwd"]),          # 存哈希，绝不存明文
                u["role"],
            )
        print(f"✅ 测试账号灌入完成（{len(users)} 个，已存在则跳过）：")
        print("   admin@research.local      / Admin@123456      (管理员)")
        print("   researcher@research.local / Research@123456   (研究员)")
        print("   risk@research.local       / Risk@123456       (风控)")
    finally:
        await conn.close()                           # 无论成败都关闭连接


async def seed_companies():
    """灌入演示数据集研究标的（12 家）。

    目的：扩充演示覆盖面 —— 深数据公司 9 家（每家 4 源各 4-6 条 fixture，四维能喂出评分与估值、
    且已在 Milvus 语料库备自家历史与行业 peer 研报，检索段有「历史对比点」）；
    浅数据公司 3 家（每家 4 源仅 1-2 条，部分维度落 insufficient → 评级观察，演示降级路径）。

    fixtures 见 backend/agents/collect/fixtures/（按 <code>_<source>.json 读取，缺文件=该源空）。
    601318.SH 刻意不纳入：tests/conftest 把它当「无数据演示」标的，语义要留给测试。
    """
    conn = await asyncpg.connect(DB_DSN)
    print("✅ 开始灌入研究标的...")
    try:
        companies = [
            # ── 深数据（9 家）────────────────────────────────────────────
            {"code": "600519.SH", "name": "贵州茅台",  "industry": "白酒"},       # 已有 fixture（标杆）
            {"code": "300750.SZ", "name": "宁德时代",  "industry": "动力电池"},   # 已有 fixture
            {"code": "000858.SZ", "name": "五粮液",    "industry": "白酒"},
            {"code": "600036.SH", "name": "招商银行",  "industry": "银行"},
            {"code": "600900.SH", "name": "长江电力",  "industry": "电力"},
            {"code": "000333.SZ", "name": "美的集团",  "industry": "家用电器"},
            {"code": "600276.SH", "name": "恒瑞医药",  "industry": "化学制药"},
            {"code": "002594.SZ", "name": "比亚迪",    "industry": "新能源汽车"},
            {"code": "600887.SH", "name": "伊利股份",  "industry": "食品饮料"},
            # ── 浅数据（3 家，演示维度不足→观察降级路径）───────────────────
            {"code": "601899.SH", "name": "紫金矿业",  "industry": "有色金属"},
            {"code": "601012.SH", "name": "隆基绿能",  "industry": "光伏"},
            {"code": "002714.SZ", "name": "牧原股份",  "industry": "养殖"},
        ]
        for c in companies:
            await conn.execute(
                """
                INSERT INTO companies (id, tenant_id, code, name, industry)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (tenant_id, code) DO NOTHING
                """,
                str(uuid.uuid4()), TENANT_ID, c["code"], c["name"], c["industry"],
            )
        print(f"✅ 研究标的灌入完成（{len(companies)} 个，已存在则跳过）：")
        print("   深数据 9 家：600519 贵州茅台 / 300750 宁德时代 / 000858 五粮液 / 600036 招商银行")
        print("             / 600900 长江电力 / 000333 美的集团 / 600276 恒瑞医药 / 002594 比亚迪")
        print("             / 600887 伊利股份")
        print("   浅数据 3 家：601899 紫金矿业 / 601012 隆基绿能 / 002714 牧原股份（演示观察降级）")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(seed_users())
    asyncio.run(seed_companies())