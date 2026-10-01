# backend/api/router.py
# API 路由总入口

from fastapi import APIRouter
from backend.api.v1 import auth, companies, research, qa   # 投研域 + 轨道B 问答

api_router = APIRouter()                              # 总路由

# 把每个子 router 带前缀 + 标签聚合进来
api_router.include_router(auth.router,      prefix="/auth",      tags=["认证"])
api_router.include_router(companies.router, prefix="/companies", tags=["研究标的"])
api_router.include_router(research.router,  prefix="/research",  tags=["研究任务"])
api_router.include_router(qa.router,        prefix="/qa",        tags=["问已发布研报"])
