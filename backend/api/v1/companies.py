# backend/api/v1/companies.py
# 研究标的接口

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from backend.core.logger import get_logger
from backend.core import research_repo as repo
from backend.dependencies import get_current_user, require_role

router = APIRouter()
logger = get_logger(__name__)


class CompanyCreate(BaseModel):
    """新增标的请求体。"""
    code:     str = Field(..., min_length=2, max_length=32, description="标的代码，如 600519.SH")
    name:     str = Field(..., min_length=1, max_length=128, description="标的名称")
    industry: str = Field(..., min_length=1, max_length=64,  description="所属行业")
    market:   str = Field("A股", max_length=16, description="市场")


@router.get("")
async def list_companies(
    limit:  int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    current_user: dict = Depends(get_current_user),      # 任意登录用户可读
):
    """标的列表（分页）。"""
    rows = await repo.list_companies(current_user["tenant_id"], limit=limit, offset=offset)
    return {"items": rows, "limit": limit, "offset": offset}


@router.post("", status_code=status.HTTP_201_CREATED,
             dependencies=[Depends(require_role("admin"))])   # 只有 admin 能新增标的
async def create_company(
    req: CompanyCreate,
    current_user: dict = Depends(get_current_user),
):
    """新增研究标的。"""
    company_id = await repo.create_company(
        tenant_id=current_user["tenant_id"],
        code=req.code, name=req.name, industry=req.industry, market=req.market,
    )
    await repo.write_audit_log(current_user["tenant_id"], current_user["user_id"],
                              "company.create", "company", company_id,
                              {"code": req.code, "name": req.name})
    return {"company_id": company_id}
