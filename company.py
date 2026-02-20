from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.middleware.auth_middleware import require_role
from app.models import Company, UserRole
from app.schemas.common import ErrorDetail, ErrorResponse, SuccessResponse
from app.schemas.company import CompanyResponse, CompanyUpdateRequest
from app.utils.validators import validate_pan

router = APIRouter(prefix="/api/company", tags=["company"])


@router.get("/profile", response_model=SuccessResponse[CompanyResponse])
def get_company_profile(db: Session = Depends(get_db), admin=Depends(require_role(UserRole.admin))):
    company = db.query(Company).filter(Company.id == admin.company_id).first()
    if not company:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Company not found")
    return SuccessResponse(data=_company_to_response(company))


@router.put("/profile", response_model=SuccessResponse[CompanyResponse])
def update_company_profile(
    payload: CompanyUpdateRequest, db: Session = Depends(get_db), admin=Depends(require_role(UserRole.admin))
):
    company = db.query(Company).filter(Company.id == admin.company_id).first()
    if not company:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Company not found")

    if payload.pan and not validate_pan(payload.pan.upper()):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(error=ErrorDetail(code="INVALID_PAN", message="PAN format is invalid")).model_dump(),
        )

    for field, value in payload.model_dump(exclude_unset=True).items():
        if value is None:
            continue
        setattr(company, field, value.upper() if field in {"pan", "tan"} else value)

    db.commit()
    db.refresh(company)
    return SuccessResponse(data=_company_to_response(company))


def _company_to_response(company: Company) -> CompanyResponse:
    return CompanyResponse(
        id=str(company.id),
        name=company.name,
        tan=company.tan,
        pan=company.pan,
        address=company.address,
        financial_year=company.financial_year,
    )