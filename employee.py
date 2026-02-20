from __future__ import annotations

from decimal import Decimal
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.middleware.auth_middleware import get_current_user, require_role
from app.models import EmployeeDeclaration, EmployeeSalary, UserRole
from app.schemas.common import ErrorDetail, ErrorResponse, SuccessResponse
from app.schemas.employee import (
    DeclarationResponse,
    DeclarationUpdateRequest,
    RegimeComparisonResponse,
    SalaryMonth,
    SalarySummaryResponse,
)
from app.services.pdf_generator import generate_form16
from app.services.regime_comparison import calculate_regime_comparison
from app.utils.dates import fy_month_range, get_fiscal_period

router = APIRouter(prefix="/api/employee", tags=["employee"])


@router.get("/salary-summary", response_model=SuccessResponse[SalarySummaryResponse])
def salary_summary(db: Session = Depends(get_db), user=Depends(require_role(UserRole.employee))):
    period = get_fiscal_period()
    start, end = fy_month_range(period.financial_year)

    salaries = (
        db.query(EmployeeSalary)
        .filter(EmployeeSalary.user_id == user.id, EmployeeSalary.month_year >= start, EmployeeSalary.month_year <= end)
        .order_by(EmployeeSalary.month_year.asc())
        .all()
    )

    monthly_data = [
        SalaryMonth(month=s.month_year, gross=s.gross_salary, tds=s.tds_deducted, net=s.net_salary) for s in salaries
    ]

    ytd_gross = sum((s.gross_salary for s in salaries), Decimal("0"))
    ytd_tds = sum((s.tds_deducted for s in salaries), Decimal("0"))

    months_count = len(salaries)
    projected_annual_income = (ytd_gross / months_count * 12) if months_count else Decimal("0")
    projected_annual_tds = (ytd_tds / months_count * 12) if months_count else Decimal("0")

    response = SalarySummaryResponse(
        current_fy=period.financial_year,
        ytd_gross_salary=ytd_gross,
        ytd_tds_deducted=ytd_tds,
        projected_annual_income=projected_annual_income,
        projected_annual_tds=projected_annual_tds,
        monthly_data=monthly_data,
    )
    return SuccessResponse(data=response)


@router.get("/monthly-breakdown", response_model=SuccessResponse[list[SalaryMonth]])
def monthly_breakdown(db: Session = Depends(get_db), user=Depends(require_role(UserRole.employee))):
    period = get_fiscal_period()
    start, end = fy_month_range(period.financial_year)
    salaries = (
        db.query(EmployeeSalary)
        .filter(EmployeeSalary.user_id == user.id, EmployeeSalary.month_year >= start, EmployeeSalary.month_year <= end)
        .order_by(EmployeeSalary.month_year.asc())
        .all()
    )
    data = [SalaryMonth(month=s.month_year, gross=s.gross_salary, tds=s.tds_deducted, net=s.net_salary) for s in salaries]
    return SuccessResponse(data=data)


@router.get("/declarations", response_model=SuccessResponse[DeclarationResponse])
def get_declarations(db: Session = Depends(get_db), user=Depends(require_role(UserRole.employee))):
    period = get_fiscal_period()
    declaration = (
        db.query(EmployeeDeclaration)
        .filter(EmployeeDeclaration.user_id == user.id, EmployeeDeclaration.financial_year == period.financial_year)
        .first()
    )
    if not declaration:
        declaration = EmployeeDeclaration(
            user_id=user.id,
            financial_year=period.financial_year,
            section_80c=Decimal("0"),
            section_80d=Decimal("0"),
            hra_claimed=Decimal("0"),
            home_loan_interest=Decimal("0"),
        )
    response = DeclarationResponse(
        financial_year=declaration.financial_year,
        section_80c=declaration.section_80c,
        section_80d=declaration.section_80d,
        hra_claimed=declaration.hra_claimed,
        home_loan_interest=declaration.home_loan_interest,
    )
    return SuccessResponse(data=response)


@router.put("/declarations", response_model=SuccessResponse[DeclarationResponse])
def update_declarations(
    payload: DeclarationUpdateRequest, db: Session = Depends(get_db), user=Depends(require_role(UserRole.employee))
):
    if payload.section_80c > Decimal("150000"):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(error=ErrorDetail(code="LIMIT_80C", message="Section 80C max is 150000")).model_dump(),
        )
    if payload.section_80d > Decimal("50000"):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(error=ErrorDetail(code="LIMIT_80D", message="Section 80D max is 50000")).model_dump(),
        )
    if payload.home_loan_interest > Decimal("200000"):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(error=ErrorDetail(code="LIMIT_HOME_LOAN", message="Home loan interest max is 200000")).model_dump(),
        )

    declaration = (
        db.query(EmployeeDeclaration)
        .filter(EmployeeDeclaration.user_id == user.id, EmployeeDeclaration.financial_year == payload.financial_year)
        .first()
    )
    if not declaration:
        declaration = EmployeeDeclaration(user_id=user.id, financial_year=payload.financial_year)
        db.add(declaration)

    declaration.section_80c = payload.section_80c
    declaration.section_80d = payload.section_80d
    declaration.hra_claimed = payload.hra_claimed
    declaration.home_loan_interest = payload.home_loan_interest

    db.commit()
    db.refresh(declaration)

    response = DeclarationResponse(
        financial_year=declaration.financial_year,
        section_80c=declaration.section_80c,
        section_80d=declaration.section_80d,
        hra_claimed=declaration.hra_claimed,
        home_loan_interest=declaration.home_loan_interest,
    )
    return SuccessResponse(data=response)


@router.get("/regime-comparison", response_model=SuccessResponse[RegimeComparisonResponse])
def regime_comparison(db: Session = Depends(get_db), user=Depends(require_role(UserRole.employee))):
    period = get_fiscal_period()
    start, end = fy_month_range(period.financial_year)
    salaries = (
        db.query(EmployeeSalary)
        .filter(EmployeeSalary.user_id == user.id, EmployeeSalary.month_year >= start, EmployeeSalary.month_year <= end)
        .all()
    )

    ytd_gross = sum((s.gross_salary for s in salaries), Decimal("0"))
    months_count = len(salaries)
    projected_annual_income = (ytd_gross / months_count * 12) if months_count else Decimal("0")

    declaration = (
        db.query(EmployeeDeclaration)
        .filter(EmployeeDeclaration.user_id == user.id, EmployeeDeclaration.financial_year == period.financial_year)
        .first()
    )
    section_80c = declaration.section_80c if declaration else Decimal("0")
    section_80d = declaration.section_80d if declaration else Decimal("0")
    hra_claimed = declaration.hra_claimed if declaration else Decimal("0")

    result = calculate_regime_comparison(
        annual_income=projected_annual_income,
        section_80c=section_80c,
        section_80d=section_80d,
        hra_claimed=hra_claimed,
    )
    return SuccessResponse(data=RegimeComparisonResponse(**result))


@router.get("/form16/{financial_year}")
def download_form16(financial_year: str, db: Session = Depends(get_db), user=Depends(require_role(UserRole.employee))):
    start, end = fy_month_range(financial_year)
    salaries = (
        db.query(EmployeeSalary)
        .filter(EmployeeSalary.user_id == user.id, EmployeeSalary.month_year >= start, EmployeeSalary.month_year <= end)
        .all()
    )
    if not salaries:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No salary data found")

    total_gross = sum((s.gross_salary for s in salaries), Decimal("0"))
    total_tds = sum((s.tds_deducted for s in salaries), Decimal("0"))

    output_path = f"{settings.upload_dir}/form16/{user.id}-{financial_year}.pdf"
    generate_form16(
        output_path=output_path,
        employee_name=user.full_name,
        financial_year=financial_year,
        total_gross=float(total_gross),
        total_tds=float(total_tds),
    )
    return FileResponse(output_path, media_type="application/pdf", filename=f"form16-{financial_year}.pdf")