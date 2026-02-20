from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.middleware.auth_middleware import require_role
from app.models import Company, Form16A, TDSPayment, UserRole, Vendor
from app.schemas.common import ErrorDetail, ErrorResponse, PaginatedResponse, Pagination, SuccessResponse
from app.schemas.tds import (
    Form16AGenerateRequest,
    Form16AResponse,
    LiabilityBySectionResponse,
    SectionLiability,
    SuggestSectionRequest,
    SuggestSectionResponse,
    TDSPaymentCreateRequest,
    TDSPaymentResponse,
    TDSPaymentUpdateRequest,
    ThresholdCheckResponse,
    TDSSummaryResponse,
)
from app.services.tds_calculator import check_tds_threshold, suggest_tds_section
from app.services.pdf_generator import generate_form16a
from app.utils.dates import days_remaining, get_fiscal_period, quarter_due_date

router = APIRouter(prefix="/api/tds", tags=["tds"])


@router.post("/suggest-section", response_model=SuccessResponse[SuggestSectionResponse])
def suggest_section(payload: SuggestSectionRequest):
    result = suggest_tds_section(payload.description, payload.vendor_type)
    return SuccessResponse(data=SuggestSectionResponse(**result))


@router.get("/dashboard", response_model=SuccessResponse[TDSSummaryResponse])
def tds_dashboard(db: Session = Depends(get_db), admin=Depends(require_role(UserRole.admin))):
    company = db.query(Company).filter(Company.id == admin.company_id).first()
    period = get_fiscal_period()
    financial_year = company.financial_year if company and company.financial_year else period.financial_year
    quarter = period.quarter

    rows = (
        db.query(
            TDSPayment.tds_section,
            func.count(TDSPayment.id),
            func.coalesce(func.sum(TDSPayment.tds_amount), 0),
        )
        .filter(
            TDSPayment.company_id == admin.company_id,
            TDSPayment.financial_year == financial_year,
            TDSPayment.quarter == quarter,
        )
        .group_by(TDSPayment.tds_section)
        .all()
    )

    by_section = [
        SectionLiability(section=row[0], count=row[1], total_tds=row[2])  # type: ignore[index]
        for row in rows
    ]
    total_tds = sum((s.total_tds for s in by_section), 0)

    fy_start_year = int(financial_year.split("-")[0])
    due_date = quarter_due_date(quarter, fy_start_year)
    response = TDSSummaryResponse(
        current_quarter=quarter,
        financial_year=financial_year,
        total_tds_deducted=total_tds,
        by_section=by_section,
        challan_due_date=due_date.isoformat(),
        days_remaining=days_remaining(due_date),
    )
    return SuccessResponse(data=response)


@router.get("/liability-by-section", response_model=SuccessResponse[LiabilityBySectionResponse])
def liability_by_section(db: Session = Depends(get_db), admin=Depends(require_role(UserRole.admin))):
    company = db.query(Company).filter(Company.id == admin.company_id).first()
    period = get_fiscal_period()
    financial_year = company.financial_year if company and company.financial_year else period.financial_year

    rows = (
        db.query(
            TDSPayment.tds_section,
            func.count(TDSPayment.id),
            func.coalesce(func.sum(TDSPayment.tds_amount), 0),
        )
        .filter(TDSPayment.company_id == admin.company_id, TDSPayment.financial_year == financial_year)
        .group_by(TDSPayment.tds_section)
        .all()
    )

    by_section = [SectionLiability(section=row[0], count=row[1], total_tds=row[2]) for row in rows]
    return SuccessResponse(data=LiabilityBySectionResponse(financial_year=financial_year, by_section=by_section))


@router.post("/payments", response_model=SuccessResponse[TDSPaymentResponse])
def create_payment(
    payload: TDSPaymentCreateRequest, db: Session = Depends(get_db), admin=Depends(require_role(UserRole.admin))
):
    vendor = db.query(Vendor).filter(Vendor.id == payload.vendor_id, Vendor.company_id == admin.company_id).first()
    if not vendor:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Vendor not found")

    company = db.query(Company).filter(Company.id == admin.company_id).first()
    financial_year = payload.financial_year or (company.financial_year if company else None)

    payment = TDSPayment(
        company_id=admin.company_id,
        vendor_id=payload.vendor_id,
        payment_date=payload.payment_date,
        description=payload.description,
        gross_amount=payload.gross_amount,
        tds_section=payload.tds_section,
        tds_rate=payload.tds_rate,
        tds_amount=payload.tds_amount,
        net_amount=payload.net_amount,
        quarter=payload.quarter,
        financial_year=financial_year,
        created_by=admin.id,
    )
    db.add(payment)
    db.commit()
    db.refresh(payment)

    return SuccessResponse(data=_payment_to_response(payment))


@router.get("/payments", response_model=PaginatedResponse[TDSPaymentResponse])
def list_payments(
    vendor_id: str | None = None,
    quarter: str | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    admin=Depends(require_role(UserRole.admin)),
):
    query = db.query(TDSPayment).filter(TDSPayment.company_id == admin.company_id)
    if vendor_id:
        query = query.filter(TDSPayment.vendor_id == vendor_id)
    if quarter:
        query = query.filter(TDSPayment.quarter == quarter)
    if date_from:
        query = query.filter(TDSPayment.payment_date >= date_from)
    if date_to:
        query = query.filter(TDSPayment.payment_date <= date_to)

    total_items = query.count()
    payments = (
        query.order_by(TDSPayment.payment_date.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    total_pages = max(1, (total_items + page_size - 1) // page_size)
    return PaginatedResponse(
        data=[_payment_to_response(p) for p in payments],
        pagination=Pagination(page=page, page_size=page_size, total_pages=total_pages, total_items=total_items),
    )


@router.get("/payments/{payment_id}", response_model=SuccessResponse[TDSPaymentResponse])
def get_payment(payment_id: str, db: Session = Depends(get_db), admin=Depends(require_role(UserRole.admin))):
    payment = _get_payment_or_404(db, payment_id, admin.company_id)
    return SuccessResponse(data=_payment_to_response(payment))


@router.put("/payments/{payment_id}", response_model=SuccessResponse[TDSPaymentResponse])
def update_payment(
    payment_id: str,
    payload: TDSPaymentUpdateRequest,
    db: Session = Depends(get_db),
    admin=Depends(require_role(UserRole.admin)),
):
    payment = _get_payment_or_404(db, payment_id, admin.company_id)

    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(payment, field, value)

    db.commit()
    db.refresh(payment)
    return SuccessResponse(data=_payment_to_response(payment))


@router.delete("/payments/{payment_id}", response_model=SuccessResponse[dict])
def delete_payment(payment_id: str, db: Session = Depends(get_db), admin=Depends(require_role(UserRole.admin))):
    payment = _get_payment_or_404(db, payment_id, admin.company_id)
    db.delete(payment)
    db.commit()
    return SuccessResponse(data={"deleted": True})


@router.get("/payments/{payment_id}/threshold", response_model=SuccessResponse[ThresholdCheckResponse])
def check_threshold(
    payment_id: str, db: Session = Depends(get_db), admin=Depends(require_role(UserRole.admin))
):
    payment = _get_payment_or_404(db, payment_id, admin.company_id)
    if not payment.financial_year:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(error=ErrorDetail(code="FY_REQUIRED", message="Financial year required")).model_dump(),
        )
    result = check_tds_threshold(
        db,
        vendor_id=str(payment.vendor_id),
        new_amount=payment.gross_amount,
        financial_year=payment.financial_year,
        section=payment.tds_section,
    )
    return SuccessResponse(data=ThresholdCheckResponse(**result))


@router.post("/form16a/generate", response_model=SuccessResponse[list[Form16AResponse]])
def generate_form16a_batch(
    payload: Form16AGenerateRequest, db: Session = Depends(get_db), admin=Depends(require_role(UserRole.admin))
):
    company = db.query(Company).filter(Company.id == admin.company_id).first()
    if not company:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Company not found")

    results: list[Form16AResponse] = []
    for vendor_id in payload.vendor_ids:
        vendor = db.query(Vendor).filter(Vendor.id == vendor_id, Vendor.company_id == admin.company_id).first()
        if not vendor:
            continue

        payments = (
            db.query(TDSPayment)
            .filter(
                TDSPayment.company_id == admin.company_id,
                TDSPayment.vendor_id == vendor_id,
                TDSPayment.financial_year == payload.financial_year,
                TDSPayment.quarter == payload.quarter,
            )
            .all()
        )
        if not payments:
            continue

        total_payment = sum((p.gross_amount for p in payments), 0)
        total_tds = sum((p.tds_amount for p in payments), 0)
        sections = {p.tds_section for p in payments}
        section = sections.pop() if len(sections) == 1 else "MULTI"

        output_path = (
            f"{settings.upload_dir}/form16a/{admin.company_id}/{vendor_id}-{payload.quarter}-{payload.financial_year}.pdf"
        )
        generate_form16a(
            output_path=output_path,
            company_tan=company.tan,
            vendor_pan=vendor.pan or "UNKNOWN",
            quarter=payload.quarter,
            total_payment=float(total_payment),
            tds_amount=float(total_tds),
            section=section,
        )

        record = Form16A(
            company_id=admin.company_id,
            vendor_id=vendor_id,
            quarter=payload.quarter,
            financial_year=payload.financial_year,
            section=section,
            total_payment=total_payment,
            tds_amount=total_tds,
            file_path=output_path,
        )
        db.add(record)
        db.commit()
        db.refresh(record)

        results.append(
            Form16AResponse(
                id=str(record.id),
                vendor_id=vendor_id,
                quarter=record.quarter,
                financial_year=record.financial_year,
                section=record.section,
                total_payment=record.total_payment,
                tds_amount=record.tds_amount,
            )
        )

    return SuccessResponse(data=results)


@router.get("/form16a/{form16a_id}/download")
def download_form16a(form16a_id: str, db: Session = Depends(get_db), admin=Depends(require_role(UserRole.admin))):
    record = (
        db.query(Form16A)
        .filter(Form16A.id == form16a_id, Form16A.company_id == admin.company_id)
        .first()
    )
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Form16A not found")
    return FileResponse(record.file_path, media_type="application/pdf", filename=f"form16a-{form16a_id}.pdf")


def _get_payment_or_404(db: Session, payment_id: str, company_id: str) -> TDSPayment:
    payment = db.query(TDSPayment).filter(TDSPayment.id == payment_id, TDSPayment.company_id == company_id).first()
    if not payment:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Payment not found")
    return payment


def _payment_to_response(payment: TDSPayment) -> TDSPaymentResponse:
    return TDSPaymentResponse(
        id=str(payment.id),
        company_id=str(payment.company_id),
        vendor_id=str(payment.vendor_id),
        payment_date=payment.payment_date,
        description=payment.description,
        gross_amount=payment.gross_amount,
        tds_section=payment.tds_section,
        tds_rate=payment.tds_rate,
        tds_amount=payment.tds_amount,
        net_amount=payment.net_amount,
        quarter=payment.quarter,
        financial_year=payment.financial_year,
    )
