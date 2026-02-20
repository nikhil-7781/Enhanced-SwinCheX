from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.middleware.auth_middleware import require_role
from app.models import Vendor, VendorDocument, VendorDocumentType, VendorType, ResidencyStatus, UserRole
from app.schemas.common import ErrorDetail, ErrorResponse, PaginatedResponse, Pagination, SuccessResponse
from app.schemas.vendor import VendorCreateRequest, VendorDocumentResponse, VendorResponse, VendorUpdateRequest
from app.utils.file_storage import save_upload
from app.utils.validators import validate_pan

router = APIRouter(prefix="/api/vendors", tags=["vendors"])


@router.get("", response_model=PaginatedResponse[VendorResponse])
def list_vendors(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    admin=Depends(require_role(UserRole.admin)),
):
    query = db.query(Vendor).filter(Vendor.company_id == admin.company_id)
    total_items = query.count()
    vendors = (
        query.order_by(Vendor.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    total_pages = max(1, (total_items + page_size - 1) // page_size)
    return PaginatedResponse(
        data=[_vendor_to_response(v) for v in vendors],
        pagination=Pagination(page=page, page_size=page_size, total_pages=total_pages, total_items=total_items),
    )


@router.post("", response_model=SuccessResponse[VendorResponse])
def create_vendor(
    payload: VendorCreateRequest, db: Session = Depends(get_db), admin=Depends(require_role(UserRole.admin))
):
    if payload.pan and not validate_pan(payload.pan.upper()):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(error=ErrorDetail(code="INVALID_PAN", message="PAN format is invalid")).model_dump(),
        )

    vendor = Vendor(
        company_id=admin.company_id,
        name=payload.name,
        pan=payload.pan.upper() if payload.pan else None,
        vendor_type=VendorType(payload.vendor_type),
        residency_status=ResidencyStatus(payload.residency_status or "resident"),
        email=payload.email,
        phone=payload.phone,
        address=payload.address,
        documents_complete=False,
    )
    db.add(vendor)
    db.commit()
    db.refresh(vendor)
    return SuccessResponse(data=_vendor_to_response(vendor))


@router.get("/{vendor_id}", response_model=SuccessResponse[VendorResponse])
def get_vendor(vendor_id: str, db: Session = Depends(get_db), admin=Depends(require_role(UserRole.admin))):
    vendor = _get_vendor_or_404(db, vendor_id, admin.company_id)
    return SuccessResponse(data=_vendor_to_response(vendor))


@router.put("/{vendor_id}", response_model=SuccessResponse[VendorResponse])
def update_vendor(
    vendor_id: str,
    payload: VendorUpdateRequest,
    db: Session = Depends(get_db),
    admin=Depends(require_role(UserRole.admin)),
):
    vendor = _get_vendor_or_404(db, vendor_id, admin.company_id)

    if payload.pan and not validate_pan(payload.pan.upper()):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(error=ErrorDetail(code="INVALID_PAN", message="PAN format is invalid")).model_dump(),
        )

    updates = payload.model_dump(exclude_unset=True)
    if "vendor_type" in updates:
        vendor.vendor_type = VendorType(updates["vendor_type"])
        updates.pop("vendor_type")
    if "residency_status" in updates:
        vendor.residency_status = ResidencyStatus(updates["residency_status"])
        updates.pop("residency_status")
    if "pan" in updates and updates["pan"]:
        updates["pan"] = updates["pan"].upper()
    for field, value in updates.items():
        setattr(vendor, field, value)

    db.commit()
    db.refresh(vendor)
    return SuccessResponse(data=_vendor_to_response(vendor))


@router.delete("/{vendor_id}", response_model=SuccessResponse[dict])
def delete_vendor(vendor_id: str, db: Session = Depends(get_db), admin=Depends(require_role(UserRole.admin))):
    vendor = _get_vendor_or_404(db, vendor_id, admin.company_id)
    db.delete(vendor)
    db.commit()
    return SuccessResponse(data={"deleted": True})


@router.post("/{vendor_id}/documents", response_model=SuccessResponse[VendorDocumentResponse])
def upload_vendor_document(
    vendor_id: str,
    document_type: VendorDocumentType = Query(...),
    expiry_date: date | None = Query(None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    admin=Depends(require_role(UserRole.admin)),
):
    vendor = _get_vendor_or_404(db, vendor_id, admin.company_id)

    try:
        file_path = save_upload(file, settings.upload_dir, "vendor_documents")
    except ValueError as exc:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(error=ErrorDetail(code="INVALID_FILE", message=str(exc))).model_dump(),
        )

    doc = VendorDocument(
        vendor_id=vendor.id,
        document_type=document_type,
        file_path=file_path,
        expiry_date=expiry_date,
    )
    db.add(doc)

    if document_type == VendorDocumentType.pan_card:
        vendor.documents_complete = True

    db.commit()
    db.refresh(doc)
    return SuccessResponse(data=_document_to_response(doc))


@router.get("/{vendor_id}/documents", response_model=SuccessResponse[list[VendorDocumentResponse]])
def list_vendor_documents(
    vendor_id: str, db: Session = Depends(get_db), admin=Depends(require_role(UserRole.admin))
):
    vendor = _get_vendor_or_404(db, vendor_id, admin.company_id)
    docs = db.query(VendorDocument).filter(VendorDocument.vendor_id == vendor.id).all()
    return SuccessResponse(data=[_document_to_response(doc) for doc in docs])


@router.post("/bulk-upload", response_model=SuccessResponse[dict])
def bulk_upload():
    return SuccessResponse(data={"todo": "Not implemented in V1"})


def _get_vendor_or_404(db: Session, vendor_id: str, company_id: str) -> Vendor:
    vendor = db.query(Vendor).filter(Vendor.id == vendor_id, Vendor.company_id == company_id).first()
    if not vendor:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Vendor not found")
    return vendor


def _vendor_to_response(vendor: Vendor) -> VendorResponse:
    return VendorResponse(
        id=str(vendor.id),
        name=vendor.name,
        pan=vendor.pan,
        vendor_type=vendor.vendor_type.value,
        residency_status=vendor.residency_status.value,
        email=vendor.email,
        phone=vendor.phone,
        address=vendor.address,
        documents_complete=vendor.documents_complete,
    )


def _document_to_response(doc: VendorDocument) -> VendorDocumentResponse:
    return VendorDocumentResponse(
        id=str(doc.id),
        document_type=doc.document_type.value,
        file_path=doc.file_path,
        expiry_date=doc.expiry_date,
        uploaded_at=doc.uploaded_at.isoformat(),
    )