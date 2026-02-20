from __future__ import annotations

import enum
from datetime import date, datetime

from sqlalchemy import Date, DateTime, Enum, ForeignKey, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.models.base import UUIDMixin


class VendorDocumentType(str, enum.Enum):
    pan_card = "pan_card"
    form_15g = "form_15g"
    form_15h = "form_15h"
    lower_deduction_certificate = "lower_deduction_certificate"


class VendorDocument(Base, UUIDMixin):
    __tablename__ = "vendor_documents"

    vendor_id: Mapped[str] = mapped_column(String(36), ForeignKey("vendors.id"), nullable=False, index=True)
    document_type: Mapped[VendorDocumentType] = mapped_column(Enum(VendorDocumentType), nullable=False)
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    expiry_date: Mapped[date | None] = mapped_column(Date)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    vendor = relationship("Vendor", back_populates="documents")
