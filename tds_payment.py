from __future__ import annotations

from datetime import date
from decimal import Decimal

from sqlalchemy import Date, ForeignKey, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.models.base import UUIDMixin, TimestampMixin


class TDSPayment(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "tds_payments"

    company_id: Mapped[str] = mapped_column(String(36), ForeignKey("companies.id"), nullable=False, index=True)
    vendor_id: Mapped[str] = mapped_column(String(36), ForeignKey("vendors.id"), nullable=False, index=True)
    payment_date: Mapped[date] = mapped_column(Date, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    gross_amount: Mapped[Decimal] = mapped_column(Numeric(15, 2), nullable=False)
    tds_section: Mapped[str] = mapped_column(String(10), nullable=False)
    tds_rate: Mapped[Decimal] = mapped_column(Numeric(5, 2), nullable=False)
    tds_amount: Mapped[Decimal] = mapped_column(Numeric(15, 2), nullable=False)
    net_amount: Mapped[Decimal] = mapped_column(Numeric(15, 2), nullable=False)
    quarter: Mapped[str | None] = mapped_column(String(5))
    financial_year: Mapped[str | None] = mapped_column(String(7))
    created_by: Mapped[str | None] = mapped_column(String(36), ForeignKey("users.id"))

    company = relationship("Company", back_populates="tds_payments")
    vendor = relationship("Vendor", back_populates="tds_payments")
