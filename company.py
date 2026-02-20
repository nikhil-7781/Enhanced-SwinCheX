from __future__ import annotations

from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.models.base import UUIDMixin, TimestampMixin


class Company(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "companies"

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    tan: Mapped[str] = mapped_column(String(10), unique=True, nullable=False)
    pan: Mapped[str] = mapped_column(String(10), nullable=False)
    address: Mapped[str | None] = mapped_column(Text)
    financial_year: Mapped[str | None] = mapped_column(String(7))

    users = relationship("User", back_populates="company", cascade="all, delete-orphan")
    vendors = relationship("Vendor", back_populates="company", cascade="all, delete-orphan")
    tds_payments = relationship("TDSPayment", back_populates="company", cascade="all, delete-orphan")