from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, ForeignKey, Numeric, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.models.base import UUIDMixin


class EmployeeDeclaration(Base, UUIDMixin):
    __tablename__ = "employee_declarations"

    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    financial_year: Mapped[str] = mapped_column(String(7), nullable=False)
    section_80c: Mapped[Decimal] = mapped_column(Numeric(12, 2), default=0)
    section_80d: Mapped[Decimal] = mapped_column(Numeric(12, 2), default=0)
    hra_claimed: Mapped[Decimal] = mapped_column(Numeric(12, 2), default=0)
    home_loan_interest: Mapped[Decimal] = mapped_column(Numeric(12, 2), default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    user = relationship("User", back_populates="declarations")
