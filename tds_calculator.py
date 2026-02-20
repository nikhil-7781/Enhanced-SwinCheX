from __future__ import annotations

from decimal import Decimal
from typing import Dict

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models import TDSPayment, Vendor, VendorType


def suggest_tds_section(description: str, vendor_type: str) -> dict:
    description_lower = description.lower()

    if any(word in description_lower for word in ["rent", "lease", "rental"]):
        rate = Decimal("10.0")
        return {"section": "194I", "rate": rate, "explanation": "Rent payments"}

    if any(word in description_lower for word in ["professional", "consultant", "legal", "ca", "architect"]):
        rate = Decimal("10.0") if vendor_type == "individual" else Decimal("2.0")
        return {
            "section": "194J",
            "rate": rate,
            "explanation": "Professional/technical services",
        }

    if any(word in description_lower for word in ["contract", "labour", "construction", "repair", "maintenance"]):
        rate = Decimal("2.0") if vendor_type == "individual" else Decimal("1.0")
        return {"section": "194C", "rate": rate, "explanation": "Contract/sub-contract work"}

    if any(word in description_lower for word in ["commission", "brokerage", "agent"]):
        rate = Decimal("5.0")
        return {"section": "194H", "rate": rate, "explanation": "Commission/brokerage"}

    if any(word in description_lower for word in ["interest", "loan"]):
        rate = Decimal("10.0")
        return {"section": "194A", "rate": rate, "explanation": "Interest payments"}

    rate = Decimal("10.0") if vendor_type == "individual" else Decimal("2.0")
    return {
        "section": "194J",
        "rate": rate,
        "explanation": "General professional services (please verify)",
    }


def check_tds_threshold(
    db: Session,
    *,
    vendor_id: str,
    new_amount: Decimal,
    financial_year: str,
    section: str,
) -> dict:
    total_paid = (
        db.query(func.coalesce(func.sum(TDSPayment.gross_amount), 0))
        .filter(
            TDSPayment.vendor_id == vendor_id,
            TDSPayment.financial_year == financial_year,
            TDSPayment.tds_section == section,
        )
        .scalar()
    )
    new_total = Decimal(total_paid) + new_amount

    vendor = db.query(Vendor).filter(Vendor.id == vendor_id).first()
    vendor_type = vendor.vendor_type.value if vendor else "individual"

    threshold = _threshold_for_section(section, vendor_type)
    remaining = threshold - new_total
    exceeds = new_total >= threshold
    message = (
        f"Threshold exceeded for section {section}" if exceeds else f"Remaining before threshold: {remaining}"
    )

    return {
        "threshold": threshold,
        "new_total": new_total,
        "exceeds": exceeds,
        "remaining": max(Decimal("0"), remaining),
        "message": message,
    }


def _threshold_for_section(section: str, vendor_type: str) -> Decimal:
    if section == "194I":
        return Decimal("240000")
    if section == "194J":
        return Decimal("30000")
    if section == "194C":
        return Decimal("100000") if vendor_type == "individual" else Decimal("30000")
    return Decimal("0")