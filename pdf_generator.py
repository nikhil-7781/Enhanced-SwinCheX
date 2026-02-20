from __future__ import annotations

from pathlib import Path
from typing import Iterable

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


def generate_form16a(
    *,
    output_path: str,
    company_tan: str,
    vendor_pan: str,
    quarter: str,
    total_payment: float,
    tds_amount: float,
    section: str,
) -> str:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(output_path, pagesize=A4)
    c.setFont("Helvetica", 12)
    c.drawString(50, 800, "Form 16A (V1)")
    c.drawString(50, 780, f"Company TAN: {company_tan}")
    c.drawString(50, 760, f"Vendor PAN: {vendor_pan}")
    c.drawString(50, 740, f"Quarter: {quarter}")
    c.drawString(50, 720, f"Section: {section}")
    c.drawString(50, 700, f"Total Payment: {total_payment}")
    c.drawString(50, 680, f"TDS Amount: {tds_amount}")
    c.showPage()
    c.save()
    return output_path


def generate_form16(
    *,
    output_path: str,
    employee_name: str,
    financial_year: str,
    total_gross: float,
    total_tds: float,
) -> str:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(output_path, pagesize=A4)
    c.setFont("Helvetica", 12)
    c.drawString(50, 800, "Form 16 (V1)")
    c.drawString(50, 780, f"Employee: {employee_name}")
    c.drawString(50, 760, f"Financial Year: {financial_year}")
    c.drawString(50, 740, f"Total Gross Salary: {total_gross}")
    c.drawString(50, 720, f"Total TDS Deducted: {total_tds}")
    c.showPage()
    c.save()
    return output_path
