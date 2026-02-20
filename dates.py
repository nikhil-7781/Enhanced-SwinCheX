from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta


@dataclass
class FiscalPeriod:
    financial_year: str
    quarter: str


def get_fiscal_period(target_date: date | None = None) -> FiscalPeriod:
    current = target_date or date.today()
    year = current.year
    month = current.month

    if month >= 4:
        fy_start = year
        fy_end = year + 1
    else:
        fy_start = year - 1
        fy_end = year

    if 4 <= month <= 6:
        quarter = "Q1"
    elif 7 <= month <= 9:
        quarter = "Q2"
    elif 10 <= month <= 12:
        quarter = "Q3"
    else:
        quarter = "Q4"

    financial_year = f"{fy_start}-{str(fy_end)[-2:]}"
    return FiscalPeriod(financial_year=financial_year, quarter=quarter)


def quarter_due_date(quarter: str, year: int) -> date:
    # Approximate due date for TDS challan: 7th of next month, except March = April 30
    if quarter == "Q1":
        return date(year, 7, 7)
    if quarter == "Q2":
        return date(year, 10, 7)
    if quarter == "Q3":
        return date(year + 1, 1, 7)
    # Q4
    return date(year + 1, 4, 30)


def days_remaining(target_date: date) -> int:
    today = date.today()
    delta = target_date - today
    return max(0, delta.days)


def fy_month_range(financial_year: str) -> tuple[str, str]:
    # financial_year format: "2024-25"
    start_year = int(financial_year.split("-")[0])
    end_year = start_year + 1
    start = f"{start_year:04d}-04"
    end = f"{end_year:04d}-03"
    return start, end
