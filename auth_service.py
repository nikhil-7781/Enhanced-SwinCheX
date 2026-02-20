from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from app.config import settings
from app.models import Company, Invite, User, UserRole
from app.utils.security import create_access_token, hash_password, verify_password
from app.utils.validators import validate_password_policy


class AuthError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def create_company_and_admin(
    db: Session,
    *,
    company_name: str,
    tan: str,
    pan: str,
    address: str | None,
    financial_year: str | None,
    admin_full_name: str,
    admin_email: str,
    admin_password: str,
) -> User:
    if not validate_password_policy(admin_password):
        raise AuthError("WEAK_PASSWORD", "Password does not meet policy requirements")

    existing = db.query(User).filter(User.email == admin_email).first()
    if existing:
        raise AuthError("EMAIL_EXISTS", "Email is already registered")

    company = Company(
        name=company_name,
        tan=tan,
        pan=pan,
        address=address,
        financial_year=financial_year,
    )
    db.add(company)
    db.flush()

    admin = User(
        company_id=company.id,
        email=admin_email,
        password_hash=hash_password(admin_password),
        full_name=admin_full_name,
        role=UserRole.admin,
        is_active=True,
    )
    db.add(admin)
    db.commit()
    db.refresh(admin)
    return admin


def authenticate_user(db: Session, *, email: str, password: str) -> User:
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(password, user.password_hash):
        raise AuthError("INVALID_CREDENTIALS", "Invalid email or password")
    if not user.is_active:
        raise AuthError("INACTIVE_USER", "User is inactive")
    return user


def create_token_for_user(user: User) -> str:
    return create_access_token(
        {
            "user_id": str(user.id),
            "company_id": str(user.company_id) if user.company_id else None,
            "role": user.role.value,
        }
    )


def create_invite(db: Session, *, company_id: str, email: str) -> Invite:
    token = uuid.uuid4().hex
    expires_at = datetime.now(timezone.utc) + timedelta(days=7)
    invite = Invite(company_id=company_id, email=email, token=token, expires_at=expires_at, used=False)
    db.add(invite)
    db.commit()
    db.refresh(invite)
    return invite


def accept_invite(
    db: Session,
    *,
    invite_token: str,
    full_name: str,
    password: str,
) -> User:
    if not validate_password_policy(password):
        raise AuthError("WEAK_PASSWORD", "Password does not meet policy requirements")

    invite = db.query(Invite).filter(Invite.token == invite_token).first()
    if not invite:
        raise AuthError("INVALID_INVITE", "Invite token is invalid")
    if invite.used:
        raise AuthError("INVITE_USED", "Invite already used")
    if invite.expires_at < datetime.now(timezone.utc):
        raise AuthError("INVITE_EXPIRED", "Invite token has expired")

    existing = db.query(User).filter(User.email == invite.email).first()
    if existing:
        raise AuthError("EMAIL_EXISTS", "Email is already registered")

    user = User(
        company_id=invite.company_id,
        email=invite.email,
        password_hash=hash_password(password),
        full_name=full_name,
        role=UserRole.employee,
        is_active=True,
    )
    invite.used = True
    db.add(user)
    db.commit()
    db.refresh(user)
    return user