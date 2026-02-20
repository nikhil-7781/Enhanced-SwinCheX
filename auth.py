from __future__ import annotations

from fastapi import APIRouter, Depends, Response, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.middleware.auth_middleware import get_current_user, require_role
from app.models import UserRole
from app.schemas.auth import (
    AuthResponse,
    CompanyRegisterRequest,
    EmployeeSignupRequest,
    InviteEmployeeRequest,
    InviteResponse,
    LoginRequest,
    UserResponse,
)
from app.schemas.common import ErrorDetail, ErrorResponse, SuccessResponse
from app.services.auth_service import AuthError, accept_invite, authenticate_user, create_company_and_admin, create_invite, create_token_for_user
from app.utils.email import send_invite_email

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register/company", response_model=SuccessResponse[AuthResponse])
def register_company(payload: CompanyRegisterRequest, db: Session = Depends(get_db)):
    try:
        admin = create_company_and_admin(
            db,
            company_name=payload.company_name,
            tan=payload.tan.upper(),
            pan=payload.pan.upper(),
            address=payload.address,
            financial_year=payload.financial_year,
            admin_full_name=payload.admin_full_name,
            admin_email=payload.admin_email.lower(),
            admin_password=payload.admin_password,
        )
    except AuthError as exc:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(error=ErrorDetail(code=exc.code, message=exc.message)).model_dump(),
        )

    token = create_token_for_user(admin)
    response = SuccessResponse(data=AuthResponse(user=_user_to_response(admin)))
    return _attach_cookie(response, token)


@router.post("/login", response_model=SuccessResponse[AuthResponse])
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    try:
        user = authenticate_user(db, email=payload.email.lower(), password=payload.password)
    except AuthError as exc:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content=ErrorResponse(error=ErrorDetail(code=exc.code, message=exc.message)).model_dump(),
        )

    token = create_token_for_user(user)
    response = SuccessResponse(data=AuthResponse(user=_user_to_response(user)))
    return _attach_cookie(response, token)


@router.post("/invite-employee", response_model=SuccessResponse[InviteResponse])
def invite_employee(
    payload: InviteEmployeeRequest,
    db: Session = Depends(get_db),
    admin=Depends(require_role(UserRole.admin)),
):
    invite = create_invite(db, company_id=str(admin.company_id), email=payload.email.lower())
    invite_link = f"{settings.frontend_url}/employee/signup/{invite.token}"
    send_invite_email(invite.email, invite_link)
    return SuccessResponse(data=InviteResponse(invite_token=invite.token, expires_at=invite.expires_at.isoformat()))


@router.post("/employee-signup/{invite_token}", response_model=SuccessResponse[AuthResponse])
def employee_signup(invite_token: str, payload: EmployeeSignupRequest, db: Session = Depends(get_db)):
    try:
        user = accept_invite(db, invite_token=invite_token, full_name=payload.full_name, password=payload.password)
    except AuthError as exc:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(error=ErrorDetail(code=exc.code, message=exc.message)).model_dump(),
        )

    token = create_token_for_user(user)
    response = SuccessResponse(data=AuthResponse(user=_user_to_response(user)))
    return _attach_cookie(response, token)


@router.get("/me", response_model=SuccessResponse[AuthResponse])
def me(user=Depends(get_current_user)):
    return SuccessResponse(data=AuthResponse(user=_user_to_response(user)))


@router.post("/logout", response_model=SuccessResponse[dict])
def logout():
    response = SuccessResponse(data={"logged_out": True})
    return _clear_cookie(response)


def _user_to_response(user) -> UserResponse:
    return UserResponse(
        id=str(user.id),
        company_id=str(user.company_id) if user.company_id else None,
        email=user.email,
        full_name=user.full_name,
        role=user.role.value,
    )


def _attach_cookie(payload: SuccessResponse, token: str):
    response = Response(content=payload.model_dump_json(), media_type="application/json")
    response.set_cookie(
        "access_token",
        token,
        httponly=True,
        samesite="lax",
        secure=settings.app_env != "development",
        max_age=settings.jwt_expiry_hours * 3600,
    )
    return response


def _clear_cookie(payload: SuccessResponse):
    response = Response(content=payload.model_dump_json(), media_type="application/json")
    response.delete_cookie("access_token")
    return response