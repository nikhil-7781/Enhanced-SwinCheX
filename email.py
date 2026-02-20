from __future__ import annotations

import smtplib
from email.message import EmailMessage

from app.config import settings


def send_invite_email(recipient: str, invite_link: str) -> None:
    if not settings.smtp_user or not settings.smtp_password:
        # Skip email in dev if not configured
        return

    msg = EmailMessage()
    msg["Subject"] = "You are invited to TDS Portal"
    msg["From"] = settings.smtp_user
    msg["To"] = recipient
    msg.set_content(f"You have been invited. Use this link to sign up: {invite_link}")

    with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as server:
        server.starttls()
        server.login(settings.smtp_user, settings.smtp_password)
        server.send_message(msg)