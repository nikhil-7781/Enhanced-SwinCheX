from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Iterable

from fastapi import UploadFile

ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png"}
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024


def sanitize_filename(filename: str) -> str:
    return "".join(c for c in filename if c.isalnum() or c in {".", "_", "-"})


def validate_file(upload: UploadFile) -> None:
    ext = Path(upload.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError("Unsupported file type")


def save_upload(upload: UploadFile, upload_dir: str, subdir: str) -> str:
    validate_file(upload)
    os.makedirs(os.path.join(upload_dir, subdir), exist_ok=True)

    ext = Path(upload.filename).suffix.lower()
    filename = f"{uuid.uuid4()}{ext}"
    file_path = os.path.join(upload_dir, subdir, filename)

    size = 0
    with open(file_path, "wb") as buffer:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_FILE_SIZE_BYTES:
                buffer.close()
                os.remove(file_path)
                raise ValueError("File too large")
            buffer.write(chunk)

    return file_path