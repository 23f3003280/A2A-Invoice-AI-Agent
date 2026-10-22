import hashlib
import json
import re
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def get_message_id(message: dict) -> str:
    value = message.get("messageId") or message.get("message_id") or message.get("id")
    if not value or not str(value).strip():
        raise ValueError("message.messageId is required")
    return str(value)


def extract_data(message: dict) -> dict:
    if isinstance(message.get("data"), dict):
        return message["data"]
    content = message.get("content")
    if isinstance(content, dict):
        return content
    for part in message.get("parts") or []:
        if isinstance(part, dict) and isinstance(part.get("data"), dict):
            return part["data"]
    return {}


def extract_media_type(message: dict) -> str | None:
    if message.get("mediaType"):
        return message["mediaType"]
    for part in message.get("parts") or []:
        if isinstance(part, dict):
            if part.get("mediaType"):
                return part["mediaType"]
            if part.get("media_type"):
                return part["media_type"]
    return None


def extract_message(request_body: dict) -> dict:
    return request_body.get("message") if isinstance(request_body.get("message"), dict) else request_body


def safe_id(prefix: str = "task") -> str:
    import uuid
    return f"{prefix}_{uuid.uuid4().hex}"


def evidence_ref_is_bracketed(value: str) -> bool:
    return bool(re.fullmatch(r"\[[^\[\]]{1,120}\]", value.strip()))


def semantic_package(package: dict) -> dict:
    """Remove namespace-only identifiers so equivalent packages can share decisions."""
    return {k: v for k, v in package.items() if k not in {"packageId", "package_id", "id"}}
