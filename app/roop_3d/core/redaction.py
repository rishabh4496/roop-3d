"""Defense in depth for structured diagnostics; request bodies are never logged."""
from __future__ import annotations

from typing import Any
from pydantic import SecretBytes, SecretStr


def redact(value: Any) -> Any:
    if isinstance(value, (SecretStr, SecretBytes)):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(key): "[REDACTED]" if any(word in str(key).lower().replace("-", "").replace("_", "") for word in
                                            ("secret", "password", "token", "apikey", "authorization", "idempotencykey", "credential", "privatekey"))
                else redact(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [redact(item) for item in value]
    return value
