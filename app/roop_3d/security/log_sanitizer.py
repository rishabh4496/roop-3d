"""Log sanitization, secret redaction, and biometric privacy filters for application logging."""
from __future__ import annotations

import logging
import re
from urllib.parse import urlsplit


def sanitize_log_message(message: str) -> str:
    """Scrub tokens, secrets, biometric embeddings, and credentials from log strings."""
    # 1. Bearer tokens
    message = re.sub(r"(?i)\bBearer\s+[^\s,;\"']+", "Bearer [REDACTED]", message)

    # 2. Key-value secrets (tokens, api keys, signatures, secrets)
    message = re.sub(
        r"(?i)\b(?:roop3d_token|consent_token|access_token|api_key|signature|token|secret|password|client_secret)[\"']?\s*[=:]\s*[\"']?[^\s,;\"']+",
        "secret=[REDACTED]",
        message,
    )

    # 3. JWTs / base64-encoded structured tokens
    message = re.sub(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)?", "[REDACTED_JWT]", message)

    # 4. PEM private keys
    message = re.sub(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----",
        "[REDACTED_PRIVATE_KEY]",
        message,
    )

    # 5. Raw 512-D biometric embedding arrays in logs
    message = re.sub(
        r"\[(?:-?\d+\.\d+,\s*){8,}-?\d+\.\d+\]",
        "[BIOMETRIC_EMBEDDING_REDACTED]",
        message,
    )

    # 6. Sanitize URLs containing sensitive query strings
    def redact_url(match: re.Match) -> str:
        url_str = match.group()
        try:
            parsed = urlsplit(url_str)
            if parsed.query:
                return f"{parsed.scheme}://{parsed.netloc}{parsed.path}?[REDACTED_QUERY]"
            return url_str
        except Exception:
            return "[REDACTED_URL]"

    message = re.sub(r"https?://[^\s\"'<>]+", redact_url, message)
    return message


class SecurityRedactionFilter(logging.Filter):
    """Logging filter that scrubs sensitive fields and strips raw stack traces."""

    def filter(self, record: logging.LogRecord) -> bool:
        original = record.getMessage()
        record.msg = sanitize_log_message(original)
        record.args = ()
        if record.exc_info:
            record.msg += " [exception details redacted for security]"
            record.exc_info = None
            record.exc_text = None
        record.stack_info = None
        return True


def install_log_sanitizer() -> None:
    """Attach SecurityRedactionFilter to all roop3d and uvicorn loggers."""
    filter_instance = SecurityRedactionFilter()
    target_loggers = (
        logging.getLogger(),
        logging.getLogger("roop3d"),
        logging.getLogger("roop3d.requests"),
        logging.getLogger("uvicorn"),
        logging.getLogger("uvicorn.access"),
        logging.getLogger("uvicorn.error"),
    )
    for logger in target_loggers:
        if not any(isinstance(f, SecurityRedactionFilter) for f in logger.filters):
            logger.addFilter(filter_instance)
