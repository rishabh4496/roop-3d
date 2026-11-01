"""Opt-in network sharing (`--server_share`): per-launch bearer token for remote peers.

Loopback peers are never offered a weaker policy than the default service: they keep the
strict loopback Host/Origin rules and need no token. Only a non-loopback peer must
present the token, and its Origin (when sent) must equal its Host.
"""
from __future__ import annotations

from dataclasses import dataclass
import hmac
import ipaddress
import secrets
from typing import Literal
from urllib.parse import parse_qs

COOKIE_NAME = "roop3d_token"


@dataclass(frozen=True)
class ShareMode:
    token: str

    @classmethod
    def create(cls) -> "ShareMode":
        return cls(secrets.token_urlsafe(32))

    def accepts(self, supplied: str | None) -> bool:
        return bool(supplied) and hmac.compare_digest(supplied.encode(), self.token.encode())

    def credential(self, scope: dict) -> tuple[str | None, Literal["bearer", "query", "cookie"] | None]:
        """Token from `Authorization: Bearer`, `?token=` or the HTTP-only cookie, in that order."""
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        authorization = headers.get(b"authorization", b"").decode("latin1")
        scheme, _, value = authorization.partition(" ")
        if scheme.lower() == "bearer" and value.strip():
            return value.strip(), "bearer"
        query = parse_qs(scope.get("query_string", b"").decode("latin1")).get("token")
        if query:
            return query[0], "query"
        for part in headers.get(b"cookie", b"").decode("latin1").split(";"):
            name, _, cookie_value = part.strip().partition("=")
            if name == COOKIE_NAME and cookie_value:
                return cookie_value, "cookie"
        return None, None


def is_loopback_peer(client: tuple[str, int] | None) -> bool:
    if not client:
        return False
    try:
        address = ipaddress.ip_address(client[0])
    except ValueError:
        return False
    mapped = getattr(address, "ipv4_mapped", None)
    return (mapped or address).is_loopback
