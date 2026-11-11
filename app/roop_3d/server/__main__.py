"""Reserve the exact loopback socket before Uvicorn starts; never kill a port owner."""
from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys

from roop_3d.telemetry.pinokio_terminal import HUD, HudLogHandler, enforce_terminal_environment

# The standalone launcher puts app/ on sys.path before this module is imported.
try:
    from core.trt_bootstrap import bootstrap_native_library_paths
except ImportError:
    from app.core.trt_bootstrap import bootstrap_native_library_paths
bootstrap_native_library_paths()

from roop_3d.core.dll_injection import setup_environment
setup_environment()

import uvicorn

from roop_3d.core.settings import Settings
from roop_3d.core.redaction import redact
from roop_3d.audit.isolation import AuditError
try:
    from app.api import create_unified_app as create_app
except ImportError:
    from roop_3d.server.app import create_app
from roop_3d.server.security import ShareMode

SHARE_BIND_HOST = "0.0.0.0"


class PortCollisionError(RuntimeError):
    pass


def reserve_socket(settings: Settings, host: str | None = None) -> socket.socket:
    host = host or settings.host
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        listener.bind((host, settings.port))
        listener.listen(128)
        listener.setblocking(False)
        return listener
    except OSError as error:
        listener.close()
        raise PortCollisionError(f"Cannot bind {host}:{settings.port}. Another process may own the port. "
                                 "Set ROOP3D_PORT to an unused port; no existing process was stopped.") from error


def lan_addresses() -> list[str]:
    try:
        found = socket.gethostbyname_ex(socket.gethostname())[2]
    except OSError:
        return []
    return sorted({address for address in found if not address.startswith("127.")})


def parse_arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="roop_3d.server", description="Roop 3D local service")
    parser.add_argument("--server_share", "--server-share", dest="server_share", action="store_true",
                        default=os.environ.get("ROOP3D_SERVER_SHARE", "").lower() in {"1", "true", "yes"},
                        help="Listen on all interfaces; remote peers need the per-launch access token")
    return parser.parse_args(argv)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        try:
            value = json.loads(message)
            if isinstance(value, dict):
                return json.dumps(redact(value))
        except (ValueError, TypeError):
            pass
        return json.dumps({"level": record.levelname.lower(), "logger": record.name, "message": message})


class ReadyServer(uvicorn.Server):
    share: ShareMode | None = None

    async def startup(self, sockets: list[socket.socket] | None = None) -> None:
        await super().startup(sockets=sockets)
        if self.started:
            # The loopback line stays first: Pinokio captures the first http://host:port it sees.
            print(json.dumps({"event": "ready", "application": "roop_3d",
                              "url": f"http://127.0.0.1:{self.config.port}"}), flush=True)
            if self.share is not None:
                # Deliberately printed to this terminal only: the per-launch token is the remote credential.
                print(json.dumps({"event": "share", "token": self.share.token,
                                  "urls": [f"http://{address}:{self.config.port}/?token={self.share.token}"
                                           for address in lan_addresses() or ["<this-machine-ip>"]]}), flush=True)
            # After the ready line: the 14-section report names URLs and Pinokio keeps the first one it sees.
            HUD.boot(getattr(self.config.app, "state", None))


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(sys.argv[1:] if argv is None else argv)
    enforce_terminal_environment()
    HUD.arm()
    handler = HudLogHandler(sys.stdout)  # a log record clears the live HUD line, prints, and redraws it
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
    share = ShareMode.create() if arguments.server_share else None
    # Stage 18: if this process dies for any reason (crash, kill, power-off of its parent) the OS reaps its FFmpeg children.
    from roop_3d.reliability.processes import bind_children_to_process_lifetime
    bind_children_to_process_lifetime()
    try:
        settings = Settings.from_env()
        listener = reserve_socket(settings, SHARE_BIND_HOST if share else None)
    except (ValueError, OSError, AuditError, PortCollisionError) as error:
        message = str(error) if isinstance(error, PortCollisionError) else "Invalid ROOP3D settings; host must be 127.0.0.1 and API port must not be 5174"
        print(json.dumps({"error": {"code": "PORT_IN_USE" if isinstance(error, PortCollisionError) else "INVALID_SETTINGS", "message": message}}), flush=True)
        return 2
    try:
        config = uvicorn.Config(create_app(settings, share=share), host=SHARE_BIND_HOST if share else settings.host,
                                port=settings.port, log_config=None,
                                access_log=False, proxy_headers=False, timeout_graceful_shutdown=10,
                                ws="websockets-sansio", ws_max_size=16384, ws_max_queue=16)
        server = ReadyServer(config)
        server.share = share
        server.run(sockets=[listener])
        return 0
    finally:
        listener.close()


if __name__ == "__main__":
    raise SystemExit(main())
