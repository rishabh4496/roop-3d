"""Stage 12: optional, clean-room bridge to an external Blender 4.x.

GPL isolation (hard rule). Blender's Python API (``bpy``) is GPL. Nothing in ``roop_3d`` imports it, directly or lazily:

* ``BlenderMCPBridge`` is a plain TCP client. It only *sends text* (JSON commands, and a short Python snippet as a string) to a
  Blender-side MCP server that the user started themselves, and reads text back. The snippet is data here, never executed here.
* ``EXPORTER_SCRIPT`` is a standalone script the user downloads and runs *inside* Blender. It is stored as a string, carries its
  own ``GPL-3.0-or-later`` header, and is never imported or executed by this process.
* ``tests/test_glb_export.py`` parses every ``roop_3d`` source file and fails on any ``bpy`` import.

Wire protocol (what is and is not verified). The default dialect, ``blendermcp``, follows the public ``ahujasid/blender-mcp`` add-on
as far as it could be confirmed: a socket server on localhost, default port 9876, with commands ``get_scene_info`` and
``execute_code``. The *exact* framing (``{"type", "params"}`` request; ``{"status": "success", "result"}`` /
``{"status": "error", "message"}`` response; no delimiter, a message is complete when it parses as JSON) is taken from
a model summary of that add-on and has NOT been checked against a running Blender here. The reader is therefore tolerant: it
accepts any reply that parses as JSON (newline-terminated or not) and understands both the status envelope and JSON-RPC 2.0
(``dialect="jsonrpc"``: newline-delimited ``{"jsonrpc", "id", "method", "params"}``).

Only loopback hosts are accepted; the bridge cannot be pointed at another machine.
"""
from __future__ import annotations

import ipaddress
import json
import os
from pathlib import Path
import re
import socket
import uuid
from typing import Any, Callable, Literal

from roop_3d.core.errors import DomainError

DEFAULT_PORT = 9876
MAX_REPLY_BYTES = 16 * 1024 * 1024
Dialect = Literal["blendermcp", "jsonrpc"]
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


class BlenderBridgeError(DomainError):
    """Any bridge failure; ``code`` says which (``BLENDER_BRIDGE_*``)."""


def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


# Run inside Blender by the MCP server (Blender's own interpreter); roop-3d only formats the path into it with json.dumps.
_IMPORT_SNIPPET = """\
import bpy, json
_path = {path}
_before = set(bpy.data.objects.keys())
bpy.ops.import_scene.gltf(filepath=_path)
_new = [o.name for o in bpy.data.objects if o.name not in _before]
print(json.dumps({{"imported": _new, "path": _path}}))
"""


class BlenderMCPBridge:
    """Minimal localhost client. One short-lived connection per call, so a Blender restart never leaves a stale socket."""

    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT, *, timeout: float = 15.0, dialect: Dialect = "blendermcp",
                 connector: Callable[..., socket.socket] | None = None):
        if not _is_loopback(host):
            raise BlenderBridgeError("BLENDER_BRIDGE_NOT_LOCAL", "The Blender bridge only talks to a loopback address", 422)
        if not 1 <= int(port) <= 65535:
            raise BlenderBridgeError("BLENDER_BRIDGE_PORT", "Port must be 1-65535", 422)
        if dialect not in ("blendermcp", "jsonrpc"):
            raise BlenderBridgeError("BLENDER_BRIDGE_DIALECT", "dialect must be 'blendermcp' or 'jsonrpc'", 422)
        self.host, self.port, self.timeout, self.dialect = host, int(port), float(timeout), dialect
        self._connect = connector or socket.create_connection
        self._next_id = 0

    # ----- transport
    def call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        """Send one command and return its ``result``; raises ``BlenderBridgeError`` for transport and remote errors."""
        params = params or {}
        self._next_id += 1
        if self.dialect == "jsonrpc":
            request = json.dumps({"jsonrpc": "2.0", "id": self._next_id, "method": method, "params": params}) + "\n"
        else:
            request = json.dumps({"type": method, "params": params})
        try:
            # Connecting to a closed loopback port is refused at once on Linux/macOS but can run into the timeout on Windows,
            # so a failure (or timeout) *while connecting* always means "no server", never "slow server".
            sock = self._connect((self.host, self.port), timeout=self.timeout)
        except OSError as err:                                                    # includes socket.timeout
            raise BlenderBridgeError(
                "BLENDER_BRIDGE_UNREACHABLE",
                f"No Blender MCP server answers on {self.host}:{self.port}. In Blender, enable the MCP add-on and start its server",
                503) from err
        try:
            with sock:
                sock.settimeout(self.timeout)
                sock.sendall(request.encode("utf-8"))
                reply = self._read_json(sock)
        except BlenderBridgeError:
            raise
        except socket.timeout as err:
            raise BlenderBridgeError("BLENDER_BRIDGE_TIMEOUT", f"Blender did not answer within {self.timeout:g} s", 504) from err
        except OSError as err:
            raise BlenderBridgeError("BLENDER_BRIDGE_CONNECTION_LOST", f"The connection to Blender broke: {err}", 502) from err
        return self._unwrap(reply)

    @staticmethod
    def _read_json(sock: socket.socket) -> Any:
        buffer = b""
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            buffer += chunk
            if len(buffer) > MAX_REPLY_BYTES:
                raise BlenderBridgeError("BLENDER_BRIDGE_REPLY_TOO_LARGE", "Blender's reply exceeds 16 MiB", 502)
            try:
                return json.loads(buffer.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue                                                          # incomplete: keep reading
        if not buffer.strip():
            raise BlenderBridgeError("BLENDER_BRIDGE_EMPTY_REPLY", "Blender closed the connection without a reply", 502)
        try:
            return json.loads(buffer.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as err:
            raise BlenderBridgeError("BLENDER_BRIDGE_BAD_REPLY", "Blender's reply is not valid JSON", 502) from err

    @staticmethod
    def _unwrap(reply: Any) -> Any:
        if not isinstance(reply, dict):
            raise BlenderBridgeError("BLENDER_BRIDGE_BAD_REPLY", "Blender's reply is not a JSON object", 502)
        error = reply.get("error")
        if error:
            message = error.get("message") if isinstance(error, dict) else str(error)
            raise BlenderBridgeError("BLENDER_BRIDGE_REMOTE_ERROR", f"Blender reported an error: {message}", 502)
        if reply.get("status") == "error":
            raise BlenderBridgeError("BLENDER_BRIDGE_REMOTE_ERROR", f"Blender reported an error: {reply.get('message', 'unknown')}", 502)
        return reply.get("result")

    # ----- commands
    def is_available(self) -> bool:
        """True when a server answers a ``get_scene_info``-style probe. Never raises."""
        try:
            self.call("get_scene_info")
            return True
        except DomainError:
            return False

    def scene_info(self) -> Any:
        return self.call("get_scene_info")

    def import_glb(self, path: Path) -> dict[str, Any]:
        """Ask Blender (same machine) to import ``path`` with its own glTF importer."""
        code = _IMPORT_SNIPPET.format(path=json.dumps(str(path)))                  # json.dumps: no way to break out of the literal
        result = self.call("execute_code", {"code": code})
        text = result.get("result") if isinstance(result, dict) else result
        imported: list[str] = []
        if isinstance(text, str):
            for line in reversed(text.strip().splitlines()):
                try:
                    imported = list(json.loads(line).get("imported", []))
                    break
                except (json.JSONDecodeError, AttributeError):
                    continue
        return {"path": str(path), "imported_objects": imported, "raw": result}

    def send_glb(self, glb: bytes, staging_dir: Path, name: str = "roop3d_head") -> dict[str, Any]:
        """Write ``glb`` atomically into ``staging_dir`` and have Blender import it. Returns the import report."""
        safe = _SAFE_NAME.sub("_", name).strip("._") or "roop3d_head"
        staging_dir.mkdir(parents=True, exist_ok=True)
        final = staging_dir / f"{safe}.glb"
        temporary = staging_dir / f".{safe}.{uuid.uuid4().hex[:8]}.part"
        try:
            temporary.write_bytes(glb)
            os.replace(temporary, final)
        finally:
            temporary.unlink(missing_ok=True)
        report = self.import_glb(final.resolve())
        report["bytes"] = len(glb)
        return report


# --------------------------------------------------------------------------------------------------------------------------------
# Standalone script for the user to run INSIDE Blender 4.x (Text Editor > Run Script, or `blender --python roop3d_blender_import.py`).
# It is text. roop-3d never imports or executes it.
EXPORTER_SCRIPT = '''\
# SPDX-License-Identifier: GPL-3.0-or-later
# roop-3d <-> Blender helper. Runs inside Blender 4.x only; it is not part of the roop-3d service and the service never imports it.
#
#   Import  : set GLB_SOURCE below (a .glb path, or an http://127.0.0.1:... URL of roop-3d's  .../identity-asset/mesh.glb),
#             or set the ROOP3D_GLB environment variable, or pass it after "--":  blender --python this.py -- head.glb
#   Export  : after sculpting, call  export_glb("C:/path/edited_head.glb")  from Blender's Python console.
#
# What you get: the head mesh with its albedo (and normal map, if the asset had one), the semantic sliders as Shape Keys, and
# - when the .glb carries one - the head track as an Action on the head object. The track holds rotation and in-plane position;
# depth is not metric (see the animation "extras" in the .glb), so translation Z is 0.

import os
import sys
import tempfile
import urllib.parse
import urllib.request

import bpy

GLB_SOURCE = ""          # path or http://127.0.0.1 / http://localhost URL


def _resolve(source):
    parsed = urllib.parse.urlparse(source)
    if parsed.scheme in ("http", "https"):
        if parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("Only loopback URLs are fetched: " + source)
        handle, target = tempfile.mkstemp(suffix=".glb", prefix="roop3d_")
        os.close(handle)
        with urllib.request.urlopen(source, timeout=60) as response, open(target, "wb") as out:
            out.write(response.read())
        return target
    return os.path.abspath(os.path.expanduser(source))


def import_glb(source):
    """Import a roop-3d .glb with Blender's built-in glTF importer; returns the new objects."""
    path = _resolve(source)
    before = set(bpy.data.objects.keys())
    bpy.ops.import_scene.gltf(filepath=path)
    created = [bpy.data.objects[name] for name in bpy.data.objects.keys() if name not in before]
    print("roop3d: imported", [o.name for o in created], "from", path)
    return created


def export_glb(path, selected_only=False):
    """Write the scene (or the selection) back to a .glb, shape keys and animation included."""
    bpy.ops.export_scene.gltf(filepath=os.path.abspath(path), export_format="GLB", use_selection=selected_only,
                              export_morph=True, export_animations=True, export_image_format="AUTO")
    print("roop3d: exported", path)


def _main():
    source = GLB_SOURCE or os.environ.get("ROOP3D_GLB", "")
    if not source and "--" in sys.argv:
        rest = sys.argv[sys.argv.index("--") + 1:]
        source = rest[0] if rest else ""
    if not source:
        print("roop3d: set GLB_SOURCE (or ROOP3D_GLB) to a .glb path or roop-3d URL, then run again.")
        return
    import_glb(source)


_main()
'''
