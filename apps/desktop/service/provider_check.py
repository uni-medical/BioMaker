"""Bounded desktop provider check, independent from the scientific engine."""

from __future__ import annotations

import ipaddress
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

DEFAULT_BASE_URL = "https://api.openai.com/v1"
TOTAL_TIMEOUT_SECONDS = 25.0
MAX_REQUEST_BYTES = 16_384
PROBE = Path(__file__).with_name("provider_probe.py")
CODES = frozenset({
    "ok", "invalid_url", "missing_key", "invalid_key", "missing_model", "auth",
    "model", "unavailable", "rate_limit", "timeout", "network", "invalid_response",
    "response_too_large", "redirect", "busy", "internal", "invalid_request", "request_rejected",
})


def result(code: str, started: float) -> dict:
    code = code if code in CODES else "internal"
    return {
        "status": "ok" if code == "ok" else "error",
        "code": code,
        "elapsed_ms": max(0, round((time.monotonic() - started) * 1000)),
        "checked_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def normalize_base_url(value: str | None) -> str:
    if value is None:
        return DEFAULT_BASE_URL
    if not isinstance(value, str) or len(value) > 4096:
        raise ValueError("invalid_url")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError("invalid_url")
    raw = value.strip()
    if not raw:
        return DEFAULT_BASE_URL
    if any(char.isspace() for char in raw) or "?" in raw or "#" in raw or "\\" in raw:
        raise ValueError("invalid_url")
    try:
        parsed = urlsplit(raw)
        port = parsed.port
        host = parsed.hostname
        if (parsed.scheme not in {"http", "https"} or not host
                or parsed.username is not None or parsed.password is not None
                or (port is not None and port <= 0)):
            raise ValueError("invalid_url")
        if ":" in host:
            host = "[" + ipaddress.IPv6Address(host).compressed + "]"
        else:
            host = host.encode("idna").decode("ascii").lower()
            if not re.fullmatch(r"[a-z0-9.-]+", host) or any(not part for part in host.rstrip(".").split(".")):
                raise ValueError("invalid_url")
        if port is not None and port != (443 if parsed.scheme == "https" else 80):
            host += f":{port}"
        path = parsed.path
        if re.search(r"%(?![0-9a-fA-F]{2})", path):
            raise ValueError("invalid_url")
        if any(part in {".", ".."} for part in path.split("/")):
            raise ValueError("invalid_url")
        path = quote(path, safe="/%:@!$&'()*+,;=-._~")
        # The SDK adds one slash only when needed; preserve repeated slashes.
        if path.endswith("/"):
            path = path[:-1]
        return urlunsplit((parsed.scheme, host, path, "", ""))
    except (ValueError, UnicodeError):
        raise ValueError("invalid_url") from None


def validated_payload(payload: object, saved: dict[str, str]) -> dict:
    if not isinstance(payload, dict) or set(payload) - {"base_url", "api_key", "model", "use_saved_key"}:
        raise ValueError("invalid_request")
    for field in ("base_url", "api_key", "model"):
        if payload.get(field) is not None and not isinstance(payload[field], str):
            raise ValueError("invalid_request")
    base = normalize_base_url(payload.get("base_url"))
    model = payload.get("model")
    if (not isinstance(model, str) or not model.strip() or len(model) > 500
            or any(ord(char) < 32 or ord(char) == 127 for char in model)):
        raise ValueError("missing_model")
    use_saved = payload.get("use_saved_key", True)
    if not isinstance(use_saved, bool):
        raise ValueError("invalid_request")
    key = payload.get("api_key")
    if isinstance(key, str) and any(ord(char) < 32 or ord(char) == 127 for char in key):
        raise ValueError("invalid_key")
    if not key or not key.strip():
        key = None
        if use_saved:
            try:
                saved_base = normalize_base_url(saved.get("OPENAI_BASE_URL"))
            except ValueError:
                saved_base = None
            if saved_base == base:
                key = saved.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("missing_key")
    if len(key) > 8192 or any(ord(char) <= 32 or ord(char) >= 127 for char in key):
        raise ValueError("invalid_key")
    return {"base_url": base, "api_key": key, "model": model.strip()}


class ProviderChecker:
    def __init__(self, env_file: Path, *, timeout: float = TOTAL_TIMEOUT_SECONDS):
        self.env_file = env_file
        self.timeout = min(max(timeout, 0.05), TOTAL_TIMEOUT_SECONDS)
        self._lock = threading.Lock()
        self._busy = False
        self._in_check = False
        self._closed = False
        self._process: subprocess.Popen | None = None

    @staticmethod
    def _stop(process: subprocess.Popen) -> bool:
        try:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=1.0)
            return True
        except (OSError, subprocess.TimeoutExpired):
            # The independent watchdog still owns the worker's total lifetime.
            # Retain its handle until a later poll confirms that it has exited.
            return process.poll() is not None

    @staticmethod
    def _close_pipes(process: subprocess.Popen) -> None:
        for pipe in (process.stdin, process.stdout):
            if pipe is not None:
                try:
                    pipe.close()
                except OSError:
                    pass

    def _reap_locked(self) -> None:
        if not self._in_check and self._process is not None and self._process.poll() is not None:
            self._close_pipes(self._process)
            self._process = None
            self._busy = False

    def check(self, payload: object, *, started: float | None = None) -> dict:
        started = time.monotonic() if started is None else started
        with self._lock:
            self._reap_locked()
            if self._closed:
                return result("internal", started)
            if self._busy:
                return result("busy", started)
            self._busy = True
            self._in_check = True
        process = None
        try:
            from webui.env_file import read_env_values
            try:
                request = validated_payload(payload, read_env_values(self.env_file))
            except ValueError as error:
                return result(str(error), started)
            remaining = self.timeout - (time.monotonic() - started)
            if remaining <= 0:
                return result("timeout", started)
            encoded = json.dumps(request, ensure_ascii=False).encode("utf-8")
            if len(encoded) > MAX_REQUEST_BYTES:
                return result("invalid_request", started)
            # Secrets travel only through stdin. Do not inherit provider keys,
            # proxy credentials, desktop auth, or engine runtime configuration.
            environment = {key: os.environ[key] for key in ("SYSTEMROOT", "WINDIR") if key in os.environ}
            with self._lock:
                if self._closed:
                    return result("internal", started)
                process = subprocess.Popen(
                    [sys.executable, "-I", "-B", str(PROBE), str(os.getpid()), str(remaining)],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    env=environment, close_fds=True,
                )
                self._process = process
            try:
                output, _ = process.communicate(encoded, timeout=max(0.01, self.timeout - (time.monotonic() - started)))
            except subprocess.TimeoutExpired:
                if self._stop(process):
                    process.communicate(timeout=1.0)
                return result("timeout", started)
            if process.returncode == 71:
                return result("timeout", started)
            if process.returncode != 0 or len(output) > 64:
                return result("internal", started)
            return result(output.decode("ascii").strip(), started)
        except Exception:
            # Never surface exception strings that could contain payload data.
            return result("internal", started)
        finally:
            stopped = process is None or self._stop(process)
            if process is not None and stopped:
                self._close_pipes(process)
            with self._lock:
                self._in_check = False
                if stopped:
                    self._process = None
                    self._busy = False

    def close(self) -> None:
        with self._lock:
            self._closed = True
            process = self._process
        if process is not None:
            self._stop(process)
        with self._lock:
            self._reap_locked()


async def check_request(request, checker: ProviderChecker) -> dict:
    """Manual bounded JSON validation prevents FastAPI from reflecting keys."""
    import asyncio
    started = time.monotonic()

    async def read_payload():
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > MAX_REQUEST_BYTES:
                raise ValueError()
            body.extend(chunk)
        return json.loads(body)

    try:
        payload = await asyncio.wait_for(read_payload(), timeout=min(2.0, checker.timeout))
    except asyncio.TimeoutError:
        return result("timeout", started)
    except Exception:
        return result("invalid_request", started)
    return await asyncio.to_thread(checker.check, payload, started=started)
