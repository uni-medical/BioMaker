"""Desktop-only provider settings transactions; no provider requests are made.

The advisory lock coordinates desktop writers. Revisions also detect edits made
before a transaction by other tools; uncooperative concurrent editors cannot be
given compare-and-swap guarantees by an advisory filesystem lock.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
import hashlib
import hmac
import io
import json
import os
from pathlib import Path
import re
import secrets
import stat
import tempfile
import time

from dotenv import dotenv_values
from fastapi.responses import JSONResponse

from provider_check import normalize_base_url

MAX_REQUEST_BYTES = 128 * 1024
MAX_CONFIG_BYTES = 256 * 1024
LOCK_TIMEOUT_SECONDS = 1.0
REQUEST_TIMEOUT_SECONDS = 2.0
MESSAGES = {
    "key_required": "Enter a key for the new API address, or explicitly remove the saved key before saving.",
    "conflict": "Saved settings changed. Reload the saved settings before saving again.",
    "invalid_url": "Enter a valid full HTTP or HTTPS API base URL.",
    "invalid_key": "The API key contains unsupported characters or is too long.",
    "invalid_request": "The settings request is invalid. Review the entered settings.",
    "busy": "Another settings operation is in progress. Try again shortly.",
    "io_error": "The settings file could not be read or saved. Check local disk access and try again.",
}


class ProviderSettingsError(Exception):
    def __init__(self, code: str):
        self.code = code if code in MESSAGES else "io_error"
        self.status_code = 409 if self.code in {"key_required", "conflict", "busy"} else 503 if self.code == "io_error" else 422
        super().__init__(f"[provider_settings:{self.code}] {MESSAGES[self.code]}")


def _validate_payload(payload: object) -> tuple[dict, str | None]:
    from webui.models import EnvUpdate

    if not isinstance(payload, dict):
        raise ProviderSettingsError("invalid_request")
    fields = dict(payload)
    revision = fields.pop("expected_revision", None)
    if "expected_revision" in payload and (not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{64}", revision)):
        raise ProviderSettingsError("invalid_request")
    try:
        model = EnvUpdate.model_validate(fields, strict=True)
    except ValueError:
        # Pydantic diagnostics can include submitted credentials. Return no
        # validation objects, payloads, paths or underlying exception strings.
        raise ProviderSettingsError("invalid_request") from None
    updates = {key: getattr(model, key) for key in model.model_fields_set}
    if "OPENAI_BASE_URL" in updates:
        try:
            normalize_base_url(updates["OPENAI_BASE_URL"])
        except ValueError:
            raise ProviderSettingsError("invalid_url") from None
    key = updates.get("OPENAI_API_KEY")
    if key is not None and key.strip():
        if len(key.strip()) > 8192 or any(ord(char) <= 32 or ord(char) >= 127 for char in key.strip()):
            raise ProviderSettingsError("invalid_key")
    return updates, revision


class ProviderSettings:
    def __init__(self, path: Path, *, lock_timeout: float = LOCK_TIMEOUT_SECONDS):
        self.path = path
        self.lock_path = path.with_name(f".{path.name}.desktop-settings.lock")
        self.lock_timeout = lock_timeout
        # Tokens are opaque, not a public digest of low-entropy credentials.
        # A sidecar restart intentionally invalidates an older form's token.
        self._revision_key = secrets.token_bytes(32)

    def _revision(self, raw: bytes, exists: bool) -> str:
        value = b"biomaker-desktop-settings-v1\0" + bytes([exists]) + raw
        return hmac.new(self._revision_key, value, hashlib.sha256).hexdigest()

    @contextmanager
    def _locked(self):
        descriptor = None
        acquired = False
        try:
            flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
            descriptor = os.open(self.lock_path, flags, 0o600)
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or self.lock_path.is_symlink():
                raise ProviderSettingsError("io_error")
            deadline = time.monotonic() + self.lock_timeout
            while True:
                try:
                    if os.name == "nt":
                        import msvcrt
                        os.lseek(descriptor, 0, os.SEEK_SET)
                        msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    break
                except (BlockingIOError, PermissionError):
                    if time.monotonic() >= deadline:
                        raise ProviderSettingsError("busy") from None
                    time.sleep(min(0.02, max(0, deadline - time.monotonic())))
            yield
        except OSError:
            raise ProviderSettingsError("io_error") from None
        finally:
            if descriptor is not None:
                if acquired:
                    try:
                        if os.name == "nt":
                            import msvcrt
                            os.lseek(descriptor, 0, os.SEEK_SET)
                            msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
                        else:
                            import fcntl
                            fcntl.flock(descriptor, fcntl.LOCK_UN)
                    except OSError:
                        pass
                os.close(descriptor)

    def _read(self, path: Path | None = None) -> tuple[bytes, bool, int]:
        candidate = self.path if path is None else path
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0)
        try:
            descriptor = os.open(candidate, flags)
        except FileNotFoundError:
            return b"", False, 0o600
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or candidate.is_symlink() or info.st_size > MAX_CONFIG_BYTES:
                raise ProviderSettingsError("io_error")
            raw = stream.read(MAX_CONFIG_BYTES + 1)
            if len(raw) > MAX_CONFIG_BYTES:
                raise ProviderSettingsError("io_error")
            return raw, True, stat.S_IMODE(info.st_mode)

    @staticmethod
    def _values(raw: bytes) -> dict[str, str]:
        from webui.env_file import RUNTIME_ENV_KEYS

        try:
            # Parse once from the captured bytes, retaining the existing dotenv
            # interpolation contract while avoiding a second live file read.
            values = dotenv_values(stream=io.StringIO(raw.decode("utf-8")))
        except (ValueError, UnicodeError):
            raise ProviderSettingsError("io_error") from None
        return {key: str(values[key]) for key in RUNTIME_ENV_KEYS if values.get(key) not in (None, "")}

    def _response(self, raw: bytes, exists: bool) -> dict:
        from webui.env_file import EDITABLE_ENV_KEYS, SECRET_ENV_KEYS, mask_value

        values = self._values(raw)
        masked = {}
        for key in EDITABLE_ENV_KEYS:
            value = values.get(key, "")
            if key in SECRET_ENV_KEYS:
                masked[key] = {"configured": key in values, "masked": mask_value(value), "secret": True}
            else:
                masked[key] = {"configured": key in values, "masked": value, "value": value, "secret": False}
        return {"values": masked, "source": {"path": str(self.path), "exists": exists}, "settings_revision": self._revision(raw, exists)}

    def get(self) -> dict:
        with self._locked():
            raw, exists, _mode = self._read()
            return self._response(raw, exists)

    def put(self, payload: object) -> dict:
        from webui.env_file import update_env_file

        updates, expected = _validate_payload(payload)
        with self._locked():
            raw, exists, mode = self._read()
            if expected is not None and not hmac.compare_digest(expected, self._revision(raw, exists)):
                raise ProviderSettingsError("conflict")
            saved = self._values(raw)
            if "OPENAI_BASE_URL" in updates and saved.get("OPENAI_API_KEY") and "OPENAI_API_KEY" not in updates:
                try:
                    old_base = normalize_base_url(saved.get("OPENAI_BASE_URL"))
                except ValueError:
                    old_base = None
                if old_base != normalize_base_url(updates["OPENAI_BASE_URL"]):
                    raise ProviderSettingsError("key_required")
            if not updates:
                return self._response(raw, exists)
            # Use the shared writer on an immutable private snapshot. Its own
            # read/merge cannot race another edit of the real configuration.
            temporary = None
            try:
                descriptor, name = tempfile.mkstemp(prefix=f".{self.path.name}.desktop-", suffix=".tmp", dir=self.path.parent)
                temporary = Path(name)
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(raw)
                update_env_file(temporary, updates)
                updated, _present, _mode = self._read(temporary)
                response = self._response(updated, True)
                os.chmod(temporary, mode)
                os.replace(temporary, self.path)
                temporary = None
                return response
            finally:
                if temporary is not None:
                    try:
                        temporary.unlink(missing_ok=True)
                    except OSError:
                        pass


async def provider_settings_request(request, settings: ProviderSettings):
    async def read_payload():
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > MAX_REQUEST_BYTES:
                raise ProviderSettingsError("invalid_request")
            body.extend(chunk)
        def object_pairs(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ProviderSettingsError("invalid_request")
                result[key] = value
            return result
        return json.loads(body, object_pairs_hook=object_pairs)

    try:
        if request.method == "GET":
            value = await asyncio.to_thread(settings.get)
        else:
            try:
                payload = await asyncio.wait_for(read_payload(), REQUEST_TIMEOUT_SECONDS)
            except ProviderSettingsError:
                raise
            except Exception:
                raise ProviderSettingsError("invalid_request") from None
            value = await asyncio.to_thread(settings.put, payload)
        return JSONResponse(value, headers={"Cache-Control": "no-store"})
    except ProviderSettingsError as exc:
        return JSONResponse({"detail": str(exc), "code": exc.code}, status_code=exc.status_code, headers={"Cache-Control": "no-store"})
