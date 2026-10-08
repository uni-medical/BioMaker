"""Desktop metadata-only dataset checks with bounded worker ownership."""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from dataset_probe import CODES, WARNINGS, HARD_SECONDS, MAX_INPUT_BYTES, MAX_OUTPUT_BYTES, MAX_NUMBER, MAX_ENTRIES, MAX_NAME, MAX_EXTENSION, report, validate_payload

PROBE = Path(__file__).with_name("dataset_probe.py")


def result(value, started):
    return {**value, "checked_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "elapsed_ms": min(MAX_NUMBER, max(0, round((time.monotonic() - started) * 1000)))}


def safe_report(value):
    if not isinstance(value, dict) or set(value) != set(report()):
        return None
    if value.get("code") not in CODES or value.get("status") != ("ok" if value.get("code") == "ok" else "error") or not isinstance(value.get("complete"), bool):
        return None
    for field in ("files", "directories", "symlinks", "other", "bytes"):
        if type(value[field]) is not int or not 0 <= value[field] <= (MAX_NUMBER if field == "bytes" else MAX_ENTRIES):
            return None
    if not isinstance(value["warnings"], list) or len(value["warnings"]) > len(WARNINGS) or any(not isinstance(code, str) or code not in WARNINGS for code in value["warnings"]):
        return None
    if not isinstance(value["extensions"], list) or len(value["extensions"]) > 20:
        return None
    for item in value["extensions"]:
        if not isinstance(item, dict) or set(item) != {"extension", "count"} or not isinstance(item["extension"], str) or len(item["extension"]) > MAX_EXTENSION or type(item["count"]) is not int or not 0 <= item["count"] <= MAX_ENTRIES:
            return None
    if not isinstance(value["entries"], list) or len(value["entries"]) > 10:
        return None
    for item in value["entries"]:
        if not isinstance(item, dict) or set(item) != {"name", "kind", "bytes"} or not isinstance(item["name"], str) or len(item["name"]) > MAX_NAME or any(char in item["name"] for char in ("/", "\x00")) or item["kind"] not in {"file", "directory", "symlink", "other"}:
            return None
        if item["bytes"] is not None and (type(item["bytes"]) is not int or not 0 <= item["bytes"] <= MAX_NUMBER):
            return None
    return value


def decode_output(output, *, timeout=False):
    if not isinstance(output, bytes) or len(output) > MAX_OUTPUT_BYTES:
        return report("internal")
    checkpoint = None
    final = None
    for line in output.splitlines()[:3]:
        try:
            packet = json.loads(line)
            value = safe_report(packet.get("report"))
            if value is None:
                continue
            if packet.get("kind") == "checkpoint" and value["status"] == "ok":
                checkpoint = value
            elif packet.get("kind") == "final":
                final = value
        except (ValueError, TypeError, AttributeError):
            continue
    if final is not None:
        return final
    if timeout and checkpoint is not None:
        checkpoint["complete"] = False
        checkpoint["warnings"] = sorted(set(checkpoint["warnings"]) | {"hard_timeout"})
        return checkpoint
    return report("timeout" if timeout else "internal")


class DatasetInspector:
    def __init__(self, workspace: Path, *, timeout=HARD_SECONDS):
        self.workspace = str(workspace)
        self.timeout = min(max(timeout, 0.05), HARD_SECONDS)
        self._lock = threading.Lock()
        self._process = None
        self._busy = False
        self._in_check = False
        self._closed = False

    @staticmethod
    def _stop(process):
        try:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=0.5)
            return True
        except (OSError, subprocess.TimeoutExpired):
            return process.poll() is not None

    @staticmethod
    def _close_pipes(process):
        for pipe in (process.stdin, process.stdout):
            if pipe is not None:
                try:
                    pipe.close()
                except OSError:
                    pass

    def _reap_locked(self):
        if not self._in_check and self._process is not None and self._process.poll() is not None:
            self._close_pipes(self._process)
            self._process = None
            self._busy = False

    def inspect(self, payload, *, mode="overview", started=None):
        started = time.monotonic() if started is None else started
        with self._lock:
            self._reap_locked()
            if self._closed:
                return result(report("internal"), started)
            if self._busy:
                return result(report("busy"), started)
            self._busy = self._in_check = True
        process = None
        try:
            invalid = validate_payload(payload, mode)
            if invalid:
                return result(report(invalid), started)
            encoded = json.dumps({"payload": payload, "mode": mode, "workspace": self.workspace}, ensure_ascii=False).encode("utf-8")
            if len(encoded) > MAX_INPUT_BYTES:
                return result(report("invalid_request"), started)
            remaining = self.timeout - (time.monotonic() - started)
            if remaining <= 0:
                return result(report("timeout"), started)
            environment = {key: os.environ[key] for key in ("SYSTEMROOT", "WINDIR") if key in os.environ}
            with self._lock:
                if self._closed:
                    return result(report("internal"), started)
                process = subprocess.Popen([sys.executable, "-I", "-B", str(PROBE), str(os.getpid()), str(remaining)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=environment, close_fds=True)
                self._process = process
            try:
                output, _ = process.communicate(encoded, timeout=max(0.01, self.timeout - (time.monotonic() - started)))
                return result(decode_output(output, timeout=process.returncode == 71) if process.returncode in {0, 71} else report("internal"), started)
            except subprocess.TimeoutExpired as error:
                output = error.output or b""
                if self._stop(process):
                    output, _ = process.communicate(timeout=0.5)
                return result(decode_output(output, timeout=True), started)
        except Exception:
            return result(report("internal"), started)
        finally:
            stopped = process is None or self._stop(process)
            if process is not None and stopped:
                self._close_pipes(process)
            with self._lock:
                self._in_check = False
                if stopped:
                    self._process = None
                    self._busy = False

    def close(self):
        with self._lock:
            self._closed = True
            process = self._process
        if process is not None:
            self._stop(process)
        with self._lock:
            self._reap_locked()


async def inspect_request(request, inspector):
    started = time.monotonic()

    async def read_payload():
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > MAX_INPUT_BYTES:
                raise ValueError()
            body.extend(chunk)
        return json.loads(body)

    try:
        payload = await asyncio.wait_for(read_payload(), timeout=min(1.0, inspector.timeout))
    except asyncio.TimeoutError:
        return result(report("timeout"), started)
    except Exception:
        return result(report("invalid_request"), started)
    return await asyncio.to_thread(inspector.inspect, payload, started=started)


ADMISSION_MESSAGES = {
    "empty": "所选文件夹为空，请选择包含数据的目录。 / The selected folder is empty; choose a folder containing data.",
    "not_found": "数据集目录不存在，请重新选择。 / The dataset folder was not found; select it again.",
    "not_directory": "数据集路径不是可用目录，请重新选择。 / The dataset path is not a usable directory; select it again.",
    "permission": "无法访问数据集目录，请检查文件夹权限。 / The dataset folder cannot be accessed; check folder permissions.",
    "timeout": "未能及时确认数据集目录可访问，请重试。 / Dataset access could not be checked within the time limit; retry.",
    "busy": "数据集检查正在进行，请稍后重试。 / A dataset check is already running; retry shortly.",
    "invalid_request": "数据集路径无效，请重新选择。 / The dataset path is invalid; select it again.",
    "unauthorized": "请重新选择数据集目录以授权访问。 / Select the dataset folder again to grant access.",
    "internal": "无法完成数据集访问检查，请重试。 / Dataset access could not be checked; retry.",
}
