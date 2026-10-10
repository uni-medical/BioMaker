"""Isolated stdlib-only, one-request provider probe. Never print provider data."""

from __future__ import annotations

import http.client
import json
import os
import socket
import sys
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

MAX_RESPONSE_BYTES = 65_536
MAX_INPUT_BYTES = 16_384
MAX_LIFETIME_SECONDS = 25.0
PROMPT = "Reply with exactly OK."
COMPLETION_TOKENS = 64


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def watch_owner(parent_pid: int, lifetime: float) -> None:
    deadline = time.monotonic() + min(max(lifetime, 0.05), MAX_LIFETIME_SECONDS)
    while True:
        if os.name == "posix" and os.getppid() != parent_pid:
            os._exit(70)
        if time.monotonic() >= deadline:
            os._exit(71)
        time.sleep(0.05)


def check(payload: dict, timeout: float) -> str:
    body = json.dumps({
        "model": payload["model"],
        "messages": [{"role": "user", "content": PROMPT}],
        "max_completion_tokens": COMPLETION_TOKENS,
        "stream": False,
    }).encode("utf-8")
    request = Request(
        payload["base_url"] + "/chat/completions",
        data=body,
        headers={
            "Authorization": "Bearer " + payload["api_key"],
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Accept-Encoding": "identity",
            "User-Agent": "BioMaker-Desktop/provider-check",
        },
        method="POST",
    )
    # Environment proxies and .netrc must not change where this key is sent.
    opener = build_opener(ProxyHandler({}), NoRedirect())
    try:
        with opener.open(request, timeout=min(timeout, 20.0)) as response:
            if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                return "invalid_response"
            declared = response.headers.get("Content-Length")
            if declared and declared.isdigit() and int(declared) > MAX_RESPONSE_BYTES:
                return "response_too_large"
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except HTTPError as error:
        try:
            if 300 <= error.code < 400:
                return "redirect"
            if error.code in {401, 403}:
                return "auth"
            if error.code == 429:
                return "rate_limit"
            if error.code in {408, 504}:
                return "timeout"
            if error.code in {400, 405, 422}:
                return "request_rejected"
            if error.code == 404:
                # An absent route is not evidence that a model is unavailable.
                raw_error = error.read(MAX_RESPONSE_BYTES + 1)
                if len(raw_error) > MAX_RESPONSE_BYTES:
                    return "response_too_large"
                try:
                    failure = json.loads(raw_error)
                    detail = failure.get("error") if isinstance(failure, dict) else None
                    error_code = detail.get("code") if isinstance(detail, dict) else None
                    if isinstance(error_code, str) and error_code in {"model_not_found", "model_not_available"}:
                        return "model"
                except (ValueError, UnicodeError):
                    pass
            return "unavailable"
        except (TimeoutError, socket.timeout):
            return "timeout"
        except (OSError, http.client.HTTPException):
            return "network"
        finally:
            error.close()
    except (TimeoutError, socket.timeout):
        return "timeout"
    except URLError as error:
        return "timeout" if isinstance(error.reason, (TimeoutError, socket.timeout)) else "network"
    except (OSError, http.client.HTTPException):
        return "network"

    if len(raw) > MAX_RESPONSE_BYTES:
        return "response_too_large"
    try:
        result = json.loads(raw)
    except (ValueError, UnicodeError):
        return "invalid_response"
    choices = result.get("choices") if isinstance(result, dict) else None
    first = choices[0] if isinstance(choices, list) and choices else None
    message = first.get("message") if isinstance(first, dict) else None
    if not isinstance(message, dict) or message.get("role") != "assistant":
        return "invalid_response"
    content = message.get("content")
    return "ok" if isinstance(content, str) and content.strip() else "invalid_response"


def main() -> None:
    # Start before reading stdin, DNS resolution, TLS, or any provider I/O.
    try:
        parent_pid = int(sys.argv[1])
        timeout = min(float(sys.argv[2]), MAX_LIFETIME_SECONDS)
    except (IndexError, ValueError):
        os._exit(72)
    threading.Thread(target=watch_owner, args=(parent_pid, timeout), daemon=True).start()
    code = "internal"
    try:
        raw = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
        if len(raw) <= MAX_INPUT_BYTES:
            payload = json.loads(raw)
            code = check(payload, timeout)
    except BaseException:
        # Credential-bearing requests and provider bodies never enter logs.
        pass
    os.write(sys.stdout.fileno(), code.encode("ascii") + b"\n")


if __name__ == "__main__":
    main()
