from __future__ import annotations

import concurrent.futures
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import psutil
import pytest

from test_desktop_service import AUTH, SERVICE, client_for, config, create_desktop_app
import provider_check
from provider_check import CODES, DEFAULT_BASE_URL, ProviderChecker, normalize_base_url, validated_payload
from provider_probe import COMPLETION_TOKENS, MAX_RESPONSE_BYTES, PROMPT

ROUTE = "/api/desktop/provider/check"
KEY = "sk-synthetic-fixture-key"
GOOD = {"choices": [{"message": {"role": "assistant", "content": "Fixture ready"}}]}


@pytest.fixture
def provider():
    class State:
        status = 200
        body = json.dumps(GOOD).encode()
        headers = {}
        mode = "normal"
        requests = []
        received = threading.Event()
        release = threading.Event()

    state = State()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            state.requests.append({"path": self.path, "headers": dict(self.headers), "body": json.loads(raw)})
            state.received.set()
            try:
                if state.mode == "stall":
                    state.release.wait(10)
                self.send_response(state.status)
                for key, value in state.headers.items():
                    self.send_header(key, value)
                self.end_headers()
                if state.mode == "trickle":
                    while not state.release.wait(0.015):
                        self.wfile.write(b" ")
                        self.wfile.flush()
                else:
                    self.wfile.write(state.body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    thread.start()
    state.base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        yield state
    finally:
        state.release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


def payload(provider, **updates):
    return {"base_url": provider.base_url, "api_key": KEY, "model": "fixture-model", **updates}


def assert_result(value, code):
    assert set(value) == {"status", "code", "elapsed_ms", "checked_at"}
    assert value["code"] == code
    assert code in CODES
    assert value["status"] == ("ok" if code == "ok" else "error")
    assert isinstance(value["elapsed_ms"], int) and value["elapsed_ms"] >= 0
    assert datetime.fromisoformat(value["checked_at"].replace("Z", "+00:00")).utcoffset().total_seconds() == 0


def test_real_request_works_without_engine_and_changes_no_configuration(config, provider, monkeypatch):
    config.env_file.write_text(f'OPENAI_BASE_URL="{provider.base_url}/v1"\nOPENAI_API_KEY="{KEY}"\nBIOMAKER_MAIN_MODEL="saved-model"\n')
    config.workspace.mkdir(exist_ok=True)
    sentinel = config.workspace / "private-dataset.txt"
    sentinel.write_text("This must never enter a check request.")
    monkeypatch.setenv("OPENAI_API_KEY", "inherited-secret-must-not-win")
    app = create_desktop_app(config)
    with client_for(app) as client:
        before = (config.env_file.read_bytes(), config.env_file.stat().st_mtime_ns, sentinel.read_bytes(), app.state.run_store.list_records("runs"))
        assert not app.state.desktop_engine.snapshot()["ready"]
        response = client.post(ROUTE, headers=AUTH, json={"base_url": provider.base_url + "/v1/", "model": "chosen-model"})
        assert response.status_code == 200
        assert_result(response.json(), "ok")
        assert KEY not in response.text and provider.base_url not in response.text
        assert before == (config.env_file.read_bytes(), config.env_file.stat().st_mtime_ns, sentinel.read_bytes(), app.state.run_store.list_records("runs"))
    assert len(provider.requests) == 1
    request = provider.requests[0]
    assert request["path"] == "/v1/chat/completions"
    assert request["headers"]["Authorization"] == f"Bearer {KEY}"
    assert request["body"] == {"model": "chosen-model", "messages": [{"role": "user", "content": PROMPT}], "max_completion_tokens": COMPLETION_TOKENS, "stream": False}


@pytest.mark.parametrize("path,expected", [("", "/chat/completions"), ("/v1/", "/v1/chat/completions"), ("/Custom/v2", "/Custom/v2/chat/completions"), ("/v1//", "/v1//chat/completions")])
def test_request_preserves_explicit_base_path(config, provider, path, expected):
    checker = ProviderChecker(config.env_file)
    assert_result(checker.check(payload(provider, base_url=provider.base_url + path)), "ok")
    assert provider.requests[0]["path"] == expected
    checker.close()


@pytest.mark.parametrize("value,expected", [(None, DEFAULT_BASE_URL), ("  ", DEFAULT_BASE_URL), ("HTTPS://Example.COM:443/v1/", "https://example.com/v1"), ("http://Example.COM:80", "http://example.com"), ("https://example.com/Case", "https://example.com/Case"), ("http://[::1]:8080/v1", "http://[::1]:8080/v1")])
def test_url_normalization(value, expected):
    assert normalize_base_url(value) == expected


@pytest.mark.parametrize("base", ["ftp://example.com", "example.com", "https://user:secret@example.com", "https://example.com?api_key=secret", "https://example.com/#secret", "https://example.com/%zz", "https://example.com/a/../b", "https://bad host/v1", "https://example.com:0", "https://example.com:65536", "https://example.com:abc", "https://example.com\\@evil.test", "\n", 123, {}])
def test_invalid_urls_are_fixed_codes(base):
    with pytest.raises(ValueError, match="^invalid_url$"):
        normalize_base_url(base)


@pytest.mark.parametrize("base,use_saved,key,code", [
    ("https://example.com/v1", True, None, "saved"),
    ("HTTPS://EXAMPLE.COM:443/v1/", True, None, "saved"),
    ("https://example.com/v2", True, None, "missing_key"),
    ("https://example.com/V1", True, None, "missing_key"),
    ("https://example.com", True, None, "missing_key"),
    ("http://example.com/v1", True, None, "missing_key"),
    ("https://other.example/v1", True, None, "missing_key"),
    (None, True, None, "missing_key"),
    ("", True, None, "missing_key"),
    ("https://example.com/v1", False, None, "missing_key"),
    ("https://example.com/v1", False, "typed", "typed"),
    ("https://other.example/v1", True, "typed", "typed"),
    ("https://example.com/v1", True, "typed", "typed"),
])
def test_saved_key_requires_full_normalized_base(base, use_saved, key, code):
    saved = {"OPENAI_BASE_URL": "https://example.com/v1", "OPENAI_API_KEY": "saved"}
    request = {"base_url": base, "api_key": key, "model": "m", "use_saved_key": use_saved}
    if code == "missing_key":
        with pytest.raises(ValueError, match="^missing_key$"):
            validated_payload(request, saved)
    else:
        assert validated_payload(request, saved)["api_key"] == code


def test_default_provider_saved_key_and_invalid_saved_base():
    assert validated_payload({"model": "m"}, {"OPENAI_API_KEY": "saved"}) == {"base_url": DEFAULT_BASE_URL, "model": "m", "api_key": "saved"}
    with pytest.raises(ValueError, match="^missing_key$"):
        validated_payload({"model": "m"}, {"OPENAI_BASE_URL": "broken", "OPENAI_API_KEY": "saved"})


@pytest.mark.parametrize("updates,code", [
    ({"api_key": None, "use_saved_key": False}, "missing_key"),
    ({"api_key": "evil\r\nHeader: value"}, "invalid_key"),
    ({"api_key": "\n"}, "invalid_key"),
    ({"api_key": {"secret": "sensitive"}}, "invalid_request"),
    ({"api_key": "a" * 8193}, "invalid_key"),
    ({"api_key": "中文"}, "invalid_key"),
    ({"model": ""}, "missing_model"),
    ({"model": "a" * 501}, "missing_model"),
    ({"model": 123}, "invalid_request"),
    ({"model": "bad\nmodel"}, "missing_model"),
    ({"use_saved_key": "false"}, "invalid_request"),
    ({"unknown": KEY}, "invalid_request"),
])
def test_invalid_input_never_reaches_provider(config, provider, updates, code):
    with client_for(create_desktop_app(config)) as client:
        response = client.post(ROUTE, headers=AUTH, json=payload(provider, **updates))
        assert response.status_code == 200
        assert_result(response.json(), code)
        assert KEY not in response.text and "sensitive" not in response.text
    assert not provider.requests


@pytest.mark.parametrize("raw", [b'{"api_key":"secret-leak",', b'["secret-leak"]', b'null', b'{"api_key":"secret-leak"}' + b" " * 17000])
def test_invalid_json_and_oversize_body_do_not_reflect_keys(config, raw):
    with client_for(create_desktop_app(config)) as client:
        response = client.post(ROUTE, headers={**AUTH, "Content-Type": "application/json"}, content=raw)
        assert response.status_code == 200
        assert_result(response.json(), "invalid_request")
        assert "secret-leak" not in response.text


def test_endpoint_requires_desktop_auth(config, provider):
    with client_for(create_desktop_app(config)) as client:
        assert client.post(ROUTE, json=payload(provider)).status_code == 401
        assert client.post(ROUTE, headers={**AUTH, "Origin": "https://evil.example"}, json=payload(provider)).status_code == 403
    assert not provider.requests


@pytest.mark.parametrize("status,code", [(401, "auth"), (403, "auth"), (429, "rate_limit"), (400, "request_rejected"), (405, "request_rejected"), (422, "request_rejected"), (500, "unavailable"), (503, "unavailable"), (404, "unavailable"), (408, "timeout"), (504, "timeout")])
def test_provider_errors_are_sanitized_and_never_retried(config, provider, status, code):
    provider.status = status
    provider.body = b'provider response contains secret-leak and key'
    checker = ProviderChecker(config.env_file)
    value = checker.check(payload(provider))
    assert_result(value, code)
    assert "secret-leak" not in json.dumps(value)
    assert len(provider.requests) == 1
    checker.close()


def test_model_error_requires_provider_model_code(config, provider):
    provider.status = 404
    provider.body = json.dumps({"error": {"code": "model_not_found", "message": KEY}}).encode()
    checker = ProviderChecker(config.env_file)
    assert_result(checker.check(payload(provider)), "model")
    checker.close()


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_redirect_never_forwards_request(config, provider, status):
    provider.status = status
    provider.headers = {"Location": provider.base_url + "/credential-trap?secret=private"}
    checker = ProviderChecker(config.env_file)
    assert_result(checker.check(payload(provider)), "redirect")
    assert len(provider.requests) == 1
    assert provider.requests[0]["path"] == "/chat/completions"
    checker.close()


@pytest.mark.parametrize("body,code", [
    (b"not-json with secret-leak", "invalid_response"),
    (b"\xff", "invalid_response"),
    (b"[]", "invalid_response"),
    (b'{"choices":[]}', "invalid_response"),
    (b'{"choices":[{"message":{"role":"user","content":"text"}}]}', "invalid_response"),
    (b'{"choices":[{"message":{"role":"assistant","content":" "}}]}', "invalid_response"),
    (b'{"choices":[{"finish_reason":"length","message":{"role":"assistant","content":null,"reasoning_content":"secret-leak"}}]}', "invalid_response"),
    (b'{"choices":[{"message":{"role":"assistant","content":[{"text":"text"}]}}]}', "invalid_response"),
    (b"a" * (MAX_RESPONSE_BYTES + 1), "response_too_large"),
])
def test_response_requires_usable_assistant_text(config, provider, body, code):
    provider.body = body
    checker = ProviderChecker(config.env_file)
    value = checker.check(payload(provider))
    assert_result(value, code)
    assert "secret-leak" not in json.dumps(value)
    checker.close()


def test_declared_oversize_and_encoded_body(config, provider):
    checker = ProviderChecker(config.env_file)
    provider.headers = {"Content-Length": str(MAX_RESPONSE_BYTES + 1)}
    assert_result(checker.check(payload(provider)), "response_too_large")
    provider.headers = {"Content-Encoding": "gzip"}
    assert_result(checker.check(payload(provider)), "invalid_response")
    checker.close()


def test_network_failure(config, provider):
    checker = ProviderChecker(config.env_file)
    assert_result(checker.check(payload(provider, base_url="http://127.0.0.1:1")), "network")
    checker.close()


@pytest.mark.parametrize("mode", ["stall", "trickle"])
def test_hard_timeout_reaps_child_during_provider_io(config, provider, mode, monkeypatch):
    provider.mode = mode
    checker = ProviderChecker(config.env_file, timeout=0.45)
    children = []
    real_popen = subprocess.Popen

    def capture(*args, **kwargs):
        child = real_popen(*args, **kwargs)
        children.append(child)
        assert KEY not in repr(args)
        assert KEY not in repr(kwargs["env"])
        return child

    monkeypatch.setattr(provider_check.subprocess, "Popen", capture)
    started = time.monotonic()
    assert_result(checker.check(payload(provider)), "timeout")
    assert time.monotonic() - started < 2
    assert provider.received.is_set()
    assert children[0].poll() is not None
    assert checker._process is None
    assert children[0].stdin.closed and children[0].stdout.closed
    provider.release.set()
    checker.close()


def test_concurrent_request_is_busy_and_close_reaps_io_worker(config, provider):
    provider.mode = "trickle"
    checker = ProviderChecker(config.env_file, timeout=5)
    with concurrent.futures.ThreadPoolExecutor() as pool:
        first = pool.submit(checker.check, payload(provider))
        assert provider.received.wait(2)
        process = checker._process
        assert_result(checker.check(payload(provider)), "busy")
        checker.close()
        assert process.poll() is not None
        assert_result(first.result(timeout=2), "internal")
    assert_result(checker.check(payload(provider)), "internal")
    assert len(provider.requests) == 1


def test_lifespan_closes_checker(config):
    app = create_desktop_app(config)
    with client_for(app):
        assert not app.state.provider_checker._closed
    assert app.state.provider_checker._closed


@pytest.mark.parametrize("failure", ["signal", "wait"])
def test_cleanup_failure_retains_busy_handle_until_confirmed_exit(config, provider, monkeypatch, failure):
    provider.mode = "trickle"
    checker = ProviderChecker(config.env_file, timeout=0.2)
    real_popen = subprocess.Popen
    children = []

    def spawn(*args, **kwargs):
        # Give the independent watchdog a later deadline than communicate so
        # failed OS cleanup is observable while the worker is still doing I/O.
        command = list(args[0])
        if not children:
            command[-1] = "0.75"
        process = real_popen(command, **kwargs)
        if not children:
            if failure == "signal":
                def denied():
                    raise PermissionError("synthetic denied signal with secret-leak")
                monkeypatch.setattr(process, "kill", denied)
            else:
                monkeypatch.setattr(process, "kill", lambda: None)
                def wait_failed(*_args, **_kwargs):
                    raise subprocess.TimeoutExpired("secret-leak", 1)
                monkeypatch.setattr(process, "wait", wait_failed)
        children.append(process)
        return process

    monkeypatch.setattr(provider_check.subprocess, "Popen", spawn)
    value = checker.check(payload(provider))
    assert_result(value, "timeout")
    assert "secret-leak" not in json.dumps(value)
    assert provider.received.is_set()
    assert checker._process is children[0] and checker._busy
    assert_result(checker.check(payload(provider)), "busy")
    deadline = time.monotonic() + 2
    while children[0].poll() is None and time.monotonic() < deadline:
        time.sleep(0.025)
    assert children[0].poll() == 71
    provider.mode = "normal"
    provider.release.set()
    assert_result(checker.check(payload(provider)), "ok")
    assert children[0].stdin.closed and children[0].stdout.closed
    assert checker._process is None and not checker._busy
    checker.close()


def test_close_signal_failure_does_not_skip_other_lifespan_cleanup(config, provider, monkeypatch):
    provider.mode = "trickle"
    app = create_desktop_app(config)
    checker = app.state.provider_checker
    checker.timeout = 0.5
    cleanup = []
    for name in ("history_importer", "run_metadata", "desktop_engine"):
        component = getattr(app.state, name)
        original = component.close
        def close(original=original, name=name):
            original()
            cleanup.append(name)
        monkeypatch.setattr(component, "close", close)
    with concurrent.futures.ThreadPoolExecutor() as pool:
        with client_for(app):
            future = pool.submit(checker.check, payload(provider))
            assert provider.received.wait(2)
            process = checker._process
            def denied():
                raise PermissionError("synthetic cleanup denial")
            monkeypatch.setattr(process, "kill", denied)
        assert cleanup == ["history_importer", "run_metadata", "desktop_engine"]
        assert_result(future.result(timeout=2), "timeout")
        deadline = time.monotonic() + 2
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.025)
        assert process.poll() == 71
        checker.close()
        assert checker._process is None


def test_worker_has_own_deadline_even_without_parent_communicate_timeout(provider):
    provider.mode = "trickle"
    child = subprocess.Popen([sys.executable, "-I", "-B", str(provider_check.PROBE), str(os.getpid()), "0.4"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = child.communicate(json.dumps(payload(provider)).encode(), timeout=2)
    assert child.returncode == 71
    assert output == error == b""
    assert provider.received.is_set()


@pytest.mark.skipif(os.name != "posix", reason="POSIX parent loss contract")
def test_worker_exits_when_owner_abruptly_exits_during_io(config, provider):
    provider.mode = "trickle"
    owner_script = """
import json, os, sys, threading, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, sys.argv[2])
from provider_check import ProviderChecker
checker = ProviderChecker(Path(sys.argv[3]), timeout=20)
payload = json.loads(sys.stdin.readline())
threading.Thread(target=checker.check, args=(payload,), daemon=True).start()
deadline = time.monotonic() + 5
while checker._process is None and time.monotonic() < deadline:
    time.sleep(0.01)
print(checker._process.pid, flush=True)
sys.stdin.readline()
os._exit(0)
"""
    owner = subprocess.Popen([sys.executable, "-I", "-B", "-c", owner_script, str(SERVICE), str(SERVICE.parents[1] / "webui"), str(config.env_file)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    child_pid = None
    try:
        owner.stdin.write(json.dumps(payload(provider)) + "\n")
        owner.stdin.flush()
        child_pid = int(owner.stdout.readline())
        assert provider.received.wait(2)
        owner.stdin.write("exit\n")
        owner.stdin.flush()
        owner.wait(timeout=2)
        deadline = time.monotonic() + 3
        while psutil.pid_exists(child_pid) and psutil.Process(child_pid).status() != psutil.STATUS_ZOMBIE and time.monotonic() < deadline:
            time.sleep(0.025)
        assert not psutil.pid_exists(child_pid) or psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE
        assert owner.stderr.read() == ""
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait(timeout=2)
        for pipe in (owner.stdin, owner.stdout, owner.stderr):
            pipe.close()
        if child_pid and psutil.pid_exists(child_pid):
            try:
                psutil.Process(child_pid).kill()
            except psutil.NoSuchProcess:
                pass
