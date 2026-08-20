from __future__ import annotations

import json
from urllib.error import HTTPError

import webui.model_discovery as model_discovery


class FakeResponse:
    def __init__(self, payload):
        self.body = (
            payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        )

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _limit):
        return self.body


class FakeOpener:
    def __init__(self, result):
        self.result = result
        self.request = None
        self.timeout = None

    def open(self, request, timeout):
        self.request = request
        self.timeout = timeout
        if isinstance(self.result, Exception):
            raise self.result
        return FakeResponse(self.result)


def test_discovery_uses_openai_models_endpoint_without_exposing_key(monkeypatch):
    opener = FakeOpener(
        {"object": "list", "data": [{"id": "model-b"}, {"id": "model-a"}]}
    )
    monkeypatch.setattr(model_discovery, "build_opener", lambda *_args: opener)

    secret = "provider-secret"
    result = model_discovery.discover_models(
        "http://127.0.0.1:11434/v1/", secret, timeout=1.25
    )

    assert result == {
        "status": "ok",
        "models": ["model-b", "model-a"],
        "base_url": "http://127.0.0.1:11434/v1",
        "detail": None,
    }
    assert opener.request.full_url == "http://127.0.0.1:11434/v1/models"
    assert opener.request.get_header("Authorization") == f"Bearer {secret}"
    assert opener.timeout == 1.25
    assert secret not in json.dumps(result)


def test_discovery_handles_unconfigured_unsupported_and_timeout(monkeypatch):
    assert model_discovery.discover_models(None, None)["status"] == "unconfigured"
    assert (
        model_discovery.discover_models("file:///tmp/provider", None)["status"]
        == "error"
    )

    unsupported = FakeOpener(
        HTTPError("https://example.test/v1/models", 404, "missing", {}, None)
    )
    monkeypatch.setattr(model_discovery, "build_opener", lambda *_args: unsupported)
    result = model_discovery.discover_models("https://example.test/v1", "secret")
    assert result["status"] == "unsupported"
    assert "secret" not in json.dumps(result)

    timed_out = FakeOpener(TimeoutError())
    monkeypatch.setattr(model_discovery, "build_opener", lambda *_args: timed_out)
    result = model_discovery.discover_models("https://example.test", "secret")
    assert result["status"] == "error"
    assert "time limit" in result["detail"]

    for payload in (b"not-json", {"data": []}):
        invalid = FakeOpener(payload)
        monkeypatch.setattr(model_discovery, "build_opener", lambda *_args: invalid)
        result = model_discovery.discover_models("https://example.test/v1", None)
        assert result["status"] == "unsupported"
