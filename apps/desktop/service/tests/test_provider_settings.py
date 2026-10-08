from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

import pytest
from starlette.requests import Request

from test_desktop_service import AUTH, client_for, config, create_desktop_app
import provider_settings
from provider_settings import ProviderSettings, ProviderSettingsError, provider_settings_request

OLD_KEY = "sk-synthetic-old-provider-secret"
NEW_KEY = "synthetic-new-provider-secret"
OLD_BASE = "https://old.example.test/v1"
NEW_BASE = "https://new.example.test/v1"
SENTINEL = "private-settings-sentinel-never-return"


def seed(config, *, base=OLD_BASE):
    original = (f'# preserved comment\nOPENAI_BASE_URL="{base}"\n'
                f'OPENAI_API_KEY="{OLD_KEY}"\nBIOMAKER_MAIN_MODEL="original-model"\n'
                'SERPER_API_KEY="other-synthetic-secret"\nUNMANAGED="untouched"\n')
    config.env_file.write_text(original)
    config.env_file.chmod(0o600)
    return original.encode()


def values(config):
    from webui.env_file import read_env_values
    return read_env_values(config.env_file)


def assert_error(response, code, status=409):
    assert response.status_code == status, response.text
    assert response.json()["code"] == code
    assert response.json()["detail"].startswith(f"[provider_settings:{code}]")
    assert response.headers["Cache-Control"] == "no-store"
    for secret in (OLD_KEY, NEW_KEY, SENTINEL, "other-synthetic-secret"):
        assert secret not in response.text


@pytest.mark.parametrize("suffix", ["", "/"])
def test_changed_provider_refusal_preserves_complete_file_and_has_no_route_bypass(config, suffix):
    original = seed(config)
    app = create_desktop_app(config)
    routes = [r for r in app.routes if getattr(r, "path", None) == "/api/env"]
    assert len([r for r in routes if "GET" in r.methods]) == 1
    assert len([r for r in routes if "PUT" in r.methods]) == 1
    with client_for(app) as client:
        initial = client.get("/api/env" + suffix, headers=AUTH)
        revision = initial.json()["settings_revision"]
        assert re.fullmatch(r"[0-9a-f]{64}", revision)
        response = client.put("/api/env" + suffix, headers=AUTH, json={
            "OPENAI_BASE_URL": NEW_BASE, "BIOMAKER_MAIN_MODEL": "must-not-save",
            "expected_revision": revision,
        }, follow_redirects=False)
        assert_error(response, "key_required")
        assert config.env_file.read_bytes() == original
        unchanged = client.get("/api/env", headers=AUTH).json()
        assert unchanged == initial.json()
        assert values(config)["OPENAI_API_KEY"] == OLD_KEY


@pytest.mark.parametrize("key", [NEW_KEY, None, "", "   ", "\t\n"])
def test_switch_requires_explicit_replacement_or_removal_and_preserves_other_settings(config, key):
    seed(config)
    store = ProviderSettings(config.env_file)
    before = store.get()
    response = store.put({"OPENAI_BASE_URL": NEW_BASE, "OPENAI_API_KEY": key,
                          "expected_revision": before["settings_revision"]})
    current = values(config)
    assert current["OPENAI_BASE_URL"] == NEW_BASE
    assert current.get("OPENAI_API_KEY") == (NEW_KEY if key == NEW_KEY else None)
    assert response["values"]["OPENAI_API_KEY"]["configured"] is (key == NEW_KEY)
    assert response["settings_revision"] != before["settings_revision"]
    assert response == store.get()
    assert current["BIOMAKER_MAIN_MODEL"] == "original-model"
    assert current["SERPER_API_KEY"] == "other-synthetic-secret"
    assert '# preserved comment\n' in config.env_file.read_text()
    assert 'UNMANAGED="untouched"' in config.env_file.read_text()
    assert os.stat(config.env_file).st_mode & 0o777 == 0o600
    assert not list(config.home.glob('*.tmp'))
    assert OLD_KEY not in json.dumps(response) and NEW_KEY not in json.dumps(response)


@pytest.mark.parametrize("saved,requested", [
    (OLD_BASE, "HTTPS://OLD.EXAMPLE.TEST:443/v1/"),
    ("https://api.openai.com/v1", None),
    ("", "https://api.openai.com:443/v1/"),
    ("http://old.example.test:80/v1", "http://OLD.example.test/v1/"),
    ("https://old.example.test/a%20b", "https://old.example.test/a%20b/"),
])
def test_equivalent_base_retains_key(config, saved, requested):
    seed(config, base=saved)
    store = ProviderSettings(config.env_file)
    assert store.put({"OPENAI_BASE_URL": requested})["values"]["OPENAI_API_KEY"]["configured"]
    assert values(config)["OPENAI_API_KEY"] == OLD_KEY


@pytest.mark.parametrize("requested", [
    "http://old.example.test/v1", "https://old.example.test", "https://old.example.test/v2",
    "https://old.example.test/V1", "https://old.example.test/v1//", "https://old.example.test:8443/v1",
])
def test_full_base_changes_do_not_rebind_saved_key(config, requested):
    original = seed(config)
    with pytest.raises(ProviderSettingsError, match="key_required"):
        ProviderSettings(config.env_file).put({"OPENAI_BASE_URL": requested})
    assert config.env_file.read_bytes() == original


def test_first_configuration_and_partial_updates_need_no_revision(config):
    store = ProviderSettings(config.env_file)
    missing = store.get()
    assert not missing["source"]["exists"]
    assert not config.env_file.exists()
    store.put({"OPENAI_BASE_URL": NEW_BASE})
    store.put({"OPENAI_API_KEY": NEW_KEY, "BIOMAKER_MAIN_MODEL": "fixture-model"})
    store.put({"BIOMAKER_MAIN_MODEL": "changed-model"})
    assert values(config) == {"OPENAI_BASE_URL": NEW_BASE, "OPENAI_API_KEY": NEW_KEY, "BIOMAKER_MAIN_MODEL": "changed-model"}
    assert config.env_file.stat().st_mode & 0o777 == 0o600
    assert store.put({}) == store.get()


def test_invalid_legacy_base_needs_explicit_key_decision_only_when_changing_base(config):
    original = seed(config, base="not a valid URL")
    store = ProviderSettings(config.env_file)
    with pytest.raises(ProviderSettingsError, match="key_required"):
        store.put({"OPENAI_BASE_URL": NEW_BASE})
    assert config.env_file.read_bytes() == original
    store.put({"BIOMAKER_MAIN_MODEL": "retained-provider-model"})
    store.put({"OPENAI_BASE_URL": NEW_BASE, "OPENAI_API_KEY": None})
    assert "OPENAI_API_KEY" not in values(config)


@pytest.mark.parametrize("payload,code", [
    ({"OPENAI_BASE_URL": "https://user:private-settings-sentinel-never-return@example.test"}, "invalid_url"),
    ({"OPENAI_BASE_URL": "https://example.test?private-settings-sentinel-never-return"}, "invalid_url"),
    ({"OPENAI_BASE_URL": "ftp://example.test"}, "invalid_url"),
    ({"OPENAI_BASE_URL": "https://example.test/../v1"}, "invalid_url"),
    ({"OPENAI_BASE_URL": "https://example.test/a b"}, "invalid_url"),
    ({"OPENAI_BASE_URL": "https://example.test/" + "x" * 4100}, "invalid_url"),
    ({"OPENAI_API_KEY": SENTINEL + "\ninside"}, "invalid_key"),
    ({"OPENAI_API_KEY": SENTINEL + " space"}, "invalid_key"),
    ({"OPENAI_API_KEY": "é" + SENTINEL}, "invalid_key"),
    ({"OPENAI_API_KEY": "x" * 8193}, "invalid_key"),
    ({"OPENAI_API_KEY": {SENTINEL: SENTINEL}}, "invalid_request"),
    ({"OPENAI_API_KEY": [SENTINEL]}, "invalid_request"),
    ({"OPENAI_API_KEY": 123}, "invalid_request"),
    ({"expected_revision": None}, "invalid_request"),
    ({"expected_revision": SENTINEL}, "invalid_request"),
    ({"unsupported": SENTINEL}, "invalid_request"),
    ([SENTINEL], "invalid_request"),
    (None, "invalid_request"),
])
def test_validation_never_echoes_payload_or_changes_file(config, payload, code, caplog):
    original = seed(config)
    with client_for(create_desktop_app(config)) as client:
        response = client.put("/api/env", headers=AUTH, json=payload)
    assert_error(response, code, 422)
    assert config.env_file.read_bytes() == original
    assert SENTINEL not in caplog.text and OLD_KEY not in caplog.text


@pytest.mark.parametrize("raw", [
    b'{"OPENAI_API_KEY":"private-settings-sentinel-never-return",',
    b'{"OPENAI_API_KEY":"x","OPENAI_API_KEY":"private-settings-sentinel-never-return"}',
    b'{"OPENAI_API_KEY":"private-settings-sentinel-never-return"}' + b" " * provider_settings.MAX_REQUEST_BYTES,
])
def test_raw_body_rejections_are_bounded_and_sanitized(config, raw):
    original = seed(config)
    with client_for(create_desktop_app(config)) as client:
        response = client.put("/api/env", headers=AUTH, content=raw)
    assert_error(response, "invalid_request", 422)
    assert config.env_file.read_bytes() == original


def test_partial_body_deadline_is_sanitized_and_saves_nothing(config, monkeypatch):
    monkeypatch.setattr(provider_settings, "REQUEST_TIMEOUT_SECONDS", 0.02)
    original = seed(config)
    async def stalled():
        await asyncio.sleep(10)
    request = Request({"type": "http", "method": "PUT", "path": "/api/env", "headers": []}, receive=stalled)
    response = asyncio.run(provider_settings_request(request, ProviderSettings(config.env_file)))
    assert response.status_code == 422
    assert json.loads(response.body)["code"] == "invalid_request"
    assert config.env_file.read_bytes() == original


def test_revision_conflict_external_edit_and_restart_preserve_unsaved_data(config):
    seed(config)
    store = ProviderSettings(config.env_file)
    first = store.get()["settings_revision"]
    raw = config.env_file.read_bytes()
    assert first != hashlib.sha256(raw).hexdigest()
    config.env_file.write_bytes(raw + b"# external change\n")
    externally_changed = config.env_file.read_bytes()
    with pytest.raises(ProviderSettingsError, match="conflict"):
        store.put({"BIOMAKER_MAIN_MODEL": "stale-model", "expected_revision": first})
    assert config.env_file.read_bytes() == externally_changed
    current = store.get()["settings_revision"]
    restarted = ProviderSettings(config.env_file)
    assert restarted.get()["settings_revision"] != current
    with pytest.raises(ProviderSettingsError, match="conflict"):
        restarted.put({"OPENAI_API_KEY": NEW_KEY, "expected_revision": current})
    assert config.env_file.read_bytes() == externally_changed


def test_revision_includes_absent_vs_existing_empty_and_noop_does_not_mutate(config):
    store = ProviderSettings(config.env_file)
    absent = store.get()
    assert store.put({"expected_revision": absent["settings_revision"]}) == absent
    assert not config.env_file.exists()
    config.env_file.touch()
    assert store.get()["settings_revision"] != absent["settings_revision"]
    with pytest.raises(ProviderSettingsError, match="conflict"):
        store.put({"OPENAI_API_KEY": NEW_KEY, "expected_revision": absent["settings_revision"]})


def test_api_revision_conflict_after_another_save_and_real_adapter_restart(config):
    seed(config)
    with client_for(create_desktop_app(config)) as client:
        baseline = client.get("/api/env", headers=AUTH).json()["settings_revision"]
        saved = client.put("/api/env", headers=AUTH, json={"BIOMAKER_MAIN_MODEL": "new-model", "expected_revision": baseline})
        assert saved.status_code == 200
        changed = config.env_file.read_bytes()
        assert_error(client.put("/api/env", headers=AUTH, json={"BIOMAKER_MAIN_MODEL": "stale-model", "expected_revision": baseline}), "conflict")
        current = saved.json()["settings_revision"]
    with client_for(create_desktop_app(config)) as restarted:
        assert_error(restarted.put("/api/env/", headers=AUTH, json={"OPENAI_API_KEY": NEW_KEY, "expected_revision": current}), "conflict")
        assert config.env_file.read_bytes() == changed
        latest = restarted.get("/api/env/", headers=AUTH).json()["settings_revision"]
        assert restarted.put("/api/env/", headers=AUTH, json={"OPENAI_BASE_URL": NEW_BASE, "OPENAI_API_KEY": NEW_KEY, "expected_revision": latest}).status_code == 200
        assert values(config)["OPENAI_API_KEY"] == NEW_KEY


def test_concurrent_compare_and_swap_allows_exactly_one_writer(config):
    seed(config)
    store = ProviderSettings(config.env_file)
    revision = store.get()["settings_revision"]
    barrier = threading.Barrier(2)
    def save(model):
        barrier.wait(timeout=2)
        try:
            return store.put({"BIOMAKER_MAIN_MODEL": model, "expected_revision": revision})
        except ProviderSettingsError as exc:
            return exc.code
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(save, ["model-one", "model-two"]))
    assert sum(isinstance(item, dict) for item in results) == 1
    assert results.count("conflict") == 1
    accepted = next(item for item in results if isinstance(item, dict))
    assert store.get() == accepted


def test_independent_store_writers_merge_disjoint_updates_without_lost_data(config, monkeypatch):
    seed(config)
    import webui.env_file as env_file
    original_update = env_file.update_env_file
    entered, release = threading.Event(), threading.Event()
    def delayed(path, updates):
        if "BIOMAKER_MAIN_MODEL" in updates:
            entered.set()
            assert release.wait(2)
        return original_update(path, updates)
    monkeypatch.setattr(env_file, "update_env_file", delayed)
    first, second = ProviderSettings(config.env_file), ProviderSettings(config.env_file)
    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(first.put, {"BIOMAKER_MAIN_MODEL": "concurrent-model"})
        assert entered.wait(2)
        b = pool.submit(second.put, {"JINA_API_KEY": "jina-fixture"})
        time.sleep(.05)
        assert not b.done()
        release.set()
        a.result(timeout=2)
        b.result(timeout=2)
    saved = values(config)
    assert saved["BIOMAKER_MAIN_MODEL"] == "concurrent-model"
    assert saved["JINA_API_KEY"] == "jina-fixture"


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-lock acceptance")
def test_separate_process_lock_contention_is_bounded_and_recoverable(config):
    seed(config)
    store = ProviderSettings(config.env_file, lock_timeout=.08)
    script = "import fcntl,os,sys; f=os.open(sys.argv[1],os.O_CREAT|os.O_RDWR,0o600); fcntl.flock(f,fcntl.LOCK_EX); print('locked',flush=True); sys.stdin.readline()"
    child = subprocess.Popen([sys.executable, "-I", "-B", "-c", script, str(store.lock_path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "locked"
        started = time.monotonic()
        with pytest.raises(ProviderSettingsError, match="busy"):
            store.put({"BIOMAKER_MAIN_MODEL": "must-not-save"})
        assert time.monotonic() - started < .8
        assert values(config)["BIOMAKER_MAIN_MODEL"] == "original-model"
    finally:
        child.communicate("release\n", timeout=2)
    assert store.put({"BIOMAKER_MAIN_MODEL": "after-lock"})["values"]["BIOMAKER_MAIN_MODEL"]["value"] == "after-lock"


def test_api_transaction_wait_does_not_block_health_endpoint(config, monkeypatch):
    seed(config)
    import webui.env_file as env_file
    original_update = env_file.update_env_file
    entered, release = threading.Event(), threading.Event()
    def delayed(path, updates):
        entered.set()
        assert release.wait(3)
        return original_update(path, updates)
    monkeypatch.setattr(env_file, "update_env_file", delayed)
    with client_for(create_desktop_app(config)) as client, ThreadPoolExecutor(max_workers=2) as pool:
        write = pool.submit(client.put, "/api/env", headers=AUTH, json={"BIOMAKER_MAIN_MODEL": "async-save"})
        try:
            assert entered.wait(2)
            health = pool.submit(client.get, "/api/health", headers=AUTH)
            assert health.result(timeout=1).status_code == 200
        finally:
            release.set()
        assert write.result(timeout=2).status_code == 200


@pytest.mark.parametrize("kind", ["config-symlink", "lock-symlink", "config-fifo", "lock-directory", "oversize", "invalid-utf8"])
def test_unsafe_or_unreadable_storage_returns_fixed_error_without_touching_target(config, kind):
    target = config.home / "sensitive-fixture"
    target.write_text(SENTINEL)
    store = ProviderSettings(config.env_file)
    if kind == "config-symlink": config.env_file.symlink_to(target)
    elif kind == "lock-symlink": store.lock_path.symlink_to(target)
    elif kind == "config-fifo": os.mkfifo(config.env_file)
    elif kind == "lock-directory": store.lock_path.mkdir()
    elif kind == "oversize": config.env_file.write_bytes(b"x" * (provider_settings.MAX_CONFIG_BYTES + 1))
    else: config.env_file.write_bytes(b"\xff\xfe" + SENTINEL.encode())
    with client_for(create_desktop_app(config)) as client:
        assert_error(client.get("/api/env", headers=AUTH), "io_error", 503)
        assert_error(client.put("/api/env", headers=AUTH, json={"OPENAI_API_KEY": NEW_KEY}), "io_error", 503)
    assert target.read_text() == SENTINEL


def test_write_error_is_sanitized_atomic_and_can_retry(config, monkeypatch):
    original = seed(config)
    store = ProviderSettings(config.env_file)
    replace = os.replace
    def fail(source, destination):
        if Path(destination) == config.env_file:
            raise PermissionError(SENTINEL)
        return replace(source, destination)
    monkeypatch.setattr(provider_settings.os, "replace", fail)
    with pytest.raises(ProviderSettingsError, match="io_error") as error:
        store.put({"OPENAI_API_KEY": NEW_KEY})
    assert SENTINEL not in str(error.value)
    assert config.env_file.read_bytes() == original
    assert not list(config.home.glob('.*.tmp'))
    monkeypatch.setattr(provider_settings.os, "replace", replace)
    store.put({"OPENAI_API_KEY": NEW_KEY})
    assert values(config)["OPENAI_API_KEY"] == NEW_KEY


def test_response_revision_and_mask_are_from_exact_committed_snapshot(config, monkeypatch):
    seed(config)
    store = ProviderSettings(config.env_file)
    replace = os.replace
    committed = []
    def later_external_write(source, destination):
        result = replace(source, destination)
        if Path(destination) == config.env_file:
            committed.append(config.env_file.read_bytes())
            config.env_file.write_text('BIOMAKER_MAIN_MODEL="external-after-save"\n')
        return result
    monkeypatch.setattr(provider_settings.os, "replace", later_external_write)
    result = store.put({"BIOMAKER_MAIN_MODEL": "our-committed-model"})
    assert result["values"]["BIOMAKER_MAIN_MODEL"]["value"] == "our-committed-model"
    assert result["settings_revision"] == store._revision(committed[0], True)
    assert store.get()["settings_revision"] != result["settings_revision"]


def test_refused_switch_cannot_enable_saved_key_at_new_provider(config, monkeypatch):
    original = seed(config)
    import webui.server as server
    import provider_check
    calls = []
    def discover(base, api_key):
        calls.append((base, api_key))
        return {"status": "missing_key" if api_key is None else "ok", "models": []}
    monkeypatch.setattr(server, "discover_models", discover)
    def no_worker(*args, **kwargs):
        raise AssertionError("A refused new provider must never launch a provider worker")
    app = create_desktop_app(config)
    with client_for(app) as client:
        assert_error(client.put("/api/env", headers=AUTH, json={"OPENAI_BASE_URL": NEW_BASE}), "key_required")
        discovered = client.post("/api/models/discover", headers=AUTH, json={"base_url": NEW_BASE})
        assert discovered.json()["status"] == "missing_key"
        assert calls == [(NEW_BASE, None)]
        monkeypatch.setattr(provider_check.subprocess, "Popen", no_worker)
        checked = client.post("/api/desktop/provider/check", headers=AUTH, json={"base_url": NEW_BASE, "model": "fixture"})
        assert checked.json()["code"] == "missing_key"
        assert config.env_file.read_bytes() == original


def test_replaced_routes_keep_desktop_auth_and_origin_gates(config):
    original = seed(config)
    with client_for(create_desktop_app(config)) as client:
        for path in ("/api/env", "/api/env/"):
            assert client.get(path).status_code == 401
            assert client.put(path, json={"OPENAI_API_KEY": NEW_KEY}).status_code == 401
            assert client.put(path, headers={**AUTH, "Origin": "https://untrusted.example"}, json={"OPENAI_API_KEY": NEW_KEY}).status_code == 403
    assert config.env_file.read_bytes() == original


def test_dotenv_interpolation_and_existing_owner_mode_are_preserved(config, monkeypatch):
    monkeypatch.setenv("QA_PROVIDER_MODEL", "resolved-model")
    config.env_file.write_text('BIOMAKER_MAIN_MODEL="${QA_PROVIDER_MODEL}"\n')
    config.env_file.chmod(0o640)
    store = ProviderSettings(config.env_file)
    assert store.get()["values"]["BIOMAKER_MAIN_MODEL"]["value"] == "resolved-model"
    store.put({"OPENAI_API_KEY": NEW_KEY})
    assert 'BIOMAKER_MAIN_MODEL="${QA_PROVIDER_MODEL}"' in config.env_file.read_text()
    assert config.env_file.stat().st_mode & 0o777 == 0o640
