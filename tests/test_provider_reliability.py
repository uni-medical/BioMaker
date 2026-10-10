from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

import biomaker.backend.backend_openai as backend_openai
from biomaker.backend.utils import (
    FunctionSpec,
    ProviderRequestError,
    backoff_create,
    sanitized_exception_metadata,
)
from biomaker.agents.base_agent import BaseAgent, CandidateGenerationError
from biomaker.utils.config import _load_cfg, _validate_exec_config


class TransientFailure(Exception):
    status_code = 500
    code = "server_error"
    request_id = "req-test-1"


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def monotonic(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


def test_bounded_retry_stops_after_two_attempts_without_raw_error() -> None:
    clock = FakeClock()
    events: list[tuple[str, dict]] = []
    calls = 0

    def create(**_kwargs):
        nonlocal calls
        calls += 1
        clock.value += 10
        raise TransientFailure("provider body with sk-secret-value")

    with pytest.raises(ProviderRequestError) as exc_info:
        backoff_create(
            create,
            (TransientFailure,),
            timeout=600,
            max_tries=2,
            max_elapsed_seconds=1250,
            event_callback=lambda event, payload: events.append((event, payload)),
            sleep_fn=clock.sleep,
            monotonic_fn=clock.monotonic,
        )

    assert calls == 2
    assert exc_info.value.__cause__ is None
    serialized = repr(events) + str(exc_info.value)
    assert "sk-secret-value" not in serialized
    assert [event for event, _payload in events] == [
        "provider_request_started",
        "provider_request_retry_scheduled",
        "provider_request_started",
        "provider_request_failed",
    ]


def test_retry_uses_remaining_deadline_as_second_timeout() -> None:
    clock = FakeClock()
    timeouts: list[float] = []

    def create(**kwargs):
        timeout = float(kwargs["timeout"])
        timeouts.append(timeout)
        clock.value += timeout
        raise TransientFailure("timeout")

    with pytest.raises(ProviderRequestError):
        backoff_create(
            create,
            (TransientFailure,),
            timeout=600,
            max_tries=2,
            max_elapsed_seconds=1000,
            sleep_fn=clock.sleep,
            monotonic_fn=clock.monotonic,
        )

    assert timeouts[0] == 600
    assert timeouts[1] == pytest.approx(398.5)


def test_transient_retry_can_succeed_and_callback_failure_is_nonfatal() -> None:
    clock = FakeClock()
    calls = 0

    def create(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TransientFailure("retry")
        return "ok"

    result = backoff_create(
        create,
        (TransientFailure,),
        max_tries=2,
        max_elapsed_seconds=1250,
        event_callback=lambda *_args: (_ for _ in ()).throw(OSError("event sink")),
        sleep_fn=clock.sleep,
        monotonic_fn=clock.monotonic,
    )

    assert result == "ok"
    assert calls == 2


def test_quota_failure_is_not_retried() -> None:
    clock = FakeClock()
    calls = 0

    class QuotaFailure(Exception):
        status_code = 429
        code = "insufficient_quota"

    def create(**_kwargs):
        nonlocal calls
        calls += 1
        raise QuotaFailure("private provider body")

    with pytest.raises(ProviderRequestError) as exc_info:
        backoff_create(
            create,
            (QuotaFailure,),
            max_tries=2,
            max_elapsed_seconds=1250,
            sleep_fn=clock.sleep,
            monotonic_fn=clock.monotonic,
        )

    assert calls == 1
    assert exc_info.value.metadata["failure_kind"] == "quota"


def test_exception_metadata_prioritizes_quota_and_rejects_unsafe_fields() -> None:
    exc = TransientFailure("do not record this body")
    exc.status_code = 429
    exc.code = "insufficient_quota\nsecret"
    exc.request_id = "unsafe request id"

    metadata = sanitized_exception_metadata(exc)

    assert metadata["failure_kind"] == "quota"
    assert metadata["error_code"] is None
    assert metadata["request_id"] is None
    assert "body" not in repr(metadata)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_backoff_rejects_non_finite_timeout_and_deadline(value: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        backoff_create(lambda **_kwargs: "ok", (), timeout=value)
    with pytest.raises(ValueError, match="finite and positive"):
        backoff_create(lambda **_kwargs: "ok", (), max_elapsed_seconds=value)


def test_openai_backend_passes_timeout_and_emits_usage_without_content() -> None:
    events: list[tuple[str, dict]] = []
    seen: dict = {}

    def create(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="answer"),
                    finish_reason="length",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=7,
                completion_tokens=11,
                completion_tokens_details={
                    "reasoning_tokens": 3,
                    "provider_private_detail": "must-not-be-recorded",
                },
            ),
            system_fingerprint="fp",
            model="deepseek-v4-flash",
            created=1,
            _request_id="req-safe-1",
        )

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    with (
        patch.object(backend_openai, "_setup_openai_client", return_value=None),
        patch.object(backend_openai, "_client", fake_client),
    ):
        output, _elapsed, in_tokens, out_tokens, info = backend_openai.query(
            "system",
            "user",
            model="deepseek-v4-flash",
            request_timeout_seconds=600,
            request_max_attempts=2,
            request_max_elapsed_seconds=1250,
            request_event_callback=lambda event, payload: events.append(
                (event, payload)
            ),
            request_event_context={"operation": "test"},
        )

    assert output == "answer"
    assert in_tokens == 7
    assert out_tokens == 11
    assert info["request_id"] == "req-safe-1"
    assert info["finish_reason"] == "length"
    assert info["reasoning_tokens"] == 3
    assert info["visible_completion_tokens"] == 8
    assert info["completion_tokens_details"] == {"reasoning_tokens": 3}
    assert seen["timeout"] == 600
    assert seen["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]
    finished = [
        payload for event, payload in events if event == "provider_request_finished"
    ]
    assert len(finished) == 1
    assert finished[0]["operation"] == "test"
    assert finished[0]["completion_tokens"] == 11
    assert finished[0]["finish_reason"] == "length"
    assert finished[0]["reasoning_tokens"] == 3
    assert finished[0]["visible_completion_tokens"] == 8
    assert finished[0]["completion_tokens_details"] == {"reasoning_tokens": 3}
    assert finished[0]["secret_values_recorded"] is False
    assert "answer" not in repr(events)
    assert "provider_private_detail" not in repr(events) + repr(info)


@pytest.mark.parametrize("details", [None, {}, {"audio_tokens": 2}])
def test_completion_token_info_keeps_missing_reasoning_unknown(details) -> None:
    assert backend_openai._completion_token_info(11, details) == (None, None)


def test_completion_token_info_accepts_explicit_zero_reasoning() -> None:
    assert backend_openai._completion_token_info(11, {"reasoning_tokens": 0}) == (
        0,
        11,
    )


@pytest.mark.parametrize(
    "request_kwargs",
    [
        {"request_timeout_seconds": float("nan")},
        {"request_timeout_seconds": float("inf")},
        {"request_max_elapsed_seconds": float("nan")},
        {"request_max_elapsed_seconds": float("inf")},
    ],
)
def test_openai_backend_rejects_non_finite_request_budget(request_kwargs) -> None:
    with patch.object(backend_openai, "_setup_openai_client", return_value=None):
        with pytest.raises(ValueError, match="finite"):
            backend_openai.query(
                "system",
                "user",
                model="deepseek-v4-flash",
                **request_kwargs,
            )


def test_success_metadata_rejects_unsafe_provider_identifiers() -> None:
    events: list[tuple[str, dict]] = []
    secret = "unsafe request id\nresponse-secret"
    completion = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="answer"),
                finish_reason="stop\nprivate-finish-reason",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=1,
            completion_tokens=1,
            completion_tokens_details=None,
        ),
        system_fingerprint="unsafe fingerprint with spaces",
        model="deepseek-v4-flash\nsecret-model-data",
        created=1,
        _request_id=secret,
    )
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_kwargs: completion)
        )
    )

    with (
        patch.object(backend_openai, "_setup_openai_client", return_value=None),
        patch.object(backend_openai, "_client", fake_client),
    ):
        _output, _elapsed, _in_tokens, _out_tokens, info = backend_openai.query(
            "system",
            "user",
            model="deepseek-v4-flash",
            request_event_callback=lambda event, payload: events.append(
                (event, payload)
            ),
        )

    assert info["request_id"] is None
    assert info["system_fingerprint"] is None
    assert info["model"] is None
    assert info["finish_reason"] == "unknown"
    assert info["reasoning_tokens"] is None
    assert info["visible_completion_tokens"] is None
    finished = [
        payload for event, payload in events if event == "provider_request_finished"
    ]
    assert finished[0]["reasoning_tokens"] is None
    assert finished[0]["visible_completion_tokens"] is None
    assert [
        payload["finish_reason"]
        for event, payload in events
        if event == "provider_request_finished"
    ] == ["unknown"]
    assert "response-secret" not in repr(events) + repr(info)
    assert "private-finish-reason" not in repr(events) + repr(info)


def test_missing_choices_sanitizes_provider_request_id() -> None:
    events: list[tuple[str, dict]] = []
    secret_marker = "private-marker-9137"
    completion = SimpleNamespace(choices=[], _request_id=f"unsafe id\n{secret_marker}")
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_kwargs: completion)
        )
    )

    with (
        patch.object(backend_openai, "_setup_openai_client", return_value=None),
        patch.object(backend_openai, "_client", fake_client),
    ):
        with pytest.raises(RuntimeError, match="no completion choices"):
            backend_openai.query(
                "system",
                "user",
                model="deepseek-v4-flash",
                request_event_callback=lambda event, payload: events.append(
                    (event, payload)
                ),
            )

    failed = [
        payload for event, payload in events if event == "provider_request_failed"
    ]
    assert len(failed) == 1
    assert failed[0]["request_id"] is None
    assert secret_marker not in repr(events)


def test_tool_fallback_warning_omits_provider_content(caplog) -> None:
    secret = "model-output-secret"
    completion = SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content=secret, tool_calls=None))
        ],
        usage=SimpleNamespace(
            prompt_tokens=1,
            completion_tokens=1,
            completion_tokens_details=None,
        ),
        system_fingerprint="fp",
        model="gpt-4",
        created=1,
        _request_id="req-safe",
    )
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_kwargs: completion)
        )
    )
    spec = FunctionSpec(name="answer", json_schema={"type": "object"}, description="")

    with (
        patch.object(backend_openai, "_setup_openai_client", return_value=None),
        patch.object(backend_openai, "_client", fake_client),
        caplog.at_level("WARNING", logger="biomaker"),
    ):
        output, *_rest = backend_openai.query(
            "system", "user", model="gpt-4", func_spec=spec
        )

    assert output == secret
    assert secret not in caplog.text
    assert "content omitted" in caplog.text


def test_invalid_deepseek_json_response_is_sanitized(caplog) -> None:
    events: list[tuple[str, dict]] = []
    secret = "malformed-json-response-secret"
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=secret))],
        usage=SimpleNamespace(
            prompt_tokens=1,
            completion_tokens=1,
            completion_tokens_details=None,
        ),
        system_fingerprint="fp",
        model="deepseek-v4-flash",
        created=1,
        _request_id="req-safe",
    )
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_kwargs: completion)
        )
    )
    spec = FunctionSpec(name="answer", json_schema={"type": "object"}, description="")

    with (
        patch.object(backend_openai, "_setup_openai_client", return_value=None),
        patch.object(backend_openai, "_client", fake_client),
        caplog.at_level("WARNING", logger="biomaker"),
    ):
        with pytest.raises(ProviderRequestError) as exc_info:
            backend_openai.query(
                "system",
                "user",
                model="deepseek-v4-flash",
                func_spec=spec,
                request_event_callback=lambda event, payload: events.append(
                    (event, payload)
                ),
            )

    serialized = caplog.text + repr(events) + str(exc_info.value)
    assert secret not in serialized
    assert exc_info.value.__cause__ is None
    assert exc_info.value.metadata["error_code"] == "invalid_json_response"


def test_invalid_function_arguments_are_sanitized(caplog) -> None:
    events: list[tuple[str, dict]] = []
    secret = '"function-arguments-secret"'
    function = SimpleNamespace(name="answer", arguments="{" + secret)
    completion = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[SimpleNamespace(function=function)],
                )
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=1,
            completion_tokens=1,
            completion_tokens_details=None,
        ),
        system_fingerprint="fp",
        model="gpt-4",
        created=1,
        _request_id="req-safe",
    )
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_kwargs: completion)
        )
    )
    spec = FunctionSpec(name="answer", json_schema={"type": "object"}, description="")

    with (
        patch.object(backend_openai, "_setup_openai_client", return_value=None),
        patch.object(backend_openai, "_client", fake_client),
        caplog.at_level("WARNING", logger="biomaker"),
    ):
        with pytest.raises(ProviderRequestError) as exc_info:
            backend_openai.query(
                "system",
                "user",
                model="gpt-4",
                func_spec=spec,
                request_event_callback=lambda event, payload: events.append(
                    (event, payload)
                ),
            )

    serialized = caplog.text + repr(events) + str(exc_info.value)
    assert "function-arguments-secret" not in serialized
    assert exc_info.value.__cause__ is None
    assert exc_info.value.metadata["error_code"] == "invalid_function_arguments"


def test_canonical_config_separates_generation_and_execution_policy() -> None:
    cfg = _load_cfg(use_cli_args=False)

    assert cfg.agent.code_generation_attempts == 1
    assert cfg.exec.parallel_workers == "auto"
    assert cfg.provider.request_timeout_seconds == 600
    assert cfg.provider.max_attempts == 2
    assert cfg.provider.max_elapsed_seconds == 1250
    _validate_exec_config(cfg)


def test_code_generation_has_one_logical_query_attempt_by_default(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = SimpleNamespace(
        log_dir=tmp_path,
        exec=SimpleNamespace(include_medical_packages=False),
        provider=SimpleNamespace(
            request_timeout_seconds=600,
            max_attempts=2,
            max_elapsed_seconds=1250,
        ),
        agent=SimpleNamespace(
            code_generation_attempts=1,
            code=SimpleNamespace(model="deepseek-v4-flash", temp=0.7),
        ),
    )
    agent = BaseAgent("task", cfg, SimpleNamespace(nodes=[]))
    calls = 0

    def invalid_query(**_kwargs):
        nonlocal calls
        calls += 1
        return "response without a code block", {
            "sent_messages": [],
            "model_info": {
                "finish_reason": "length",
                "reasoning_tokens": 20,
                "visible_completion_tokens": 1,
            },
        }

    monkeypatch.setattr("biomaker.agents.base_agent.query", invalid_query)
    with pytest.raises(CandidateGenerationError, match="after 1 attempts") as exc_info:
        agent.plan_and_code_query({"Task": "write code"})
    assert calls == 1
    assert exc_info.value.reason == "no valid Python code block"
    assert exc_info.value.response_chars == 29
    assert exc_info.value.prompt_capture["stats"]["model_info"] == {
        "finish_reason": "length",
        "reasoning_tokens": 20,
        "visible_completion_tokens": 1,
    }
