import logging
import math
import re
import time
from dataclasses import dataclass
from typing import Any, Callable

import jsonschema
from dataclasses_json import DataClassJsonMixin

PromptType = str | dict | list
FunctionCallType = dict
OutputType = str | FunctionCallType


logger = logging.getLogger("biomaker")


RequestEventCallback = Callable[[str, dict[str, Any]], None]

PROVIDER_FINISH_REASONS = frozenset(
    {
        "content_filter",
        "function_call",
        "insufficient_system_resource",
        "length",
        "stop",
        "tool_calls",
    }
)


class ProviderRequestError(RuntimeError):
    """A provider failure whose public message contains no response body."""

    def __init__(self, metadata: dict[str, Any]):
        self.metadata = dict(metadata)
        self.metadata["error_code"] = sanitize_provider_token(
            self.metadata.get("error_code")
        )
        self.metadata["request_id"] = sanitize_provider_token(
            self.metadata.get("request_id")
        )
        summary = [
            str(self.metadata.get("failure_kind") or "provider"),
            str(self.metadata.get("exception_type") or "unknown error"),
        ]
        if self.metadata.get("status_code") is not None:
            summary.append(f"HTTP {self.metadata['status_code']}")
        if self.metadata.get("attempt") is not None:
            summary.append(f"attempt {self.metadata['attempt']}")
        super().__init__("Provider request failed (" + ", ".join(summary) + ").")


def sanitize_provider_token(value: Any, *, max_chars: int = 128) -> str | int | None:
    """Return a small log-safe provider identifier or ``None``.

    Provider-controlled identifiers are diagnostic metadata, not trusted log
    text.  Keep only the conservative token alphabet already used by the queue
    preflight and reject values that could contain response text, paths, or
    control characters.
    """

    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text or len(text) > max_chars:
        return None
    if re.fullmatch(r"[A-Za-z0-9_.:-]+", text) is None:
        return None
    return text


def sanitize_provider_finish_reason(value: Any) -> str | None:
    """Map provider finish reasons to a fixed, non-content-bearing enum."""

    if value is None:
        return None
    token = sanitize_provider_token(value)
    if isinstance(token, str) and token in PROVIDER_FINISH_REASONS:
        return token
    return "unknown"


def sanitized_exception_metadata(exc: BaseException) -> dict[str, Any]:
    """Return diagnostic fields without recording the provider response body."""

    response = getattr(exc, "response", None)
    status_code = getattr(exc, "status_code", None)
    if status_code is None and response is not None:
        status_code = getattr(response, "status_code", None)
    if isinstance(status_code, bool) or not isinstance(status_code, int):
        status_code = None

    request_id = getattr(exc, "request_id", None)
    headers = getattr(response, "headers", None)
    if request_id is None and headers is not None:
        try:
            request_id = headers.get("x-request-id") or headers.get("request-id")
        except Exception:
            request_id = None

    error_code = getattr(exc, "code", None)
    body = getattr(exc, "body", None)
    if error_code is None and isinstance(body, dict):
        body_error = body.get("error")
        if isinstance(body_error, dict):
            error_code = body_error.get("code") or body_error.get("type")
        else:
            error_code = body.get("code") or body.get("type")

    exception_type = type(exc).__name__
    lowered = exception_type.casefold()
    normalized_code = str(error_code or "").casefold()
    if "quota" in normalized_code or status_code == 402:
        failure_kind = "quota"
    elif status_code in {401, 403}:
        failure_kind = "auth"
    elif status_code == 429:
        failure_kind = "rate_limit"
    elif status_code is not None and status_code >= 500:
        failure_kind = "server"
    elif "timeout" in lowered:
        failure_kind = "timeout"
    elif "connection" in lowered:
        failure_kind = "connection"
    else:
        failure_kind = "provider"

    return {
        "exception_type": exception_type,
        "failure_kind": failure_kind,
        "status_code": status_code,
        "error_code": sanitize_provider_token(error_code),
        "request_id": sanitize_provider_token(request_id),
    }


def _emit_request_event(
    callback: RequestEventCallback | None,
    event: str,
    **payload: Any,
) -> None:
    if callback is None:
        return
    try:
        callback(event, payload)
    except Exception:
        logger.debug("Provider request event callback failed.", exc_info=True)


def backoff_create(
    create_fn: Callable,
    retry_exceptions: tuple[type[BaseException], ...] | list[type[BaseException]],
    *args,
    max_tries: int = 2,
    max_elapsed_seconds: float = 1250.0,
    event_callback: RequestEventCallback | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    monotonic_fn: Callable[[], float] = time.monotonic,
    **kwargs,
):
    """Call a provider with a small, time-bounded transient retry budget."""

    if (
        isinstance(max_tries, bool)
        or not isinstance(max_tries, int)
        or not 1 <= max_tries <= 2
    ):
        raise ValueError("max_tries must be an integer in [1, 2]")
    if (
        isinstance(max_elapsed_seconds, bool)
        or not isinstance(max_elapsed_seconds, (int, float))
        or not math.isfinite(float(max_elapsed_seconds))
        or max_elapsed_seconds <= 0
    ):
        raise ValueError("max_elapsed_seconds must be finite and positive")
    configured_timeout = kwargs.get("timeout")
    if configured_timeout is not None and (
        isinstance(configured_timeout, bool)
        or not isinstance(configured_timeout, (int, float))
        or not math.isfinite(float(configured_timeout))
        or configured_timeout <= 0
    ):
        raise ValueError("timeout must be finite and positive")
    retry_types = tuple(retry_exceptions)
    started_at = monotonic_fn()

    for attempt in range(1, max_tries + 1):
        elapsed_before = max(0.0, monotonic_fn() - started_at)
        remaining = max_elapsed_seconds - elapsed_before
        if remaining <= 0:
            metadata = {
                "exception_type": "ProviderDeadlineExceeded",
                "failure_kind": "timeout",
                "status_code": None,
                "error_code": None,
                "request_id": None,
                "attempt": attempt - 1,
                "max_attempts": max_tries,
                "elapsed_seconds": round(elapsed_before, 3),
                "max_elapsed_seconds": max_elapsed_seconds,
            }
            _emit_request_event(event_callback, "provider_request_failed", **metadata)
            raise ProviderRequestError(metadata)

        attempt_kwargs = dict(kwargs)
        request_timeout = attempt_kwargs.get("timeout")
        if isinstance(request_timeout, (int, float)) and not isinstance(
            request_timeout, bool
        ):
            attempt_kwargs["timeout"] = max(
                0.001, min(float(request_timeout), remaining)
            )
        attempt_started = monotonic_fn()
        _emit_request_event(
            event_callback,
            "provider_request_started",
            attempt=attempt,
            max_attempts=max_tries,
            elapsed_seconds=round(elapsed_before, 3),
            max_elapsed_seconds=max_elapsed_seconds,
            request_timeout_seconds=attempt_kwargs.get("timeout"),
        )
        try:
            result = create_fn(*args, **attempt_kwargs)
        except retry_types as exc:
            now = monotonic_fn()
            elapsed = max(0.0, now - started_at)
            metadata = {
                **sanitized_exception_metadata(exc),
                "attempt": attempt,
                "max_attempts": max_tries,
                "attempt_elapsed_seconds": round(max(0.0, now - attempt_started), 3),
                "elapsed_seconds": round(elapsed, 3),
                "max_elapsed_seconds": max_elapsed_seconds,
            }
            delay = min(60.0, 1.5 * (2 ** (attempt - 1)))
            remaining_after = max_elapsed_seconds - elapsed
            if (
                metadata["failure_kind"] in {"auth", "quota"}
                or attempt >= max_tries
                or remaining_after <= delay
            ):
                _emit_request_event(
                    event_callback, "provider_request_failed", **metadata
                )
                logger.warning(
                    "Provider request failed after bounded retries: %s",
                    metadata,
                )
                raise ProviderRequestError(metadata) from None
            _emit_request_event(
                event_callback,
                "provider_request_retry_scheduled",
                **metadata,
                delay_seconds=delay,
            )
            logger.warning(
                "Retrying provider request after transient failure: %s",
                {**metadata, "delay_seconds": delay},
            )
            sleep_fn(delay)
            continue
        else:
            now = monotonic_fn()
            _emit_request_event(
                event_callback,
                "provider_request_attempt_succeeded",
                attempt=attempt,
                max_attempts=max_tries,
                attempt_elapsed_seconds=round(max(0.0, now - attempt_started), 3),
                elapsed_seconds=round(max(0.0, now - started_at), 3),
            )
            return result

    raise AssertionError("bounded provider retry loop exited unexpectedly")


def opt_messages_to_list(
    system_message: str | None, user_message: str | None
) -> list[dict[str, str]]:
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    if user_message:
        messages.append({"role": "user", "content": user_message})
    return messages


def compile_prompt_to_md(prompt: PromptType, _header_depth: int = 1) -> str:
    if isinstance(prompt, str):
        return prompt.strip() + "\n"
    elif isinstance(prompt, list):
        items = []
        for value in prompt:
            text = compile_prompt_to_md(value, _header_depth=_header_depth).strip()
            if "\n" in text:
                text = "\n  ".join(text.splitlines())
            items.append(f"- {text}")
        return "\n".join(items + ["\n"])
    elif not isinstance(prompt, dict):
        return str(prompt).strip() + "\n"

    out = []
    header_prefix = "#" * _header_depth
    for k, v in prompt.items():
        out.append(f"{header_prefix} {k}\n")
        out.append(compile_prompt_to_md(v, _header_depth=_header_depth + 1))
    return "\n".join(out)


@dataclass
class FunctionSpec(DataClassJsonMixin):
    name: str
    json_schema: dict  # JSON schema
    description: str

    def __post_init__(self):
        # validate the schema
        jsonschema.Draft7Validator.check_schema(self.json_schema)

    @property
    def as_openai_tool_dict(self):
        """Convert to OpenAI's function format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.json_schema,
            },
        }

    @property
    def openai_tool_choice_dict(self):
        return {
            "type": "function",
            "function": {"name": self.name},
        }
