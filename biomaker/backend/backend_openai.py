"""Backend for OpenAI API."""

import json
import logging
import math
import os
import re
import time
import uuid

import jsonschema
import openai
from funcy import notnone, once, select_values

from ..utils.env import load_local_env
from .utils import (
    FunctionSpec,
    OutputType,
    ProviderRequestError,
    RequestEventCallback,
    backoff_create,
    opt_messages_to_list,
    sanitize_provider_finish_reason,
    sanitize_provider_token,
    sanitized_exception_metadata,
)

logger = logging.getLogger("biomaker")

_client: openai.OpenAI = None  # type: ignore

OPENAI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.InternalServerError,
)


def _positive_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, str(default))
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %s.", name, raw, default)
        return default
    if not math.isfinite(value) or value <= 0:
        logger.warning("%s must be finite and positive; using %s.", name, default)
        return default
    return value


def _positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %s.", name, raw, default)
        return default
    if value < 1:
        logger.warning("%s must be >= 1; using %s.", name, default)
        return default
    return value


def _request_timeout_seconds() -> float:
    return _positive_float_env("OPENAI_REQUEST_TIMEOUT_SECONDS", 600.0)


def _request_max_attempts() -> int:
    value = _positive_int_env("OPENAI_REQUEST_MAX_ATTEMPTS", 2)
    if value > 2:
        logger.warning(
            "OPENAI_REQUEST_MAX_ATTEMPTS=%s exceeds the safety cap; using 2.",
            value,
        )
        return 2
    return value


def _request_max_elapsed_seconds() -> float:
    return _positive_float_env("OPENAI_REQUEST_MAX_ELAPSED_SECONDS", 1250.0)


def request_policy() -> dict[str, float | int]:
    """Return the effective bounded provider policy without any credentials."""

    return {
        "request_timeout_seconds": _request_timeout_seconds(),
        "max_attempts": _request_max_attempts(),
        "max_elapsed_seconds": _request_max_elapsed_seconds(),
    }


def _supports_openai_tool_choice(model_name: str) -> bool:
    return not model_name.startswith("deepseek")


def _append_json_fallback_instruction(
    system_message: str | None, func_spec: FunctionSpec
) -> str:
    instruction = (
        "\n\nYou must return exactly one raw JSON object and no markdown. "
        f"The JSON object must satisfy this schema for `{func_spec.name}`:\n"
        + json.dumps(func_spec.json_schema, ensure_ascii=False, indent=2)
    )
    return (system_message or "") + instruction


def _parse_json_object(text: str, func_spec: FunctionSpec) -> dict:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        if start < 0:
            raise
        data, _ = json.JSONDecoder().raw_decode(stripped[start:])
    jsonschema.validate(data, func_spec.json_schema)
    return data


_COMPLETION_TOKEN_DETAIL_FIELDS = frozenset(
    {
        "accepted_prediction_tokens",
        "audio_tokens",
        "reasoning_tokens",
        "rejected_prediction_tokens",
    }
)


def _sanitize_completion_token_details(value) -> dict[str, int] | None:
    if value is not None and hasattr(value, "model_dump"):
        value = value.model_dump()
    if not isinstance(value, dict):
        return None
    sanitized = {
        key: item
        for key, item in value.items()
        if key in _COMPLETION_TOKEN_DETAIL_FIELDS
        and isinstance(item, int)
        and not isinstance(item, bool)
        and item >= 0
    }
    return sanitized or None


def _completion_token_info(
    completion_tokens: int,
    details: dict[str, int] | None,
) -> tuple[int | None, int | None]:
    """Return reasoning and visible token counts without provider text."""

    if (
        isinstance(completion_tokens, bool)
        or not isinstance(completion_tokens, int)
        or completion_tokens < 0
    ):
        return None, None
    if not isinstance(details, dict) or "reasoning_tokens" not in details:
        return None, None
    reasoning_tokens = details["reasoning_tokens"]
    if (
        isinstance(reasoning_tokens, bool)
        or not isinstance(reasoning_tokens, int)
        or reasoning_tokens < 0
        or reasoning_tokens > completion_tokens
    ):
        return None, None
    return reasoning_tokens, completion_tokens - reasoning_tokens


@once
def _setup_openai_client():
    global _client
    load_local_env()
    _client = openai.OpenAI(max_retries=0, timeout=_request_timeout_seconds())


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    request_timeout_seconds: float | None = None,
    request_max_attempts: int | None = None,
    request_max_elapsed_seconds: float | None = None,
    request_event_callback: RequestEventCallback | None = None,
    request_event_context: dict | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    """
    Query the OpenAI API, optionally with function calling.
    Function calling support is only checked for feedback/review operations.
    """
    _setup_openai_client()
    filtered_kwargs: dict = select_values(notnone, model_kwargs)
    model_name = filtered_kwargs.get("model", "")
    logger.debug(f"OpenAI query called with model='{model_name}'")

    messages = opt_messages_to_list(system_message, user_message)

    use_tool_call = func_spec is not None and _supports_openai_tool_choice(model_name)
    if func_spec is not None and not use_tool_call:
        messages = opt_messages_to_list(
            _append_json_fallback_instruction(system_message, func_spec),
            user_message,
        )

    if func_spec is not None and use_tool_call:
        filtered_kwargs["tools"] = [func_spec.as_openai_tool_dict]
        filtered_kwargs["tool_choice"] = func_spec.openai_tool_choice_dict

    policy = request_policy()
    timeout_seconds = float(
        request_timeout_seconds
        if request_timeout_seconds is not None
        else policy["request_timeout_seconds"]
    )
    max_attempts = int(
        request_max_attempts
        if request_max_attempts is not None
        else policy["max_attempts"]
    )
    max_elapsed_seconds = float(
        request_max_elapsed_seconds
        if request_max_elapsed_seconds is not None
        else policy["max_elapsed_seconds"]
    )
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("request_timeout_seconds must be finite and positive")
    if not 1 <= max_attempts <= 2:
        raise ValueError("request_max_attempts must be in [1, 2]")
    if (
        not math.isfinite(max_elapsed_seconds)
        or max_elapsed_seconds <= 0
        or max_elapsed_seconds < timeout_seconds
    ):
        raise ValueError(
            "request_max_elapsed_seconds must be finite, positive, and >= "
            "request_timeout_seconds"
        )

    event_context = dict(request_event_context or {})
    event_context.update(
        {
            "query_id": uuid.uuid4().hex,
            "provider": "openai",
            "model": model_name,
            "request_timeout_seconds": timeout_seconds,
            "request_max_attempts": max_attempts,
            "request_max_elapsed_seconds": max_elapsed_seconds,
            "secret_values_recorded": False,
        }
    )

    def observe(event: str, payload: dict) -> None:
        if request_event_callback is not None:
            try:
                request_event_callback(event, {**event_context, **payload})
            except Exception:
                logger.debug("Provider request event callback failed.", exc_info=True)

    t0 = time.time()
    try:
        completion = backoff_create(
            _client.chat.completions.create,
            OPENAI_TIMEOUT_EXCEPTIONS,
            messages=messages,
            timeout=timeout_seconds,
            max_tries=max_attempts,
            max_elapsed_seconds=max_elapsed_seconds,
            event_callback=observe,
            **filtered_kwargs,
        )
    except ProviderRequestError:
        raise
    except openai.APIError as exc:
        metadata = {
            **sanitized_exception_metadata(exc),
            "attempt": 1,
            "max_attempts": max_attempts,
            "elapsed_seconds": round(time.time() - t0, 3),
            "max_elapsed_seconds": max_elapsed_seconds,
        }
        observe("provider_request_failed", metadata)
        raise ProviderRequestError(metadata) from None
    req_time = time.time() - t0

    choices = getattr(completion, "choices", None)
    request_id = sanitize_provider_token(getattr(completion, "_request_id", None))
    if not choices:
        observe(
            "provider_request_failed",
            {
                "exception_type": "InvalidProviderResponse",
                "failure_kind": "provider",
                "status_code": None,
                "error_code": "missing_choices",
                "request_id": request_id,
                "elapsed_seconds": round(req_time, 3),
            },
        )
        raise RuntimeError(
            "OpenAI completion failed after retries; no completion choices were returned."
        )
    choice = choices[0]
    finish_reason = sanitize_provider_finish_reason(
        getattr(choice, "finish_reason", None)
    )

    if func_spec is None:
        output = choice.message.content
    elif not use_tool_call:
        try:
            output = _parse_json_object(choice.message.content or "", func_spec)
        except (json.JSONDecodeError, jsonschema.ValidationError):
            metadata = {
                "exception_type": "InvalidProviderResponse",
                "failure_kind": "provider",
                "status_code": None,
                "error_code": "invalid_json_response",
                "request_id": request_id,
                "elapsed_seconds": round(req_time, 3),
            }
            observe("provider_request_failed", metadata)
            logger.warning("Provider JSON response was invalid; content omitted.")
            raise ProviderRequestError(metadata) from None
    else:
        tool_calls = getattr(choice.message, "tool_calls", None)

        if not tool_calls:
            logger.warning(
                "No function call used despite function spec; falling back to "
                "text with provider content omitted from logs (chars=%d).",
                len(choice.message.content or ""),
            )
            output = choice.message.content
        else:
            first_call = tool_calls[0]
            if first_call.function.name != func_spec.name:
                metadata = {
                    "exception_type": "InvalidProviderResponse",
                    "failure_kind": "provider",
                    "status_code": None,
                    "error_code": "function_name_mismatch",
                    "request_id": request_id,
                    "elapsed_seconds": round(req_time, 3),
                }
                observe("provider_request_failed", metadata)
                raise ProviderRequestError(metadata) from None
            try:
                output = json.loads(first_call.function.arguments)
            except json.JSONDecodeError:
                metadata = {
                    "exception_type": "InvalidProviderResponse",
                    "failure_kind": "provider",
                    "status_code": None,
                    "error_code": "invalid_function_arguments",
                    "request_id": request_id,
                    "elapsed_seconds": round(req_time, 3),
                }
                observe("provider_request_failed", metadata)
                logger.warning(
                    "Provider function arguments were invalid JSON; content "
                    "omitted from logs (chars=%d).",
                    len(first_call.function.arguments or ""),
                )
                raise ProviderRequestError(metadata) from None

    usage = completion.usage
    in_tokens = usage.prompt_tokens
    out_tokens = usage.completion_tokens
    completion_details = _sanitize_completion_token_details(
        getattr(usage, "completion_tokens_details", None)
    )
    reasoning_tokens, visible_completion_tokens = _completion_token_info(
        out_tokens, completion_details
    )

    info = {
        "system_fingerprint": sanitize_provider_token(completion.system_fingerprint),
        "model": sanitize_provider_token(completion.model),
        "created": completion.created,
        "request_id": request_id,
        "finish_reason": finish_reason,
        "reasoning_tokens": reasoning_tokens,
        "visible_completion_tokens": visible_completion_tokens,
        "completion_tokens_details": completion_details,
    }

    observe(
        "provider_request_finished",
        {
            "elapsed_seconds": round(req_time, 3),
            "prompt_tokens": in_tokens,
            "completion_tokens": out_tokens,
            "finish_reason": finish_reason,
            "reasoning_tokens": reasoning_tokens,
            "visible_completion_tokens": visible_completion_tokens,
            "request_id": info["request_id"],
            "completion_tokens_details": completion_details,
        },
    )

    return output, req_time, in_tokens, out_tokens, info
