"""Backend for OpenAI API."""

import json
import logging
import os
import re
import time

import jsonschema
import openai
from funcy import notnone, once, select_values

from ..utils.env import load_local_env
from .utils import FunctionSpec, OutputType, backoff_create, opt_messages_to_list

logger = logging.getLogger("biomaker")

_client: openai.OpenAI = None  # type: ignore

OPENAI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.InternalServerError,
)


def _request_timeout_seconds() -> float:
    raw = os.environ.get("OPENAI_REQUEST_TIMEOUT_SECONDS", "300")
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "Invalid OPENAI_REQUEST_TIMEOUT_SECONDS=%r; using 300 seconds.", raw
        )
        return 300.0
    if value <= 0:
        logger.warning(
            "OPENAI_REQUEST_TIMEOUT_SECONDS must be positive; using 300 seconds."
        )
        return 300.0
    return value


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


@once
def _setup_openai_client():
    global _client
    load_local_env()
    _client = openai.OpenAI(max_retries=0, timeout=_request_timeout_seconds())


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
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

    t0 = time.time()
    completion = backoff_create(
        _client.chat.completions.create,
        OPENAI_TIMEOUT_EXCEPTIONS,
        messages=messages,
        **filtered_kwargs,
    )
    req_time = time.time() - t0

    choices = getattr(completion, "choices", None)
    if not choices:
        raise RuntimeError(
            "OpenAI completion failed after retries; no completion choices were returned."
        )
    choice = choices[0]

    if func_spec is None:
        output = choice.message.content
    elif not use_tool_call:
        output = _parse_json_object(choice.message.content or "", func_spec)
    else:
        tool_calls = getattr(choice.message, "tool_calls", None)

        if not tool_calls:
            logger.warning(
                f"No function call used despite function spec. Fallback to text. "
                f"Message content: {choice.message.content}"
            )
            output = choice.message.content
        else:
            first_call = tool_calls[0]
            if first_call.function.name != func_spec.name:
                raise RuntimeError(
                    f"Function name mismatch: expected {func_spec.name}, "
                    f"got {first_call.function.name}"
                )
            try:
                output = json.loads(first_call.function.arguments)
            except json.JSONDecodeError as e:
                logger.error(
                    f"Error decoding function arguments:\n{first_call.function.arguments}"
                )
                raise e

    in_tokens = completion.usage.prompt_tokens
    out_tokens = completion.usage.completion_tokens

    info = {
        "system_fingerprint": completion.system_fingerprint,
        "model": completion.model,
        "created": completion.created,
    }

    return output, req_time, in_tokens, out_tokens, info
