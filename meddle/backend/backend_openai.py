"""Backend for OpenAI API."""

import json
import logging
import time

from .utils import FunctionSpec, OutputType, opt_messages_to_list, backoff_create
from funcy import notnone, once, select_values
import openai
import os

logger = logging.getLogger("meddle")

_client: openai.OpenAI = None  # type: ignore

OPENAI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


@once
def _setup_openai_client():
    global _client
    _client = openai.OpenAI(
        max_retries=0
    )

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

    if func_spec is not None:
        # Only check function call support for feedback/search operations
        if func_spec.name == "submit_review":
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

    choice = completion.choices[0]

    if func_spec is None or "tools" not in filtered_kwargs:
        output = choice.message.content
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
            assert first_call.function.name == func_spec.name, (
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
