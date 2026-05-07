import logging
import re

from . import backend_openai
from .utils import FunctionSpec, OutputType, PromptType, compile_prompt_to_md

logger = logging.getLogger("biomaker")
PROMPT_WARNING_CHARS = 45000


def determine_provider(model: str) -> str:
    # All supported backends currently use the OpenAI-compatible API.
    return "openai"


provider_to_query_func = {
    "openai": backend_openai.query,
}


def _prompt_section_sizes(prompt: PromptType | None) -> str:
    if prompt is None:
        return "none"
    if isinstance(prompt, str):
        return f"str:{len(prompt)}"
    if isinstance(prompt, list):
        return f"list:{len(compile_prompt_to_md(prompt))}"
    if isinstance(prompt, dict):
        return ", ".join(
            f"{key}:{len(compile_prompt_to_md(value))}" for key, value in prompt.items()
        )
    return f"{type(prompt).__name__}:{len(str(prompt))}"


def query(
    system_message: PromptType | None,
    user_message: PromptType | None,
    model: str,
    temperature: float | None = None,
    max_completion_tokens: int | None = None,
    func_spec: FunctionSpec | None = None,
    return_metadata: bool = False,
    **model_kwargs,
) -> OutputType | tuple[OutputType, dict]:
    """
    General LLM query for various backends with a single system and user message.
    Supports function calling for some backends.

    Args:
        system_message (PromptType | None): Uncompiled system message (will generate a message following the OpenAI format)
        user_message (PromptType | None): Uncompiled user message (will generate a message following the OpenAI format)
        model (str): string identifier for the model to use (e.g. "gpt-4-turbo")
        temperature (float | None, optional): Temperature to sample at. Defaults to the model-specific default.
        max_completion_tokens (int | None, optional): Maximum number of tokens to generate. Defaults to the model-specific max tokens.
        func_spec (FunctionSpec | None, optional): Optional FunctionSpec object defining a function call. If given, the return value will be a dict.

    Returns:
        OutputType: A string completion if func_spec is None, otherwise a dict with the function call details.
    """

    model_kwargs = model_kwargs | {
        "model": model,
        "temperature": temperature,
        "max_completion_tokens": max_completion_tokens,
    }

    # Handle models with beta limitations
    # ref: https://platform.openai.com/docs/guides/reasoning/beta-limitations
    if re.match(r"^o\d", model):
        if system_message:
            user_message = {
                "System instructions": system_message,
                "User request": (
                    user_message
                    if user_message
                    else "Please complete the task as described above."
                ),
            }
        system_message = None
        model_kwargs["temperature"] = 1

    compiled_system = compile_prompt_to_md(system_message) if system_message else None
    compiled_user = (
        compile_prompt_to_md(user_message)
        if user_message
        else "Please complete the task as described above."
    )
    logger.debug(
        "Prompt context summary | model=%s | func=%s | system_chars=%s | user_chars=%s | system_sections=[%s] | user_sections=[%s]",
        model,
        func_spec.name if func_spec is not None else "none",
        len(compiled_system or ""),
        len(compiled_user or ""),
        _prompt_section_sizes(system_message),
        _prompt_section_sizes(user_message),
    )
    if len(compiled_system or "") + len(compiled_user or "") > PROMPT_WARNING_CHARS:
        logger.warning(
            "Prompt context is large (%d chars). Consider reducing data overview, memory, retrieved knowledge, or code context.",
            len(compiled_system or "") + len(compiled_user or ""),
        )

    provider = determine_provider(model)
    query_func = provider_to_query_func[provider]
    output, req_time, in_tok_count, out_tok_count, info = query_func(
        system_message=compiled_system,
        user_message=compiled_user,
        func_spec=func_spec,
        **model_kwargs,
    )

    if return_metadata:
        metadata = {
            "request_time_seconds": req_time,
            "prompt_tokens": in_tok_count,
            "completion_tokens": out_tok_count,
            "provider": provider,
            "model": model,
            "model_info": info,
            "compiled_system_chars": len(compiled_system or ""),
            "compiled_user_chars": len(compiled_user or ""),
            "sent_messages": [
                message
                for message in [
                    (
                        {"role": "system", "content": compiled_system}
                        if compiled_system
                        else None
                    ),
                    (
                        {"role": "user", "content": compiled_user}
                        if compiled_user
                        else None
                    ),
                ]
                if message is not None
            ],
        }
        return output, metadata  # type: ignore[return-value]

    return output
