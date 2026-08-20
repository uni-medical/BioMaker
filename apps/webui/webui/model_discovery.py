from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

DISCOVERY_TIMEOUT_SECONDS = 5.0
MAX_RESPONSE_BYTES = 1_000_000
MAX_MODELS = 1_000
MAX_MODEL_ID_LENGTH = 500


class _NoRedirectHandler(HTTPRedirectHandler):
    """Do not forward an API key to a redirect target."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


def _normalized_urls(base_url: str) -> tuple[str, str]:
    raw = base_url.strip()
    if not raw or any(ord(character) < 32 for character in raw):
        raise ValueError("Enter a valid HTTP(S) Base URL.")

    try:
        parsed = urlsplit(raw)
        # Accessing port also validates that it is numeric and in range.
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("Enter a valid HTTP(S) Base URL.") from exc

    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "Enter a valid HTTP(S) Base URL without credentials or query parameters."
        )

    scheme = parsed.scheme.lower()
    path = parsed.path.rstrip("/")
    normalized_base = urlunsplit((scheme, parsed.netloc.lower(), path, "", ""))
    if path.endswith("/models"):
        models_path = path
    elif path:
        models_path = f"{path}/models"
    else:
        models_path = "/v1/models"
    endpoint = urlunsplit((scheme, parsed.netloc, models_path, "", ""))
    return normalized_base, endpoint


def normalize_base_url(base_url: str) -> str:
    return _normalized_urls(base_url)[0]


def _result(
    status: str,
    *,
    base_url: str,
    models: list[str] | None = None,
    detail: str | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "models": models or [],
        "base_url": base_url,
        "detail": detail,
    }


def discover_models(
    base_url: str | None,
    api_key: str | None,
    *,
    timeout: float = DISCOVERY_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Discover model IDs without returning provider bodies or credentials."""

    if not base_url:
        return _result(
            "unconfigured",
            base_url="",
            detail="Configure an OpenAI-compatible Base URL first.",
        )

    try:
        normalized_base, endpoint = _normalized_urls(base_url)
    except ValueError as exc:
        return _result("error", base_url="", detail=str(exc))

    if api_key and any(
        ord(character) < 32 or ord(character) == 127 for character in api_key
    ):
        return _result(
            "error",
            base_url=normalized_base,
            detail="The configured API key contains invalid characters.",
        )

    headers = {
        "Accept": "application/json",
        "User-Agent": "BioMaker-WebUI/0.1 model-discovery",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(endpoint, headers=headers, method="GET")
    opener = build_opener(_NoRedirectHandler())

    try:
        with opener.open(request, timeout=timeout) as response:
            body = response.read(MAX_RESPONSE_BYTES + 1)
    except HTTPError as exc:
        if exc.code in {301, 302, 303, 307, 308, 404, 405, 501}:
            detail = f"This endpoint does not expose model discovery (HTTP {exc.code})."
            return _result("unsupported", base_url=normalized_base, detail=detail)
        detail = f"Model discovery failed (HTTP {exc.code})."
        return _result("error", base_url=normalized_base, detail=detail)
    except (TimeoutError, URLError, OSError):
        return _result(
            "error",
            base_url=normalized_base,
            detail="Could not reach the model endpoint within the time limit.",
        )

    if len(body) > MAX_RESPONSE_BYTES:
        return _result(
            "error",
            base_url=normalized_base,
            detail="The model endpoint returned an unexpectedly large response.",
        )

    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return _result(
            "unsupported",
            base_url=normalized_base,
            detail="This endpoint did not return an OpenAI-compatible model list.",
        )

    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return _result(
            "unsupported",
            base_url=normalized_base,
            detail="This endpoint did not return an OpenAI-compatible model list.",
        )

    models: list[str] = []
    seen: set[str] = set()
    for item in data:
        model_id = item.get("id") if isinstance(item, dict) else None
        if not isinstance(model_id, str):
            continue
        model_id = model_id.strip()
        if (
            not model_id
            or len(model_id) > MAX_MODEL_ID_LENGTH
            or any(ord(character) < 32 for character in model_id)
            or model_id in seen
        ):
            continue
        seen.add(model_id)
        models.append(model_id)
        if len(models) >= MAX_MODELS:
            break

    if not models:
        return _result(
            "unsupported",
            base_url=normalized_base,
            detail="The endpoint returned no usable model IDs.",
        )
    return _result("ok", base_url=normalized_base, models=models)
