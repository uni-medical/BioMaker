"""Shared result-marker semantics for runtime scoring and historical journals."""


def has_fallback_marker(value: object) -> bool:
    if isinstance(value, dict):
        if any(
            value.get(key) for key in ("fallback", "fallback_used", "used_fallback")
        ):
            return True
        return any(has_fallback_marker(child) for child in value.values())
    if isinstance(value, list):
        return any(has_fallback_marker(child) for child in value)
    return False


def has_invalid_result(value: object) -> bool:
    """An explicit failure anywhere in a metric result cannot become a score."""
    if isinstance(value, dict):
        if (
            any(
                value.get(key)
                for key in ("fallback", "fallback_used", "used_fallback", "error")
            )
            or value.get("valid") is False
            or value.get("ok") is False
            or ("maximize" in value and not isinstance(value["maximize"], bool))
        ):
            return True
        return any(has_invalid_result(child) for child in value.values())
    if isinstance(value, list):
        return any(has_invalid_result(child) for child in value)
    return False
