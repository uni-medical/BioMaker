import os
import logging
from typing import Any
from urllib.parse import urlparse

import requests

from ..utils.env import load_local_env
from ..utils.response import truncate_text

logger = logging.getLogger("biomaker")

SERPER_URL = "https://google.serper.dev/search"
JINA_READER_URL = "https://r.jina.ai/http://{url}"
MAX_TOPK = 5


def _read_with_jina(url: str, timeout: int = 20, max_chars: int = 1000) -> str:
    api_key = os.getenv("JINA_API_KEY")
    if not api_key or not url:
        return ""

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return ""

    response = requests.get(
        JINA_READER_URL.format(url=url),
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.text.strip()[:max_chars]


def _format_result(result: dict[str, Any]) -> str:
    title = result.get("title") or "Untitled"
    link = result.get("link") or ""
    snippet = result.get("snippet") or ""
    parts = [f"Title: {title}", f"URL: {link}", f"Snippet: {snippet}"]
    try:
        content = _read_with_jina(link)
    except Exception as exc:
        logger.debug("Jina reader failed for %s: %s", link, exc)
        content = ""
    if content:
        parts.append(f"Jina reader excerpt:\n{content}")
    return "\n".join(parts)


def web_search(query: str, topk: int = 2) -> str:
    """Search with Serper, then enrich result pages with Jina Reader when configured."""

    load_local_env()
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        return ""

    topk = max(1, min(int(topk), MAX_TOPK))
    query = " ".join(str(query).split())[:300]
    if not query:
        return ""

    response = requests.post(
        SERPER_URL,
        headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
        json={"q": query, "num": topk},
        timeout=20,
    )
    response.raise_for_status()
    results = response.json().get("organic", [])[:topk]
    if not results:
        return ""

    result_text = "\n\n---\n\n".join(_format_result(result) for result in results)
    return truncate_text(result_text, max_chars=1500, marker="web search trimmed")
