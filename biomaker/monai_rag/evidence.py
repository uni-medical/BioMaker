"""Deterministic, content-free evidence for MONAI RAG availability and use."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .create_monai_rag_db import CHUNKS_FILE_NAME
from .query_rag_db import MED_KNOWLEDGE_RAG, PERSIST_DIR

MONAI_RAG_SMOKE_QUERY = "monai.transforms.EnsureChannelFirstd"
MONAI_RAG_SMOKE_RECEIPT = "monai_rag_smoke.json"
MONAI_RAG_QUERY_EVENT = "monai_rag_query"
MONAI_RAG_EVENT_LOG = "events.jsonl"


class MonaiRagEvidenceError(RuntimeError):
    """Raised when required MONAI RAG evidence cannot be established."""

    def __init__(
        self,
        message: str,
        *,
        receipt: dict[str, Any] | None = None,
        receipt_path: Path | None = None,
    ) -> None:
        super().__init__(message)
        self.receipt = receipt
        self.receipt_path = receipt_path


def query_sha256(query: str) -> str:
    """Return a stable digest without persisting the query itself."""
    return hashlib.sha256(str(query).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _error_summary(exc: BaseException, persist_dir: Path) -> str:
    # Exception bodies may contain a query, a document, or provider credentials.
    # Persist only the exception class to keep the evidence content-free.
    return type(exc).__name__


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _legacy_result_count(result: Any) -> int:
    """Best-effort compatibility for injected RAGs using the old search API."""
    if isinstance(result, (list, tuple)):
        return len(result)
    return int(bool(result))


def run_monai_rag_smoke(
    receipt_path: Path,
    *,
    persist_dir: Path = PERSIST_DIR,
    required: bool = False,
    rag_factory=None,
) -> dict[str, Any]:
    """Run one offline keyword lookup and atomically write its compact receipt.

    The receipt intentionally has no timestamp, query text, or retrieved content, so
    identical index state and settings produce byte-identical evidence.  When
    ``required`` is true, any missing/unusable index, query error, zero-hit result,
    or receipt-write failure raises :class:`MonaiRagEvidenceError`.
    """
    receipt_path = Path(receipt_path)
    persist_dir = Path(persist_dir)
    index_file = persist_dir / CHUNKS_FILE_NAME
    index_available = index_file.is_file()
    chunks_sha256: str | None = None
    index_read_error: str | None = None
    if index_available:
        try:
            chunks_sha256 = _sha256_file(index_file)
        except Exception as exc:
            index_read_error = _error_summary(exc, persist_dir)
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "mode": "keyword",
        "required": bool(required),
        "index_available": index_available,
        "index_usable": False,
        "query_attempted": False,
        "query_succeeded": False,
        "returned_count": 0,
        "topk": 1,
        "query_sha256": query_sha256(MONAI_RAG_SMOKE_QUERY),
        "chunks_sha256": chunks_sha256,
        "error": None,
        "ok": False,
        "receipt_written": True,
    }

    if not receipt["index_available"]:
        receipt["error"] = "FileNotFoundError: MONAI RAG chunks index is missing"
    elif index_read_error is not None:
        receipt["error"] = index_read_error
    else:
        try:
            factory = rag_factory or MED_KNOWLEDGE_RAG
            rag = factory(persist_dir=persist_dir, mode="keyword")
            receipt["index_usable"] = True
            receipt["query_attempted"] = True
            if hasattr(rag, "search_with_count"):
                _, returned_count = rag.search_with_count(
                    MONAI_RAG_SMOKE_QUERY,
                    k=1,
                    enable_query2doc=False,
                )
            else:
                result = rag.search(
                    MONAI_RAG_SMOKE_QUERY,
                    k=1,
                    enable_query2doc=False,
                )
                returned_count = _legacy_result_count(result)
            if (
                isinstance(returned_count, bool)
                or not isinstance(returned_count, int)
                or returned_count < 0
            ):
                raise ValueError(
                    f"RAG returned an invalid document count: {returned_count!r}"
                )
            receipt["query_succeeded"] = True
            receipt["returned_count"] = returned_count
            if returned_count == 0:
                receipt["error"] = (
                    "LookupError: MONAI RAG smoke query returned no documents"
                )
        except Exception as exc:
            if receipt["error"] is None:
                receipt["error"] = _error_summary(exc, persist_dir)

    receipt["ok"] = bool(
        receipt["index_available"]
        and receipt["index_usable"]
        and receipt["query_attempted"]
        and receipt["query_succeeded"]
        and receipt["returned_count"] > 0
    )

    try:
        _atomic_write_json(receipt_path, receipt)
    except Exception as exc:
        receipt["receipt_written"] = False
        if required:
            raise MonaiRagEvidenceError(
                "Required MONAI RAG smoke evidence could not be written.",
                receipt=receipt,
                receipt_path=receipt_path,
            ) from exc
        return receipt

    if required and not receipt["ok"]:
        raise MonaiRagEvidenceError(
            "Required MONAI RAG smoke evidence failed; see the smoke receipt.",
            receipt=receipt,
            receipt_path=receipt_path,
        )
    return receipt
