import argparse
import hashlib
import json
import inspect
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from bs4 import BeautifulSoup

from biomaker.utils.env import load_local_env

INDEX_DIR = Path(__file__).with_name("monai_rag_db")
CHUNKS_FILE_NAME = "chunks.jsonl"
EMBEDDINGS_FILE_NAME = "embeddings.npy"
MANIFEST_FILE_NAME = "manifest.json"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
RAG_MODES = {"keyword", "embedding"}


def _validate_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in RAG_MODES:
        choices = ", ".join(sorted(RAG_MODES))
        raise ValueError(f"RAG mode must be one of {choices}; got {mode!r}.")
    return normalized


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _default_embedding_client():
    """Create an OpenAI-compatible embedding client from the dedicated env first."""
    load_local_env()
    api_key = os.getenv("BIOMAKER_EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("BIOMAKER_EMBEDDING_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    if not api_key:
        raise RuntimeError(
            "Embedding retrieval requires BIOMAKER_EMBEDDING_API_KEY or "
            "OPENAI_API_KEY. Configure a provider that supports the "
            "OpenAI-compatible /embeddings endpoint."
        )

    from openai import OpenAI

    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def _embedding_endpoint_label() -> str:
    if os.getenv("BIOMAKER_EMBEDDING_BASE_URL"):
        return "BIOMAKER_EMBEDDING_BASE_URL"
    if os.getenv("OPENAI_BASE_URL"):
        return "OPENAI_BASE_URL"
    return "the OpenAI default endpoint"


def _extract_embedding_vectors(response: Any, expected_count: int) -> np.ndarray:
    """Normalize callable, OpenAI, and LangChain-style embedding responses."""
    if isinstance(response, dict) and "data" in response:
        entries = response["data"]
    elif hasattr(response, "data"):
        entries = response.data
    else:
        entries = response

    if entries is None:
        raise ValueError("The embedding provider returned no data.")

    entries = list(entries)
    if entries and (
        isinstance(entries[0], dict)
        and "embedding" in entries[0]
        or hasattr(entries[0], "embedding")
    ):
        entries.sort(
            key=lambda item: (
                item.get("index", 0)
                if isinstance(item, dict)
                else getattr(item, "index", 0)
            )
        )
        vectors = [
            item["embedding"] if isinstance(item, dict) else getattr(item, "embedding")
            for item in entries
        ]
    else:
        vectors = entries

    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim == 1 and expected_count == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise ValueError(
            f"Embedding provider returned an array with shape {matrix.shape}; "
            "expected a two-dimensional batch."
        )
    if matrix.shape[0] != expected_count:
        raise ValueError(
            f"Embedding provider returned {matrix.shape[0]} vectors for "
            f"{expected_count} texts."
        )
    if matrix.shape[1] < 1:
        raise ValueError("Embedding provider returned zero-dimensional vectors.")
    if not np.isfinite(matrix).all():
        raise ValueError("Embedding provider returned non-finite vector values.")
    return matrix


def _embed_texts(
    texts: list[str],
    *,
    model: str,
    embedding_client=None,
) -> np.ndarray:
    """Embed one batch; a callable client enables fully offline tests."""
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    client = embedding_client or _default_embedding_client()
    try:
        if callable(client):
            response = client(texts, model=model)
        elif hasattr(client, "embed_documents"):
            response = client.embed_documents(texts)
        elif hasattr(client, "embeddings") and hasattr(client.embeddings, "create"):
            response = client.embeddings.create(model=model, input=texts)
        elif hasattr(client, "create"):
            response = client.create(model=model, input=texts)
        else:
            raise TypeError(
                "embedding_client must be callable or expose "
                "embeddings.create, create, or embed_documents."
            )
        return _extract_embedding_vectors(response, len(texts))
    except Exception as exc:
        if isinstance(exc, (TypeError, ValueError)) and embedding_client is not None:
            raise
        raise RuntimeError(
            f"Embedding request failed for model {model!r} via "
            f"{_embedding_endpoint_label()}. Verify that the configured endpoint "
            "supports the OpenAI-compatible /embeddings API and this model. "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _embed_texts_in_batches(
    texts: list[str],
    *,
    model: str,
    embedding_client=None,
    batch_size: int = 64,
) -> np.ndarray:
    if batch_size < 1:
        raise ValueError(f"embedding_batch_size must be >= 1, got {batch_size}.")
    batches = []
    dimension = None
    for start in range(0, len(texts), batch_size):
        batch = _embed_texts(
            texts[start : start + batch_size],
            model=model,
            embedding_client=embedding_client,
        )
        if dimension is None:
            dimension = batch.shape[1]
        elif batch.shape[1] != dimension:
            raise ValueError(
                "Embedding dimensions changed between batches: "
                f"expected {dimension}, got {batch.shape[1]}."
            )
        batches.append(batch)
    if not batches:
        raise ValueError("Cannot build a RAG index from an empty document corpus.")
    return np.concatenate(batches, axis=0)


def _clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _split_text(text: str, chunk_size=1200, overlap=150) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(end - overlap, start + 1)
    return chunks


def parse_monai_docs(root_dir="docs.monai.io/en/1.4.0") -> list[dict]:
    docs = []
    for root, _, files in os.walk(root_dir):
        for file in sorted(files):
            if file.startswith("whatsnew") or not file.endswith(".html"):
                continue

            path = Path(root) / file
            with open(path, encoding="utf-8") as f:
                soup = BeautifulSoup(f.read(), "html.parser")

            for elem in soup(["nav", "footer", "script", "style"]):
                elem.decompose()

            main_content = soup.select_one(':is(div, main)[role="main"]')
            if not main_content:
                main_content = soup.body
            if not main_content:
                continue

            title = soup.title.get_text(" ", strip=True) if soup.title else ""
            text = _clean_text(main_content.get_text("\n", strip=True))
            if not text:
                continue

            docs.append(
                {
                    "page_content": text,
                    "metadata": {
                        "source": str(path),
                        "version": "1.4.0",
                        "title": title,
                    },
                }
            )
    return docs


def parse_installed_monai_docs() -> list[dict]:
    """Build a small documentation corpus from the installed MONAI package."""
    import monai.data
    import monai.losses
    import monai.networks.nets
    import monai.transforms

    modules = [
        monai.data,
        monai.transforms,
        monai.networks.nets,
        monai.losses,
    ]
    docs = []
    for module in modules:
        for name in sorted(dir(module)):
            if name.startswith("_"):
                continue
            obj = getattr(module, name)
            if not (inspect.isclass(obj) or inspect.isfunction(obj)):
                continue
            doc = inspect.getdoc(obj)
            if not doc:
                continue
            title = f"{module.__name__}.{name}"
            details = [title]
            try:
                details.append(f"Signature: {name}{inspect.signature(obj)}")
            except (TypeError, ValueError):
                pass
            details.append(doc)
            try:
                source = inspect.getsource(obj)
                details.append(f"Source excerpt:\n{_clean_text(source)[:6000]}")
            except (OSError, TypeError):
                pass
            docs.append(
                {
                    "page_content": _clean_text("\n\n".join(details)),
                    "metadata": {
                        "source": "installed-monai",
                        "version": getattr(__import__("monai"), "__version__", ""),
                        "title": title,
                    },
                }
            )
    return docs


def parse_and_index_docs(
    persist_dir=INDEX_DIR,
    *,
    mode: str = "keyword",
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_client=None,
    embedding_batch_size: int = 64,
):
    mode = _validate_mode(mode)
    embedding_model = str(embedding_model).strip()
    if mode == "embedding" and not embedding_model:
        raise ValueError("embedding_model must be set when RAG mode is embedding.")

    documents = parse_monai_docs()
    if not documents:
        documents = parse_installed_monai_docs()
    records = []
    for doc in documents:
        for split_id, chunk in enumerate(_split_text(doc["page_content"])):
            metadata = dict(doc["metadata"])
            metadata["split_id"] = split_id
            records.append({"page_content": chunk, "metadata": metadata})
    if not records:
        raise ValueError("Cannot build a RAG index from an empty MONAI corpus.")

    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    source_counts = Counter(record["metadata"].get("source", "") for record in records)
    document_versions = sorted(
        {
            str(record["metadata"].get("version", "")).strip()
            for record in records
            if str(record["metadata"].get("version", "")).strip()
        }
    )
    chunks_path = persist_dir / CHUNKS_FILE_NAME
    embeddings_path = persist_dir / EMBEDDINGS_FILE_NAME
    manifest_path = persist_dir / MANIFEST_FILE_NAME
    chunks_tmp = persist_dir / f".{CHUNKS_FILE_NAME}.tmp"
    embeddings_tmp = persist_dir / f".{EMBEDDINGS_FILE_NAME}.tmp"
    manifest_tmp = persist_dir / f".{MANIFEST_FILE_NAME}.tmp"

    try:
        with chunks_tmp.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        dimension = None
        embeddings_sha256 = None
        embedding_dtype = None
        if mode == "embedding":
            embeddings = _embed_texts_in_batches(
                [record["page_content"] for record in records],
                model=embedding_model,
                embedding_client=embedding_client,
                batch_size=embedding_batch_size,
            ).astype(np.float32, copy=False)
            dimension = int(embeddings.shape[1])
            embedding_dtype = str(embeddings.dtype)
            with embeddings_tmp.open("wb") as f:
                np.save(f, embeddings, allow_pickle=False)
            embeddings_sha256 = _sha256_file(embeddings_tmp)

        manifest = {
            "schema_version": 1,
            "version": document_versions[0] if len(document_versions) == 1 else None,
            "document_versions": document_versions,
            "record_count": len(records),
            "source_counts": dict(source_counts),
            "mode": mode,
            "embedding_model": embedding_model if mode == "embedding" else None,
            "dimension": dimension,
            "chunks_sha256": _sha256_file(chunks_tmp),
            "embeddings_sha256": embeddings_sha256,
            "embedding_dtype": embedding_dtype,
        }
        with manifest_tmp.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
            f.write("\n")

        chunks_tmp.replace(chunks_path)
        if mode == "embedding":
            embeddings_tmp.replace(embeddings_path)
        else:
            embeddings_path.unlink(missing_ok=True)
        manifest_tmp.replace(manifest_path)
    finally:
        chunks_tmp.unlink(missing_ok=True)
        embeddings_tmp.unlink(missing_ok=True)
        manifest_tmp.unlink(missing_ok=True)

    detail = (
        f" using {embedding_model} ({dimension} dimensions)"
        if mode == "embedding"
        else ""
    )
    print(f"Indexed {len(records)} MONAI documentation chunks in {mode} mode{detail}.")


def _parse_args():
    parser = argparse.ArgumentParser(description="Build the local MONAI RAG index.")
    parser.add_argument(
        "--mode",
        choices=sorted(RAG_MODES),
        default=os.getenv("BIOMAKER_RAG_MODE", "keyword"),
        help="Retrieval backend to build (default: keyword).",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("BIOMAKER_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        help="OpenAI-compatible embedding model used in embedding mode.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=os.getenv("BIOMAKER_EMBEDDING_BATCH_SIZE", "64"),
        help="Number of documentation chunks per embedding request.",
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=INDEX_DIR,
        help="Directory for chunks.jsonl, embeddings.npy, and manifest.json.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    parse_and_index_docs(
        persist_dir=args.persist_dir,
        mode=args.mode,
        embedding_model=args.embedding_model,
        embedding_batch_size=args.embedding_batch_size,
    )
