import argparse
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from biomaker.backend import query
from biomaker.monai_rag.create_monai_rag_db import (
    CHUNKS_FILE_NAME,
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDINGS_FILE_NAME,
    MANIFEST_FILE_NAME,
    RAG_MODES,
    _embed_texts,
    _sha256_file,
    _validate_mode,
)
from biomaker.utils.response import truncate_text

PERSIST_DIR = Path(__file__).with_name("monai_rag_db")
INDEX_FILE = PERSIST_DIR / CHUNKS_FILE_NAME
GENERIC_TITLE_TOKENS = {
    "monai",
    "data",
    "transforms",
    "networks",
    "nets",
    "losses",
    "metrics",
}
LOW_SIGNAL_QUERY_TOKENS = GENERIC_TITLE_TOKENS | {
    "medical",
    "image",
    "images",
    "classification",
    "segmentation",
    "task",
    "tasks",
    "input",
    "format",
    "expected",
    "best",
    "practice",
    "practices",
    "usage",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "without",
    "using",
    "confirm",
    "ensure",
    "runtimeerror",
    "error",
    "applying",
    "transform",
    "transforms",
}


@dataclass
class RagDocument:
    page_content: str
    metadata: dict
    retrieval_note: str = ""


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", text)}


def _normalize_symbol(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _extract_query_symbols(text: str) -> set[str]:
    """Extract exact API-like symbols from a natural language or traceback query."""
    symbols: set[str] = set()
    patterns = [
        r"\b__[A-Za-z0-9_]+__\b",
        r"\b[A-Za-z_][A-Za-z0-9_]*\.py\b",
        r"\b(?:monai|torch|torchvision|numpy|pandas|sklearn)(?:\.[A-Za-z_][A-Za-z0-9_]*)+",
        r"\b[A-Z][A-Za-z0-9_]*[a-z][A-Za-z0-9_]*\b",
    ]
    for pattern in patterns:
        for match in re.findall(pattern, text):
            normalized = _normalize_symbol(match)
            if len(normalized) >= 4 and normalized not in LOW_SIGNAL_QUERY_TOKENS:
                symbols.add(normalized)
            if match.endswith(".py"):
                module_symbol = _normalize_symbol(match.removesuffix(".py"))
                if len(module_symbol) >= 4:
                    symbols.add(module_symbol)
    return symbols


def _extract_raw_query_symbols(text: str) -> list[str]:
    raw: list[str] = []
    seen: set[str] = set()
    patterns = [
        r"\b__[A-Za-z0-9_]+__\b",
        r"\b[A-Za-z_][A-Za-z0-9_]*\.py\b",
        r"\b(?:monai|torch|torchvision|numpy|pandas|sklearn)(?:\.[A-Za-z_][A-Za-z0-9_]*)+",
        r"\b[A-Z][A-Za-z0-9_]*[a-z][A-Za-z0-9_]*\b",
    ]
    for pattern in patterns:
        for match in re.findall(pattern, text):
            normalized = _normalize_symbol(match)
            if len(normalized) < 4 or normalized in LOW_SIGNAL_QUERY_TOKENS:
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            raw.append(match)
    return raw


def _load_documents(index_file: Path = INDEX_FILE) -> list[RagDocument]:
    if not index_file.exists():
        raise FileNotFoundError(
            f"MONAI RAG index not found at {index_file}. "
            "Run `bash scripts/create_rag_db.sh` to build it."
        )

    documents = []
    with open(index_file, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            documents.append(
                RagDocument(
                    page_content=record["page_content"],
                    metadata=record.get("metadata", {}),
                )
            )
    if not documents:
        raise ValueError(f"MONAI RAG index is empty at {index_file}.")
    return documents


def _load_manifest(manifest_path: Path) -> dict:
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Embedding RAG manifest not found at {manifest_path}. "
            "Rebuild it with `bash scripts/create_rag_db.sh`."
        )
    try:
        with manifest_path.open(encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as exc:
        raise ValueError(
            f"Embedding RAG manifest is invalid at {manifest_path}: {exc}"
        ) from exc
    if not isinstance(manifest, dict):
        raise ValueError(
            f"Embedding RAG manifest must be a JSON object at {manifest_path}."
        )
    return manifest


def _load_embedding_index(
    persist_dir: Path,
    *,
    embedding_model: str,
    documents: list[RagDocument],
) -> np.ndarray:
    chunks_path = persist_dir / CHUNKS_FILE_NAME
    embeddings_path = persist_dir / EMBEDDINGS_FILE_NAME
    manifest_path = persist_dir / MANIFEST_FILE_NAME
    manifest = _load_manifest(manifest_path)

    if manifest.get("mode") != "embedding":
        raise ValueError(
            f"RAG index at {persist_dir} was built for mode "
            f"{manifest.get('mode')!r}, not 'embedding'. Rebuild the index in "
            "embedding mode or select keyword mode explicitly."
        )
    manifest_model = manifest.get("embedding_model")
    if manifest_model != embedding_model:
        raise ValueError(
            f"RAG index embedding model mismatch: configured {embedding_model!r}, "
            f"but {manifest_path} records {manifest_model!r}. Rebuild the index "
            "with the configured model."
        )
    if manifest.get("record_count") != len(documents):
        raise ValueError(
            "RAG index record count mismatch: "
            f"manifest={manifest.get('record_count')!r}, chunks={len(documents)}. "
            "Rebuild the index."
        )

    expected_chunks_hash = manifest.get("chunks_sha256")
    actual_chunks_hash = _sha256_file(chunks_path)
    if not expected_chunks_hash or expected_chunks_hash != actual_chunks_hash:
        raise ValueError(
            "RAG chunks checksum does not match manifest.json. Rebuild the index; "
            "embedding vectors must never be paired with different chunks."
        )

    if not embeddings_path.is_file():
        raise FileNotFoundError(
            f"Embedding RAG vectors not found at {embeddings_path}. "
            "Rebuild them with `bash scripts/create_rag_db.sh`."
        )
    expected_embeddings_hash = manifest.get("embeddings_sha256")
    actual_embeddings_hash = _sha256_file(embeddings_path)
    if (
        not expected_embeddings_hash
        or expected_embeddings_hash != actual_embeddings_hash
    ):
        raise ValueError(
            "RAG embeddings checksum does not match manifest.json. Rebuild the index."
        )

    try:
        matrix = np.load(embeddings_path, allow_pickle=False)
    except Exception as exc:
        raise ValueError(
            f"Unable to load embedding RAG vectors from {embeddings_path}: {exc}"
        ) from exc
    if matrix.ndim != 2:
        raise ValueError(
            f"Embedding RAG vectors must be two-dimensional, got shape {matrix.shape}."
        )
    expected_dimension = manifest.get("dimension")
    if (
        isinstance(expected_dimension, bool)
        or not isinstance(expected_dimension, int)
        or expected_dimension < 1
    ):
        raise ValueError(
            f"Embedding RAG manifest has invalid dimension {expected_dimension!r}."
        )
    expected_shape = (len(documents), expected_dimension)
    if matrix.shape != expected_shape:
        raise ValueError(
            f"Embedding RAG vector shape mismatch: expected {expected_shape}, "
            f"got {matrix.shape}. Rebuild the index."
        )
    expected_dtype = manifest.get("embedding_dtype")
    if not expected_dtype or str(matrix.dtype) != expected_dtype:
        raise ValueError(
            "Embedding RAG dtype mismatch: "
            f"manifest={expected_dtype!r}, vectors={str(matrix.dtype)!r}. "
            "Rebuild the index."
        )
    if not np.issubdtype(matrix.dtype, np.floating) or not np.isfinite(matrix).all():
        raise ValueError("Embedding RAG vectors must contain only finite floats.")

    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms <= 0):
        raise ValueError("Embedding RAG index contains zero-length vectors.")
    return matrix.astype(np.float32, copy=False) / norms[:, None]


def query2doc_search(search_query, model="gpt-4o"):
    prompt = (
        "Rewrite the query as a short MONAI documentation search passage. "
        "Focus on API names, data pipeline classes, transforms, networks, losses, "
        "and likely error keywords. Output only the passage."
    )
    return query(
        system_message=prompt,
        user_message=str(search_query),
        model=model,
        temperature=0.0,
    )


class MED_KNOWLEDGE_RAG:
    def __init__(
        self,
        model="gpt-4o",
        persist_dir=PERSIST_DIR,
        mode: str = "keyword",
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        embedding_client=None,
    ):
        self.model = model
        self.mode = _validate_mode(mode)
        self.embedding_model = str(embedding_model).strip()
        self.embedding_client = embedding_client
        self.persist_dir = Path(persist_dir)
        self.index_file = self.persist_dir / CHUNKS_FILE_NAME
        self.documents = _load_documents(self.index_file)
        self._doc_tokens: dict[int, set[str]] = {}
        self._title_tokens: dict[int, set[str]] = {}
        self._normalized_title: dict[int, str] = {}
        self._normalized_text: dict[int, str] = {}
        self._embedding_matrix: np.ndarray | None = None

        if self.mode == "embedding":
            if not self.embedding_model:
                raise ValueError(
                    "embedding_model must be set when RAG mode is embedding."
                )
            self._embedding_matrix = _load_embedding_index(
                self.persist_dir,
                embedding_model=self.embedding_model,
                documents=self.documents,
            )
            self._doc_freq = Counter()
            self._num_docs = max(1, len(self.documents))
            return

        doc_freq: Counter[str] = Counter()
        for doc in self.documents:
            key = id(doc)
            tokens = _tokenize(doc.page_content)
            title_tokens = _tokenize(str(doc.metadata.get("title", "")))
            self._doc_tokens[key] = tokens
            self._title_tokens[key] = title_tokens
            self._normalized_title[key] = _normalize_symbol(
                str(doc.metadata.get("title", ""))
            )
            self._normalized_text[key] = _normalize_symbol(doc.page_content)
            doc_freq.update(tokens | title_tokens)
        self._doc_freq = doc_freq
        self._num_docs = max(1, len(self.documents))

    def _idf(self, token: str) -> float:
        return math.log((self._num_docs + 1) / (self._doc_freq.get(token, 0) + 1)) + 1.0

    def _score(
        self, query_tokens: set[str], query_symbols: set[str], doc: RagDocument
    ) -> float:
        key = id(doc)
        doc_tokens = self._doc_tokens[key]
        title_tokens = self._title_tokens[key]
        normalized_title = self._normalized_title[key]
        normalized_text = self._normalized_text[key]

        score = 0.0
        for token in query_tokens:
            if token not in doc_tokens and token not in title_tokens:
                continue
            weight = self._idf(token)
            if token in LOW_SIGNAL_QUERY_TOKENS:
                weight *= 0.15
            if token in doc_tokens:
                score += weight
            if token in title_tokens and token not in GENERIC_TITLE_TOKENS:
                score += weight * 8.0

        for symbol in query_symbols:
            if symbol in normalized_title:
                score += 60.0
            elif symbol in normalized_text:
                score += 15.0
        return score

    def _search_keyword_once(self, search_query: str, k: int) -> list[RagDocument]:
        query_tokens = _tokenize(search_query)
        query_symbols = _extract_query_symbols(search_query)
        if not query_tokens:
            return []
        if not query_symbols:
            # This index is API/traceback oriented. Broad tutorial-style queries tend to
            # retrieve unrelated API pages by shared words, which is worse than no hit.
            return []

        scored = [
            (self._score(query_tokens, query_symbols, doc), doc)
            for doc in self.documents
            if doc.page_content.strip()
        ]
        scored = [(score, doc) for score, doc in scored if score > 0]
        scored.sort(key=lambda item: item[0], reverse=True)
        raw_symbols = _extract_raw_query_symbols(search_query)
        ranked = []
        for score, doc in scored[:k]:
            key = id(doc)
            matched_symbols = [
                symbol
                for symbol in raw_symbols
                if _normalize_symbol(symbol) in self._normalized_title[key]
                or _normalize_symbol(symbol) in self._normalized_text[key]
            ]
            note = (
                f"Keyword/API match: {', '.join(matched_symbols)}"
                if matched_symbols
                else f"Weighted keyword score: {score:.3f}"
            )
            ranked.append(
                RagDocument(
                    page_content=doc.page_content,
                    metadata=dict(doc.metadata),
                    retrieval_note=note,
                )
            )
        return ranked

    def _search_embedding_once(self, search_query: str, k: int) -> list[RagDocument]:
        if not search_query.strip() or k <= 0:
            return []
        if self._embedding_matrix is None:
            raise RuntimeError("Embedding RAG index was not loaded.")
        query_vector = _embed_texts(
            [search_query],
            model=self.embedding_model,
            embedding_client=self.embedding_client,
        )[0]
        expected_dimension = self._embedding_matrix.shape[1]
        if query_vector.shape != (expected_dimension,):
            raise ValueError(
                "Embedding query dimension mismatch: "
                f"index={expected_dimension}, query={query_vector.shape[0]}. "
                "Ensure indexing and querying use the same embedding model and endpoint."
            )
        norm = float(np.linalg.norm(query_vector))
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("Embedding provider returned a zero-length query vector.")
        scores = self._embedding_matrix @ (query_vector / norm)
        order = np.argsort(-scores, kind="stable")[: min(k, len(self.documents))]
        return [
            RagDocument(
                page_content=self.documents[int(index)].page_content,
                metadata=dict(self.documents[int(index)].metadata),
                retrieval_note=(
                    f"Embedding cosine similarity: {float(scores[int(index)]):.4f}"
                ),
            )
            for index in order
        ]

    def _search_once(self, search_query: str, k: int) -> list[RagDocument]:
        if self.mode == "embedding":
            return self._search_embedding_once(search_query, k)
        return self._search_keyword_once(search_query, k)

    def _dedupe_key(self, doc: RagDocument) -> str:
        title = str(doc.metadata.get("title", ""))
        module, _, name = title.rpartition(".")
        if module == "monai.transforms":
            name = re.sub(r"(Dict|D|d)$", "", name)
        return f"{module}.{name}" if module else title or doc.page_content[:200]

    def _focused_snippet(
        self,
        doc: RagDocument,
        search_query: str,
        max_chars: int = 850,
    ) -> str:
        content = doc.page_content.strip()
        if len(content) <= max_chars:
            return content

        lowered = content.lower()
        raw_symbols = _extract_raw_query_symbols(search_query)
        raw_symbols.sort(
            key=lambda symbol: (
                0 if symbol.startswith("__") else 1 if symbol.endswith(".py") else 2,
                len(symbol),
            )
        )
        query_tokens = [
            token
            for token in _tokenize(search_query)
            if token not in LOW_SIGNAL_QUERY_TOKENS
        ]
        position = -1
        for symbol in raw_symbols:
            position = lowered.find(symbol.lower())
            if position >= 0:
                break
        if position < 0:
            for token in sorted(query_tokens, key=len, reverse=True):
                position = lowered.find(token.lower())
                if position >= 0:
                    break
        if position < 0:
            position = 0

        start = max(0, position - max_chars // 3)
        end = min(len(content), start + max_chars)
        if end - start < max_chars:
            start = max(0, end - max_chars)
        snippet = content[start:end].strip()
        if start > 0:
            snippet = "... " + snippet
        if end < len(content):
            snippet = snippet + " ..."
        return snippet

    def search(self, search_query, k=5, enable_query2doc=False, convert_to_str=True):
        k = int(k)
        if k <= 0:
            return [] if not convert_to_str else ""
        candidate_k = max(k * 5, k + 5)
        documents = self._search_once(str(search_query), candidate_k)

        if enable_query2doc:
            try:
                expanded_query = query2doc_search(search_query, model=self.model)
                documents.extend(self._search_once(str(expanded_query), candidate_k))
            except Exception:
                pass

        deduped = []
        seen = set()
        for doc in documents:
            key = self._dedupe_key(doc)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(doc)

        final_documents = deduped[:k]
        if not final_documents:
            return [] if not convert_to_str else ""

        if not convert_to_str:
            return final_documents

        result = "\n\n---\n\n".join(
            (
                "Source: "
                f"{truncate_text(str(doc.metadata.get('source', '')), max_chars=240, marker='source trimmed')}\n"
                "Title: "
                f"{truncate_text(str(doc.metadata.get('title', '')), max_chars=240, marker='title trimmed')}\n"
                "Match: "
                f"{truncate_text(doc.retrieval_note, max_chars=240, marker='match note trimmed')}\n"
                f"Snippet: {self._focused_snippet(doc, str(search_query))}"
            )
            for doc in final_documents
        )
        result_budget = max(2500, min(len(final_documents), 10) * 1600)
        return truncate_text(
            result,
            max_chars=result_budget,
            marker="MONAI RAG trimmed",
        )


def _manifest_cli_defaults(persist_dir: Path) -> tuple[str, str]:
    mode = os.getenv("BIOMAKER_RAG_MODE") or "keyword"
    embedding_model = os.getenv("BIOMAKER_EMBEDDING_MODEL")
    manifest_path = persist_dir / MANIFEST_FILE_NAME
    if not embedding_model and manifest_path.is_file():
        try:
            with manifest_path.open(encoding="utf-8") as f:
                manifest = json.load(f)
            if not embedding_model and manifest.get("embedding_model"):
                embedding_model = str(manifest["embedding_model"])
        except Exception:
            pass
    return mode, embedding_model or DEFAULT_EMBEDDING_MODEL


def _parse_args():
    parser = argparse.ArgumentParser(description="Query the local MONAI RAG index.")
    parser.add_argument(
        "query",
        nargs="?",
        default="monai.transforms.Compose API signature",
        help="MONAI documentation query.",
    )
    parser.add_argument(
        "--mode",
        choices=sorted(RAG_MODES),
        default=None,
        help="Retrieval backend (default: BIOMAKER_RAG_MODE or keyword).",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Embedding model used to build and query an embedding index.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=int(os.getenv("BIOMAKER_RAG_TOPK", "5")),
        help="Number of chunks to return.",
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=Path(os.getenv("BIOMAKER_RAG_INDEX_DIR", str(PERSIST_DIR))),
        help="Directory containing the local RAG index.",
    )
    args = parser.parse_args()
    default_mode, default_embedding_model = _manifest_cli_defaults(args.persist_dir)
    args.mode = args.mode or default_mode
    args.embedding_model = args.embedding_model or default_embedding_model
    return args


if __name__ == "__main__":
    args = _parse_args()
    rag = MED_KNOWLEDGE_RAG(
        persist_dir=args.persist_dir,
        mode=args.mode,
        embedding_model=args.embedding_model,
    )
    print(f"\n=== Query ({args.mode}): {args.query} ===")
    results = rag.search(
        args.query,
        k=args.top_k,
        enable_query2doc=False,
        convert_to_str=False,
    )
    for i, doc in enumerate(results, start=1):
        print(f"[Result {i}] {doc.metadata.get('source', '')}")
        print(f"Preview: {doc.page_content[:200]}...")
        print("-" * 50)
