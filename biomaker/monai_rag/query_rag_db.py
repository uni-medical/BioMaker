import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from biomaker.backend import query
from biomaker.utils.response import truncate_text

PERSIST_DIR = Path(__file__).with_name("monai_rag_db")
INDEX_FILE = PERSIST_DIR / "chunks.jsonl"
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
    def __init__(self, model="gpt-4o", persist_dir=PERSIST_DIR):
        self.model = model
        self.index_file = Path(persist_dir) / "chunks.jsonl"
        self.documents = _load_documents(self.index_file)
        self._doc_tokens: dict[int, set[str]] = {}
        self._title_tokens: dict[int, set[str]] = {}
        self._normalized_title: dict[int, str] = {}
        self._normalized_text: dict[int, str] = {}
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

    def _search_once(self, search_query: str, k: int) -> list[RagDocument]:
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
        return [doc for _, doc in scored[:k]]

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

    def search(self, search_query, k=3, enable_query2doc=False, convert_to_str=True):
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
                f"Source: {doc.metadata.get('source', '')}\n"
                f"Title: {doc.metadata.get('title', '')}\n"
                f"Snippet: {self._focused_snippet(doc, str(search_query))}"
            )
            for doc in final_documents
        )
        return truncate_text(result, max_chars=2500, marker="MONAI RAG trimmed")


if __name__ == "__main__":
    rag = MED_KNOWLEDGE_RAG()
    test_queries = [("The basic knowledge of MONAI Data Pipeline usage", 5)]
    for q, k in test_queries:
        print(f"\n=== Query: {q} ===")
        results = rag.search(q, k=k, enable_query2doc=False, convert_to_str=False)
        for i, doc in enumerate(results, start=1):
            print(f"[Result {i}] {doc.metadata.get('source', '')}")
            print(f"Preview: {doc.page_content[:200]}...")
            print("-" * 50)
