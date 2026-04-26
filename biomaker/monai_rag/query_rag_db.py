import json
import re
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


@dataclass
class RagDocument:
    page_content: str
    metadata: dict


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", text)}


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

    def _score(self, query_tokens: set[str], doc: RagDocument) -> float:
        text = doc.page_content.lower()
        doc_tokens = _tokenize(text)
        overlap = len(query_tokens & doc_tokens)
        phrase_bonus = sum(1 for token in query_tokens if token in text)
        title = str(doc.metadata.get("title", "")).lower()
        title_tokens = _tokenize(title)
        title_bonus = sum(
            8
            for token in query_tokens
            if token in title_tokens and token not in GENERIC_TITLE_TOKENS
        )
        return overlap + phrase_bonus + title_bonus

    def _search_once(self, search_query: str, k: int) -> list[RagDocument]:
        query_tokens = _tokenize(search_query)
        if not query_tokens:
            return []

        scored = [
            (self._score(query_tokens, doc), doc)
            for doc in self.documents
            if doc.page_content.strip()
        ]
        scored = [(score, doc) for score, doc in scored if score > 0]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored[:k]]

    def search(self, search_query, k=3, enable_query2doc=False, convert_to_str=True):
        documents = self._search_once(str(search_query), k)

        if enable_query2doc:
            try:
                expanded_query = query2doc_search(search_query, model=self.model)
                documents.extend(self._search_once(str(expanded_query), k))
            except Exception:
                pass

        deduped = []
        seen = set()
        for doc in documents:
            key = doc.page_content[:200]
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
                f"{doc.page_content}"
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
