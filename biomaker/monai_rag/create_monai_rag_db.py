import json
import inspect
import os
import re
from collections import Counter
from pathlib import Path

from bs4 import BeautifulSoup

INDEX_DIR = Path(__file__).with_name("monai_rag_db")


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
            docs.append(
                {
                    "page_content": _clean_text(f"{title}\n\n{doc}"),
                    "metadata": {
                        "source": "installed-monai",
                        "version": getattr(__import__("monai"), "__version__", ""),
                        "title": title,
                    },
                }
            )
    return docs


def parse_and_index_docs(persist_dir=INDEX_DIR):
    documents = parse_monai_docs()
    if not documents:
        documents = parse_installed_monai_docs()
    records = []
    for doc in documents:
        for split_id, chunk in enumerate(_split_text(doc["page_content"])):
            metadata = dict(doc["metadata"])
            metadata["split_id"] = split_id
            records.append({"page_content": chunk, "metadata": metadata})

    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    with open(persist_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    source_counts = Counter(record["metadata"].get("source", "") for record in records)
    manifest = {
        "version": "1.4.0",
        "record_count": len(records),
        "source_counts": dict(source_counts),
    }
    with open(persist_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Indexed {len(records)} MONAI documentation chunks.")


if __name__ == "__main__":
    parse_and_index_docs()
