import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import biomaker.monai_rag.query_rag_db as query_rag_db
from biomaker.agents.proposal_agent import ProposalAgent
from biomaker.journal import Journal
from biomaker.monai_rag import create_monai_rag_db
from biomaker.monai_rag.query_rag_db import MED_KNOWLEDGE_RAG
from biomaker.utils.config import _is_usable_rag_index, _load_cfg

EMBEDDING_MODEL = "text-embedding-3-large"


def _documents() -> list[dict]:
    return [
        {
            "page_content": (
                "monai.transforms.EnsureChannelFirstd ensures that image tensors "
                "have a channel-first layout."
            ),
            "metadata": {
                "source": "transforms.html",
                "version": "1.4.0",
                "title": "monai.transforms.EnsureChannelFirstd",
            },
        },
        {
            "page_content": (
                "monai.losses.DiceLoss computes a soft Dice objective for "
                "segmentation."
            ),
            "metadata": {
                "source": "losses.html",
                "version": "1.4.0",
                "title": "monai.losses.DiceLoss",
            },
        },
    ]


def _offline_embed(texts: list[str], model: str = EMBEDDING_MODEL) -> list[list[float]]:
    assert model == EMBEDDING_MODEL
    embeddings = []
    for text in texts:
        lowered = text.lower()
        if "channel" in lowered or "ensurechannelfirst" in lowered:
            embeddings.append([1.0, 0.0, 0.0])
        elif "dice" in lowered:
            embeddings.append([0.0, 1.0, 0.0])
        else:
            embeddings.append([0.0, 0.0, 1.0])
    return embeddings


class MonaiRagTests(unittest.TestCase):
    def test_embedding_index_when_selected_and_search_are_offline(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(create_monai_rag_db, "parse_monai_docs", _documents),
        ):
            index_dir = Path(tmp)
            create_monai_rag_db.parse_and_index_docs(
                persist_dir=index_dir,
                mode="embedding",
                embedding_client=_offline_embed,
            )

            manifest = json.loads(
                (index_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["mode"], "embedding")
            self.assertEqual(manifest["embedding_model"], EMBEDDING_MODEL)
            self.assertEqual(manifest["version"], "1.4.0")
            self.assertEqual(manifest["document_versions"], ["1.4.0"])
            self.assertEqual(manifest["record_count"], 2)
            self.assertEqual(manifest["dimension"], 3)
            self.assertEqual(manifest["embedding_dtype"], "float32")
            self.assertEqual(len(manifest["chunks_sha256"]), 64)
            self.assertEqual(len(manifest["embeddings_sha256"]), 64)

            stored_embeddings = np.load(
                index_dir / "embeddings.npy", allow_pickle=False
            )
            self.assertEqual(stored_embeddings.shape, (2, 3))
            self.assertEqual(stored_embeddings.dtype, np.float32)

            usable, reason = _is_usable_rag_index(
                index_dir / "chunks.jsonl",
                retrieval_mode="embedding",
                embedding_model=EMBEDDING_MODEL,
            )
            self.assertEqual((usable, reason), (True, None))

            usable, reason = _is_usable_rag_index(
                index_dir / "chunks.jsonl",
                retrieval_mode="embedding",
                embedding_model="different-model",
            )
            self.assertFalse(usable)
            self.assertIn("model", str(reason))

            rag = MED_KNOWLEDGE_RAG(
                persist_dir=index_dir,
                mode="embedding",
                embedding_client=_offline_embed,
            )
            results = rag.search(
                "How should I make an image channel first?",
                k=1,
                convert_to_str=False,
            )
            self.assertEqual(len(results), 1)
            self.assertEqual(
                results[0].metadata["title"],
                "monai.transforms.EnsureChannelFirstd",
            )
            self.assertIn("cosine similarity", results[0].retrieval_note)

    def test_keyword_index_and_exact_api_lookup_do_not_embed(self) -> None:
        def fail_if_called(*_args, **_kwargs):
            raise AssertionError("keyword mode must not call the embedding endpoint")

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(create_monai_rag_db, "parse_monai_docs", _documents),
        ):
            index_dir = Path(tmp)
            create_monai_rag_db.parse_and_index_docs(
                persist_dir=index_dir,
                embedding_client=fail_if_called,
            )

            manifest = json.loads(
                (index_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["mode"], "keyword")
            self.assertEqual(manifest["record_count"], 2)
            self.assertFalse((index_dir / "embeddings.npy").exists())

            rag = MED_KNOWLEDGE_RAG(
                persist_dir=index_dir,
                embedding_client=fail_if_called,
            )
            results = rag.search(
                "monai.losses.DiceLoss",
                k=1,
                convert_to_str=False,
            )
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].metadata["title"], "monai.losses.DiceLoss")
            self.assertIn("monai.losses.DiceLoss", results[0].retrieval_note)

    def test_embedding_index_checksum_failure_does_not_fallback(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(create_monai_rag_db, "parse_monai_docs", _documents),
        ):
            index_dir = Path(tmp)
            create_monai_rag_db.parse_and_index_docs(
                persist_dir=index_dir,
                mode="embedding",
                embedding_client=_offline_embed,
            )
            chunks_path = index_dir / "chunks.jsonl"
            records = [
                json.loads(line)
                for line in chunks_path.read_text(encoding="utf-8").splitlines()
            ]
            records[0]["page_content"] += " tampered"
            chunks_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "checksum"):
                MED_KNOWLEDGE_RAG(
                    persist_dir=index_dir,
                    mode="embedding",
                    embedding_client=_offline_embed,
                )

    def test_default_top_five_survive_prompt_serialization(self) -> None:
        documents = [
            {
                "page_content": f"API{index} " + ("documentation detail " * 80),
                "metadata": {
                    "source": "/documentation/" + (f"source-{index}-" * 20),
                    "version": "1.4.0",
                    "title": f"monai.module.API{index}" + ("Title" * 25),
                },
            }
            for index in range(5)
        ]
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(create_monai_rag_db, "parse_monai_docs", lambda: documents),
        ):
            index_dir = Path(tmp)
            create_monai_rag_db.parse_and_index_docs(
                persist_dir=index_dir,
                mode="embedding",
                embedding_client=_offline_embed,
            )
            rag = MED_KNOWLEDGE_RAG(
                persist_dir=index_dir,
                mode="embedding",
                embedding_client=_offline_embed,
            )

            result = rag.search("documentation", k=5)

            for index in range(5):
                self.assertIn(f"monai.module.API{index}", result)

    def test_default_config_uses_keyword_retrieval_and_top_five(self) -> None:
        with patch.dict(os.environ):
            os.environ.pop("BIOMAKER_RAG_MODE", None)
            os.environ.pop("BIOMAKER_EMBEDDING_MODEL", None)
            os.environ.pop("BIOMAKER_RAG_TOPK", None)
            cfg = _load_cfg(use_cli_args=False)

        self.assertEqual(cfg.agent.knowledge_base.retrieval_mode, "keyword")
        self.assertEqual(
            cfg.agent.knowledge_base.embedding_model,
            EMBEDDING_MODEL,
        )
        self.assertEqual(cfg.agent.knowledge_base.topk, 5)

    def test_cli_default_remains_keyword_with_an_embedding_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            index_dir = Path(tmp)
            (index_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "mode": "embedding",
                        "embedding_model": "manifest-embedding-model",
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(os.environ):
                os.environ.pop("BIOMAKER_RAG_MODE", None)
                os.environ.pop("BIOMAKER_EMBEDDING_MODEL", None)
                mode, model = query_rag_db._manifest_cli_defaults(index_dir)

        self.assertEqual(mode, "keyword")
        self.assertEqual(model, "manifest-embedding-model")

    def test_proposal_agent_passes_configured_embedding_settings(self) -> None:
        calls: dict[str, object] = {}

        class FakeRag:
            def __init__(self, **kwargs):
                calls["init"] = kwargs

            def search(self, query, **kwargs):
                calls["search"] = (query, kwargs)
                return "retrieved documentation"

        cfg = SimpleNamespace(
            workspace_dir=Path("."),
            log_dir=Path("."),
            exec=SimpleNamespace(include_medical_packages=True),
            agent=SimpleNamespace(
                data_summary=False,
                enable_node_tools=False,
                max_node_tool_calls=4,
                enable_monai_knowledge_base=True,
                enable_query2doc=False,
                enable_web_search=False,
                web_search_topk=2,
                code=SimpleNamespace(model="test-model", temp=0.0),
                knowledge_base=SimpleNamespace(
                    model="helper-model",
                    temp=0.0,
                    retrieval_mode="embedding",
                    embedding_model=EMBEDDING_MODEL,
                    topk=5,
                ),
            ),
        )
        agent = ProposalAgent("task", cfg, Journal())

        with patch.object(query_rag_db, "MED_KNOWLEDGE_RAG", FakeRag):
            result = agent.query_knowledge_base("monai.transforms.EnsureChannelFirstd")

        self.assertEqual(result, "retrieved documentation")
        self.assertEqual(
            calls["init"],
            {
                "model": "helper-model",
                "mode": "embedding",
                "embedding_model": EMBEDDING_MODEL,
            },
        )
        self.assertEqual(
            calls["search"],
            (
                "monai.transforms.EnsureChannelFirstd",
                {"k": 5, "enable_query2doc": False},
            ),
        )


if __name__ == "__main__":
    unittest.main()
