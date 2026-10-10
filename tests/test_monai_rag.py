import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import biomaker.agents.proposal_agent as proposal_agent_module
import biomaker.monai_rag.query_rag_db as query_rag_db
from biomaker.agents.proposal_agent import ProposalAgent
from biomaker.journal import Journal
from biomaker.monai_rag import create_monai_rag_db
from biomaker.monai_rag.evidence import (
    MONAI_RAG_SMOKE_QUERY,
    MonaiRagEvidenceError,
    query_sha256,
    run_monai_rag_smoke,
)
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

            rendered, returned_count = rag.search_with_count(
                "monai.losses.DiceLoss",
                k=1,
            )
            self.assertEqual(returned_count, 1)
            self.assertIn("monai.losses.DiceLoss", rendered)

    def test_keyword_search_accepts_natural_language_without_api_names(self) -> None:
        documents = [query_rag_db.RagDocument(**record) for record in _documents()]
        with (
            patch.object(query_rag_db, "_load_documents", return_value=documents),
            patch.object(query_rag_db, "_embed_texts", side_effect=AssertionError),
        ):
            rag = MED_KNOWLEDGE_RAG(mode="keyword")
            for query, expected_title in [
                (
                    "channel first layout for image tensors",
                    "monai.transforms.EnsureChannelFirstd",
                ),
                (
                    "soft dice objective for segmentation",
                    "monai.losses.DiceLoss",
                ),
            ]:
                with self.subTest(query=query):
                    results = rag.search(query, k=1, convert_to_str=False)
                    self.assertEqual(len(results), 1)
                    self.assertEqual(results[0].metadata["title"], expected_title)

    def test_keyword_search_returns_no_hits_for_missing_full_api(self) -> None:
        documents = [query_rag_db.RagDocument(**record) for record in _documents()]
        with patch.object(query_rag_db, "_load_documents", return_value=documents):
            rag = MED_KNOWLEDGE_RAG(mode="keyword")
            for query in [
                "monai.metrics.DiceMetric",
                "monai.inferers.sliding_window_inference",
                "monai.losses.Dice",
            ]:
                with self.subTest(query=query):
                    result, count = rag.search_with_count(query, k=5)
                    self.assertEqual(result, "")
                    self.assertEqual(count, 0)

    def test_keyword_search_keeps_array_and_dictionary_transforms(self) -> None:
        documents = [
            query_rag_db.RagDocument(
                page_content=f"{name} applies a random affine transformation.",
                metadata={"title": f"monai.transforms.{name}"},
            )
            for name in ["RandAffine", "RandAffined", "RandAffineD", "RandAffineDict"]
        ]
        with patch.object(query_rag_db, "_load_documents", return_value=documents):
            rag = MED_KNOWLEDGE_RAG(mode="keyword")
            results = rag.search(
                "random affine transformation", k=5, convert_to_str=False
            )
            self.assertEqual(len(results), 2)
            self.assertEqual(
                {doc.metadata["title"] for doc in results},
                {"monai.transforms.RandAffine", "monai.transforms.RandAffined"},
            )

    def test_keyword_smoke_receipt_is_successful_and_byte_stable(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(create_monai_rag_db, "parse_monai_docs", _documents),
        ):
            root = Path(tmp)
            index_dir = root / "index"
            receipt_path = root / "logs" / "monai_rag_smoke.json"
            create_monai_rag_db.parse_and_index_docs(
                persist_dir=index_dir,
                mode="keyword",
            )

            first = run_monai_rag_smoke(
                receipt_path,
                persist_dir=index_dir,
                required=True,
            )
            first_bytes = receipt_path.read_bytes()
            second = run_monai_rag_smoke(
                receipt_path,
                persist_dir=index_dir,
                required=True,
            )

            self.assertEqual(first, second)
            self.assertEqual(first_bytes, receipt_path.read_bytes())
            self.assertTrue(first["ok"])
            self.assertTrue(first["index_available"])
            self.assertTrue(first["index_usable"])
            self.assertTrue(first["query_attempted"])
            self.assertTrue(first["query_succeeded"])
            self.assertEqual(first["returned_count"], 1)
            self.assertEqual(first["query_sha256"], query_sha256(MONAI_RAG_SMOKE_QUERY))
            self.assertNotIn(MONAI_RAG_SMOKE_QUERY, receipt_path.read_text())

    def test_missing_index_smoke_writes_receipt_and_only_required_mode_raises(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index_dir = root / "missing-index"
            optional_path = root / "optional.json"
            optional = run_monai_rag_smoke(
                optional_path,
                persist_dir=index_dir,
                required=False,
            )
            self.assertFalse(optional["ok"])
            self.assertFalse(optional["index_available"])
            self.assertFalse(optional["query_attempted"])
            self.assertTrue(optional_path.is_file())

            required_path = root / "required.json"
            with self.assertRaises(MonaiRagEvidenceError):
                run_monai_rag_smoke(
                    required_path,
                    persist_dir=index_dir,
                    required=True,
                )
            required = json.loads(required_path.read_text(encoding="utf-8"))
            self.assertTrue(required["required"])
            self.assertFalse(required["ok"])
            self.assertIn("missing", required["error"])

    def test_smoke_distinguishes_query_exception_from_successful_zero_hits(
        self,
    ) -> None:
        class QueryFailureRag:
            def __init__(self, **_kwargs):
                pass

            def search_with_count(self, *_args, **_kwargs):
                raise RuntimeError("synthetic confidential retrieved text")

        class ZeroHitRag:
            def __init__(self, **_kwargs):
                pass

            def search_with_count(self, *_args, **_kwargs):
                return "", 0

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index_dir = root / "index"
            index_dir.mkdir()
            (index_dir / "chunks.jsonl").write_text(
                json.dumps(_documents()[0]) + "\n",
                encoding="utf-8",
            )

            failed = run_monai_rag_smoke(
                root / "failed.json",
                persist_dir=index_dir,
                rag_factory=QueryFailureRag,
            )
            zero = run_monai_rag_smoke(
                root / "zero.json",
                persist_dir=index_dir,
                rag_factory=ZeroHitRag,
            )

            self.assertTrue(failed["index_usable"])
            self.assertTrue(failed["query_attempted"])
            self.assertFalse(failed["query_succeeded"])
            self.assertIn("RuntimeError", failed["error"])
            self.assertNotIn("confidential", (root / "failed.json").read_text())
            self.assertTrue(zero["query_succeeded"])
            self.assertEqual(zero["returned_count"], 0)
            self.assertIn("returned no documents", zero["error"])

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

    def test_large_result_budget_keeps_passages_with_their_sources(self) -> None:
        documents = [
            {
                "page_content": f"PASSAGE_{index:02d} " * 200,
                "metadata": {"source": f"guide-{index}.html", "title": f"API{index}"},
            }
            for index in range(20)
        ]
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(create_monai_rag_db, "parse_monai_docs", lambda: documents),
        ):
            index_dir = Path(tmp)
            create_monai_rag_db.parse_and_index_docs(persist_dir=index_dir)
            rag = MED_KNOWLEDGE_RAG(persist_dir=index_dir)
            with patch.object(rag, "_search_once", return_value=rag.documents):
                result, count = rag.search_with_count("documentation", k=20)

        self.assertEqual(count, 20)
        self.assertLessEqual(len(result), 16000)
        self.assertIn("Showing", result)
        for index, passage in enumerate(result.split("\n\n---\n\n")):
            self.assertIn(f"Source: guide-{index}.html", passage)
            self.assertIn(f"PASSAGE_{index:02d}", passage)
            for other in range(20):
                if other != index:
                    self.assertNotIn(f"PASSAGE_{other:02d}", passage)

    def test_default_config_uses_keyword_retrieval_and_top_five(self) -> None:
        with patch.dict(os.environ):
            os.environ.pop("BIOMAKER_RAG_MODE", None)
            os.environ.pop("BIOMAKER_EMBEDDING_MODEL", None)
            os.environ.pop("BIOMAKER_RAG_TOPK", None)
            os.environ.pop("BIOMAKER_RAG_REQUIRED", None)
            cfg = _load_cfg(use_cli_args=False)

        self.assertEqual(cfg.agent.knowledge_base.retrieval_mode, "keyword")
        self.assertEqual(
            cfg.agent.knowledge_base.embedding_model,
            EMBEDDING_MODEL,
        )
        self.assertEqual(cfg.agent.knowledge_base.topk, 5)
        self.assertFalse(cfg.agent.knowledge_base.required)

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
        events: list[tuple[str, dict[str, object]]] = []

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
                    required=False,
                ),
            ),
        )
        agent = ProposalAgent("task", cfg, Journal())

        def capture_event(_log_dir, event, **payload):
            events.append((event, payload))
            return True

        with (
            patch.object(query_rag_db, "MED_KNOWLEDGE_RAG", FakeRag),
            patch.object(proposal_agent_module, "append_event", capture_event),
        ):
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
        self.assertEqual(events[0][0], "monai_rag_query")
        runtime_evidence = events[0][1]
        self.assertTrue(runtime_evidence["query_succeeded"])
        self.assertEqual(runtime_evidence["returned_count"], 1)
        self.assertNotIn("query", runtime_evidence)
        self.assertNotIn("result", runtime_evidence)

    def test_required_runtime_zero_hit_fails_after_content_free_event(self) -> None:
        class ZeroHitRag:
            def __init__(self, **_kwargs):
                pass

            def search_with_count(self, *_args, **_kwargs):
                return "", 0

        with tempfile.TemporaryDirectory() as tmp:
            cfg = SimpleNamespace(
                workspace_dir=Path(tmp),
                log_dir=Path(tmp),
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
                        retrieval_mode="keyword",
                        embedding_model=EMBEDDING_MODEL,
                        topk=5,
                        required=True,
                    ),
                ),
            )
            agent = ProposalAgent("task", cfg, Journal())
            with (
                patch.object(query_rag_db, "MED_KNOWLEDGE_RAG", ZeroHitRag),
                self.assertRaises(MonaiRagEvidenceError),
            ):
                agent.query_knowledge_base("monai.transforms.EnsureChannelFirstd")

            event = json.loads(
                (Path(tmp) / "events.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )
            self.assertEqual(event["event"], "monai_rag_query")
            self.assertEqual(event["outcome"], "zero_hits")
            self.assertTrue(event["query_succeeded"])
            self.assertEqual(event["returned_count"], 0)
            self.assertNotIn("query", event)
            self.assertNotIn("result", event)
            self.assertNotIn(MONAI_RAG_SMOKE_QUERY, json.dumps(event))

            with (
                patch.object(
                    proposal_agent_module,
                    "append_event",
                    return_value=False,
                ),
                self.assertRaisesRegex(
                    MonaiRagEvidenceError,
                    "could not be written",
                ),
            ):
                agent.query_knowledge_base("monai.transforms.EnsureChannelFirstd")


if __name__ == "__main__":
    unittest.main()
