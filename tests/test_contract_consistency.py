from __future__ import annotations
import csv
import io
import importlib.util
import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from rich.console import Console

import biomaker.backend.backend_openai as backend_openai
import biomaker.agents.proposal_agent as proposal_agent
from biomaker.backend import compile_prompt_to_md
from biomaker.agents.base_agent import BaseAgent, parse_factual_exec_result
from biomaker.agents.proposal_agent import ProposalAgent
from biomaker.interpreter import ExecutionResult, Interpreter
from biomaker.journal import Journal, Node
from biomaker.tools.python_probe import python_probe
from biomaker.run import (
    _freeze_generation_inputs,
    _final_retrain_timeout_seconds,
    _private_metrics_summary,
    _remaining_schedule_budget,
    _run_overview_text,
    journal_to_rich_tree,
)
from biomaker.utils.metric import MetricValue
from biomaker.utils.config import (
    _copy_best_working_outputs,
    _validate_exec_config,
    sync_node_working_to_run_workspace,
)
from biomaker.utils.artifact_validation import (
    ArtifactValidationResult,
    normalize_working_outputs,
    validate_working_outputs,
)
from biomaker.utils import agentic_data_planner
from biomaker.utils.environment import allowed_package_specs, validate_code_environment
from biomaker.utils.evaluator_setup import EvaluatorRunResult
from biomaker.utils.execution_record import (
    attach_artifact_validation_result,
    attach_evaluator_run_result,
)
from biomaker.utils.experiment_contract import load_experiment_contract
from biomaker.utils.exp_guide import build_exp_guide
from biomaker.utils.prediction_contract import load_prediction_contract
from biomaker.utils.private_evaluation import evaluate_private_outputs
from biomaker.utils.response import extract_code_blocks
from biomaker.utils.serialize import dumps_json, loads_json
from biomaker.utils.tree_export import cfg_to_tree_struct


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class ContractConsistencyTests(unittest.TestCase):
    def _render_live_tree(self, nodes: list[Node]) -> str:
        output = io.StringIO()
        console = Console(file=output, force_terminal=False, width=100)
        console.print(journal_to_rich_tree(SimpleNamespace(nodes=nodes)))
        return output.getvalue()

    def test_prompt_compiler_handles_structured_numeric_lists(self) -> None:
        prompt = {
            "Experiment Guide": {
                "array_profile": {
                    "shape": [224, 224, 3],
                    "examples": [[224, 224, 3]],
                    "counts": {"0": 2, "1": 3},
                }
            }
        }

        rendered = compile_prompt_to_md(prompt)

        self.assertIn("- 224", rendered)
        self.assertIn("## array_profile", rendered)
        self.assertIn("### shape", rendered)

    def test_openai_backend_rejects_failed_completion_sentinel(self) -> None:
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kwargs: None)
            )
        )

        with (
            patch.object(backend_openai, "_setup_openai_client", return_value=None),
            patch.object(backend_openai, "_client", fake_client),
            patch.object(backend_openai, "backoff_create", return_value=False),
        ):
            with self.assertRaisesRegex(RuntimeError, "no completion choices"):
                backend_openai.query("system", "user", model="test-model")

    def test_node_tool_requests_normalize_nullable_optional_fields(self) -> None:
        cfg = SimpleNamespace(
            workspace_dir=Path("."),
            log_dir=Path("."),
            exec=SimpleNamespace(include_medical_packages=True),
            agent=SimpleNamespace(
                data_summary=False,
                enable_node_tools=True,
                max_node_tool_calls=4,
                code=SimpleNamespace(model="test-model"),
            ),
        )
        agent = ProposalAgent("task", cfg, Journal())  # type: ignore[arg-type]
        agent.data_summary = "summary"

        with patch.object(
            proposal_agent,
            "query",
            return_value={
                "requests": [
                    {
                        "tool": "inspect_api",
                        "reason": None,
                        "query": None,
                        "code": None,
                    },
                    {"tool": "unknown", "reason": "ignored"},
                    "not a request",
                ]
            },
        ):
            requests = agent._request_node_tools("draft", {})

        self.assertEqual(
            requests,
            [{"tool": "inspect_api", "reason": "", "query": "", "code": ""}],
        )

    def test_live_tree_best_uses_only_committed_nodes(self) -> None:
        committed = Node(code="print('committed')")
        committed.step = 0
        committed.is_buggy = False
        committed.metric = MetricValue(0.8, maximize=True)
        transient = Node(code="print('transient')")
        transient.scheduled_step = 1
        transient.mark_status("completed")
        transient.is_buggy = False
        transient.metric = MetricValue(0.99, maximize=True)

        rendered = self._render_live_tree([committed, transient])

        self.assertIn("0.800 (best)", rendered)
        self.assertIn("1: 0.990", rendered)
        self.assertNotIn("0.990 (best)", rendered)

    def test_pending_generation_counts_against_schedule_budget(self) -> None:
        official_pending = Node(code="print('official')")
        generating = Node(code="", plan="generating")

        budget = _remaining_schedule_budget(
            total_steps=5,
            committed_steps=1,
            round_nodes_by_round={
                1: [SimpleNamespace(node=official_pending)],
            },
            generating_nodes=[generating],
        )

        self.assertEqual(budget, 2)

    def test_tree_root_exposes_pre_node_setup_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            log_dir = root / "logs"
            split_dir = workspace / "input" / "biomaker_splits"
            split_dir.mkdir(parents=True)
            log_dir.mkdir(parents=True)
            _write_json(
                workspace / "exp_guide.json",
                {"prediction": {"key_columns": ["Id"]}},
            )
            _write_json(
                workspace / "evaluator_prediction_spec.json",
                {"test_submission": "./working/submission.csv"},
            )
            _write_json(split_dir / "split_plan.json", {"task_type": "classification"})
            _write_json(log_dir / "run_manifest.json", {"exp_name": "tree-root"})
            cfg = SimpleNamespace(
                exp_name="tree-root",
                workspace_dir=workspace,
                log_dir=log_dir,
                data_dir=root / "data",
                desc_file=root / "task.md",
            )

            tree = cfg_to_tree_struct(cfg, Journal())
            root_prompt = tree["prompts"][0]

            self.assertTrue(root_prompt["available"])
            section_names = {section["name"] for section in root_prompt["sections"]}
            self.assertIn("Run Setup", section_names)
            self.assertIn("Experiment Guide", section_names)
            self.assertIn("Evaluator Prediction Spec", section_names)
            self.assertIn("Split Plan", section_names)
            self.assertIn("Pre-node setup", tree["summaries"][0]["plan"])

    def test_generation_freeze_excludes_non_journal_children(self) -> None:
        parent = Node(code="print('official')")
        pending = Node(code="print('pending')", parent=parent)
        invisible = Node(code="print('not in journal')", parent=parent)

        frozen_parent, frozen_pending, frozen_journal = _freeze_generation_inputs(
            task_parent=parent,
            pending_context=[pending],
            journal_snapshot=[parent],
        )

        self.assertIs(frozen_parent, frozen_journal[0])
        self.assertEqual(frozen_parent.id, parent.id)
        self.assertEqual([node.id for node in frozen_pending], [pending.id])
        self.assertIs(frozen_pending[0].parent, frozen_parent)
        self.assertNotIn(frozen_pending[0], frozen_parent.children)
        self.assertNotIn(invisible.id, {child.id for child in frozen_parent.children})

    def test_task_selection_ignores_children_outside_journal(self) -> None:
        cfg = SimpleNamespace(
            agent=SimpleNamespace(
                search=SimpleNamespace(
                    max_debug_depth=2,
                    max_children_per_node=1,
                    topk_improve=1,
                )
            ),
            exec=SimpleNamespace(
                include_medical_packages=True,
            ),
        )
        parent = Node(code="print('official')")
        parent.step = 0
        parent.is_buggy = False
        parent.metric = MetricValue(0.8, maximize=True)
        _child_not_in_journal = Node(code="print('not in journal')", parent=parent)

        tasks = BaseAgent("task", cfg, Journal(nodes=[parent])).select_round_tasks()  # type: ignore[arg-type]

        self.assertEqual(len(tasks), 1)
        self.assertIs(tasks[0].parent, parent)

    def test_round_best_node_before_is_fixed_snapshot(self) -> None:
        snapshot_best = Node(code="print('old best')")
        snapshot_best.is_buggy = False
        snapshot_best.metric = MetricValue(0.8, maximize=True)
        later_best = Node(code="print('later best')")
        later_best.is_buggy = False
        later_best.metric = MetricValue(0.95, maximize=True)
        candidate = Node(code="print('candidate')")
        exec_result = ExecutionResult(
            term_out=[
                'BioMakerExternalValidation: {"metric": 0.85, "maximize": true, "details": {}}\n'
            ],
            exec_time=0.1,
            exc_type=None,
        )

        parse_factual_exec_result(
            candidate,
            exec_result,
            best_node_before=snapshot_best,
            record_context=True,
        )

        self.assertEqual(candidate.best_metric_before, snapshot_best.metric.value)
        self.assertNotEqual(candidate.best_metric_before, later_best.metric.value)
        self.assertAlmostEqual(candidate.delta_vs_best, 0.05)
        self.assertEqual(candidate.outcome, "new_best")

    def test_parse_exec_result_prefers_structured_evaluator_result(self) -> None:
        candidate = Node(code="print('candidate')")
        exec_result = ExecutionResult(
            term_out=[
                "candidate stdout\n",
                'BioMakerExternalValidation: {"metric": 0.1, "maximize": true, "details": {}}\n',
            ],
            exec_time=0.2,
            exc_type=None,
            stdout="candidate stdout\n",
            candidate_output="candidate stdout\n",
            artifact_validation={"ok": True, "messages": []},
            evaluator_result={
                "ok": True,
                "metric": 0.91,
                "maximize": True,
                "details": {
                    "score": {"metric": 0.91, "accuracy": 0.91},
                    "artifact_report": {
                        "available": True,
                        "alignment": {"method": "key_columns", "matched_rows": 3},
                        "warnings": [],
                    },
                },
                "error": None,
            },
        )

        parse_factual_exec_result(candidate, exec_result, record_context=True)

        self.assertFalse(candidate.is_buggy)
        self.assertAlmostEqual(candidate.metric.value, 0.91)
        self.assertEqual(candidate.artifact_report["alignment"]["matched_rows"], 3)
        self.assertIsInstance(candidate.execution, dict)
        self.assertEqual(candidate.execution["evaluator"]["metric"], 0.91)
        self.assertIn("candidate stdout", "".join(candidate._term_out))

    def test_debug_tree_records_use_prompt_safe_execution_summary(self) -> None:
        parent = Node(code="raise RuntimeError('bad')", plan="buggy")
        exec_result = ExecutionResult(
            term_out=[
                "Traceback (most recent call last):\n"
                '  File "/home/hzy/projects/BioMaker/work/runfile.py", line 1, in <module>\n'
                "RuntimeError: bad\n",
                'BioMakerExternalValidation: {"metric": 0.9, "maximize": true, "details": {}}\n',
            ],
            exec_time=0.1,
            exc_type="RuntimeError",
            exc_info={"reason": "/home/hzy/projects/BioMaker/private/path"},
            stderr="Traceback (most recent call last):\nRuntimeError: bad\n",
            candidate_output="Traceback (most recent call last):\nRuntimeError: bad\n",
            candidate_exc_type="RuntimeError",
            candidate_exc_info={"returncode": "1"},
            failure_phase="candidate_run",
        )
        parse_factual_exec_result(parent, exec_result, record_context=True)
        journal = Journal()
        journal.append(parent)

        records = journal.generate_tree_records(
            stage="debug",
            selected_parent=parent,
        )

        self.assertIn("Execution Summary:", records)
        self.assertIn("candidate_failure:", records)
        self.assertIn("RuntimeError: bad", records)
        self.assertNotIn("BioMakerExternalValidation", records)
        self.assertNotIn("/home/hzy", records)

    def test_tree_export_keeps_raw_node_output_for_human_debug(self) -> None:
        node = Node(code="print('raw')")
        exec_result = ExecutionResult(
            term_out=["RAW_NODE_OUTPUT_SENTINEL\n"],
            exec_time=0.1,
            exc_type=None,
            stdout="RAW_NODE_OUTPUT_SENTINEL\n",
            candidate_output="RAW_NODE_OUTPUT_SENTINEL\n",
            evaluator_result={
                "ok": True,
                "metric": 0.5,
                "maximize": True,
                "details": {"score": {"metric": 0.5}},
                "error": None,
            },
        )
        parse_factual_exec_result(node, exec_result, record_context=True)
        journal = Journal()
        journal.append(node)
        cfg = SimpleNamespace(
            exp_name="test", workspace_dir=Path("."), log_dir=Path(".")
        )

        tree = cfg_to_tree_struct(cfg, journal)

        self.assertIn("RAW_NODE_OUTPUT_SENTINEL", tree["term_out"][1])
        restored = loads_json(dumps_json(journal), Journal)
        self.assertIn("RAW_NODE_OUTPUT_SENTINEL", "".join(restored.nodes[0]._term_out))

    def test_legacy_journal_without_execution_or_parent_map_loads(self) -> None:
        node = Node(code="print('legacy')")
        node._term_out = ["legacy stdout\n"]
        journal = Journal()
        journal.append(node)
        payload = json.loads(dumps_json(journal))
        payload.pop("node2parent", None)
        payload["nodes"][0].pop("execution", None)

        restored = loads_json(json.dumps(payload), Journal)

        self.assertIsNone(restored.nodes[0].execution)
        self.assertIn("legacy stdout", restored.nodes[0].term_out)

    def test_node_term_out_tolerates_missing_execution_output(self) -> None:
        node = Node(code="print('pending')")
        node._term_out = None

        self.assertEqual(node.term_out, "")

    def test_candidate_runtime_survives_node_runtime_override(self) -> None:
        result = ExecutionResult(
            term_out=["candidate\n"],
            exec_time=1.0,
            exc_type=None,
            stdout="candidate\n",
            candidate_output="candidate\n",
            candidate_exec_time=1.0,
        )
        result.exec_time = 3.0
        node = Node(code="print('candidate')")
        node.absorb_exec_result(result)

        self.assertEqual(node.execution["candidate"]["runtime_s"], 1.0)
        self.assertEqual(node.execution["node_runtime_s"], 3.0)

    def test_execution_record_helpers_attach_failures_consistently(self) -> None:
        result = ExecutionResult(
            term_out=["candidate ok\n"],
            exec_time=1.0,
            exc_type=None,
            stdout="candidate ok\n",
            candidate_output="candidate ok\n",
        )
        attach_artifact_validation_result(
            result,
            ArtifactValidationResult(ok=False, messages=["submission.csv missing"]),
            failure_reason="artifacts failed",
        )

        self.assertEqual(result.exc_type, "ArtifactValidationError")
        self.assertEqual(result.failure_phase, "artifact_validation")
        self.assertIn("submission.csv missing", result.system_outputs[-1]["message"])

        result = ExecutionResult(
            term_out=["candidate ok\n"],
            exec_time=1.0,
            exc_type=None,
            stdout="candidate ok\n",
            candidate_output="candidate ok\n",
        )
        attach_evaluator_run_result(
            result,
            EvaluatorRunResult(
                ok=False,
                metric=None,
                maximize=True,
                details={},
                output="evaluator raw output",
                error="bad submission",
            ),
            failure_phase="evaluator_scoring",
            failure_type="EvaluatorValidationError",
            failure_message_prefix="EvaluatorValidationError",
        )

        self.assertEqual(result.exc_type, "EvaluatorValidationError")
        self.assertEqual(result.failure_phase, "evaluator_scoring")
        self.assertFalse(result.evaluator_result["ok"])
        self.assertIn("bad submission", result.system_outputs[-1]["message"])

    def test_debug_tree_records_do_not_include_children_of_selected_bug(self) -> None:
        buggy_parent = Node(code="raise RuntimeError('bug')", plan="buggy parent")
        buggy_parent.step = 0
        buggy_parent.is_buggy = True
        buggy_parent.metric = MetricValue(None, maximize=True)
        buggy_parent.exc_type = "RuntimeError"
        buggy_parent._term_out = ["RuntimeError: bug\n"]
        debug_child = Node(
            code="print('debug attempt')",
            plan="previous debug repair",
            parent=buggy_parent,
        )
        debug_child.step = 1
        debug_child.is_buggy = True
        debug_child.metric = MetricValue(None, maximize=True)
        debug_child.exc_type = "ValueError"
        debug_child._term_out = ["ValueError: still broken\n"]
        journal = Journal(nodes=[buggy_parent, debug_child])

        records = journal.generate_tree_records(
            stage="debug",
            selected_parent=buggy_parent,
        )

        self.assertNotIn("previous debug repair", records)
        self.assertNotIn("ValueError: still broken", records)

    def test_task_selection_debugs_each_buggy_node_once(self) -> None:
        cfg = SimpleNamespace(
            agent=SimpleNamespace(
                search=SimpleNamespace(
                    max_debug_depth=2,
                    max_children_per_node=1,
                    topk_improve=1,
                )
            ),
            exec=SimpleNamespace(
                include_medical_packages=True,
            ),
        )
        buggy_parent = Node(code="raise RuntimeError('bug')")
        buggy_parent.step = 0
        buggy_parent.is_buggy = True
        existing_debug = Node(code="print('debug')", parent=buggy_parent)
        existing_debug.step = 1
        existing_debug.is_buggy = True

        tasks = BaseAgent(
            "task",
            cfg,
            Journal(nodes=[buggy_parent, existing_debug]),
        ).select_round_tasks()  # type: ignore[arg-type]

        self.assertFalse(any(task.parent is buggy_parent for task in tasks))

    def test_debug_prompt_pending_context_uses_buggy_parent_siblings(self) -> None:
        cfg = SimpleNamespace(
            workspace_dir=Path("."),
            log_dir=Path("."),
            exec=SimpleNamespace(include_medical_packages=True),
            agent=SimpleNamespace(
                data_summary=False,
                expose_prediction=False,
                enable_node_tools=False,
                max_node_tool_calls=4,
                enable_monai_knowledge_base=False,
                enable_web_search=False,
                web_search_topk=2,
                code=SimpleNamespace(model="test-model", temp=0.0),
                knowledge_base=SimpleNamespace(model="test-model", temp=0.0),
            ),
        )
        root = Node(code="print('root')", plan="root")
        root.step = 0
        root.is_buggy = False
        root.metric = MetricValue(0.7, maximize=True)
        buggy_parent = Node(
            code="raise RuntimeError('bug')", plan="buggy parent", parent=root
        )
        buggy_parent.step = 1
        buggy_parent.is_buggy = True
        sibling_pending = Node(
            code="print('pending parent sibling')",
            plan="pending parent sibling",
            parent=root,
        )
        debug_pending = Node(
            code="print('pending debug child')",
            plan="pending debug child",
            parent=buggy_parent,
        )
        agent = ProposalAgent("task", cfg, Journal(nodes=[root, buggy_parent]))  # type: ignore[arg-type]
        agent.set_generation_context(
            pending_round_nodes=[sibling_pending, debug_pending]
        )

        prompt = agent.context_builder.build_debug(buggy_parent)
        records = prompt["Tree Records"]

        self.assertIn("pending parent sibling", records)
        self.assertNotIn("pending debug child", records)

    def test_journal_commit_step_is_monotonic_not_planned_slot(self) -> None:
        first = Node(code="print('first')")
        second = Node(code="print('second')")
        first.scheduled_step = 5
        second.scheduled_step = 5
        journal = Journal()

        journal.append(first)
        journal.append(second)

        self.assertEqual(first.step, 0)
        self.assertEqual(second.step, 1)
        self.assertEqual(first.scheduled_step, 5)
        self.assertEqual(second.scheduled_step, 5)

    def test_exec_worker_config_rejects_invalid_values(self) -> None:
        valid_exec = SimpleNamespace(
            timeout=1200,
            parallel_workers=2,
        )
        _validate_exec_config(SimpleNamespace(exec=valid_exec))

        invalid_exec = SimpleNamespace(
            timeout=1200,
            parallel_workers=0,
        )
        with self.assertRaises(ValueError):
            _validate_exec_config(SimpleNamespace(exec=invalid_exec))

    def test_final_retrain_timeout_is_one_and_half_node_timeout(self) -> None:
        cfg = SimpleNamespace(exec=SimpleNamespace(timeout=601))

        self.assertEqual(_final_retrain_timeout_seconds(cfg), 902)

    def test_split_builder_retries_response_without_code_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            input_dir.mkdir()
            task_file = root / "task.md"
            task_file.write_text("Build a tiny split.", encoding="utf-8")
            output_dir = root / "package"
            query_failures = []

            def fake_query_builder(*args, previous_failure=None, **kwargs):
                query_failures.append(previous_failure)
                if len(query_failures) == 1:
                    return (
                        "",
                        "This explains the split but forgot the code fence.",
                        {"prompt_tokens": 10, "completion_tokens": 5},
                    )
                return (
                    "print('ok')",
                    "```python\nprint('ok')\n```",
                    {"prompt_tokens": 11, "completion_tokens": 6},
                )

            def fake_run(*args, **kwargs):
                return SimpleNamespace(returncode=0, stdout="ok")

            def fake_verify(output_dir, **kwargs):
                (output_dir / "biomaker_splits").mkdir(parents=True, exist_ok=True)
                return {"ok": True, "row_counts": {}}

            with (
                patch.object(
                    agentic_data_planner,
                    "_query_builder",
                    side_effect=fake_query_builder,
                ),
                patch.object(
                    agentic_data_planner.subprocess, "run", side_effect=fake_run
                ),
                patch.object(
                    agentic_data_planner,
                    "verify_split_package",
                    side_effect=fake_verify,
                ),
            ):
                result = agentic_data_planner.run_agentic_split_builder_probe(
                    input_dir,
                    task_file,
                    output_dir,
                    max_attempts=2,
                    debug_artifacts=True,
                )

            self.assertTrue(result.verifier["ok"])
            self.assertEqual(len(query_failures), 2)
            self.assertIsNone(query_failures[0])
            self.assertIn("valid Python code block", query_failures[1]["errors"][0])
            self.assertTrue((output_dir / "_debug" / "response_1.txt").is_file())

    def test_working_latest_and_best_artifacts_are_separate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            log_dir = root / "logs"
            best_workspace = root / "best_node"
            latest_workspace = root / "latest_node"
            cfg = SimpleNamespace(workspace_dir=workspace, log_dir=log_dir)
            log_dir.mkdir(parents=True)

            best_working = best_workspace / "working"
            latest_working = latest_workspace / "working"
            (best_working / "pred_masks").mkdir(parents=True)
            latest_working.mkdir(parents=True)
            (best_working / "pred_masks" / "best.txt").write_text(
                "best mask", encoding="utf-8"
            )
            _write_csv(
                best_working / "submission.csv",
                [{"Id": "a", "mask_path": "working/pred_masks/best.txt"}],
            )
            _write_csv(
                latest_working / "submission.csv",
                [{"Id": "a", "prediction": "latest"}],
            )

            sync_node_working_to_run_workspace(
                cfg,
                best_workspace,
                target_name="working_best",
            )
            sync_node_working_to_run_workspace(
                cfg,
                latest_workspace,
                target_name="working_latest",
            )
            sync_node_working_to_run_workspace(cfg, latest_workspace)
            _copy_best_working_outputs(cfg)

            self.assertIn(
                "latest",
                (workspace / "working_latest" / "submission.csv").read_text(
                    encoding="utf-8"
                ),
            )
            self.assertIn(
                "best",
                (workspace / "working_best" / "submission.csv").read_text(
                    encoding="utf-8"
                ),
            )
            self.assertIn(
                "latest",
                (workspace / "working" / "submission.csv").read_text(encoding="utf-8"),
            )
            self.assertIn(
                "best",
                (log_dir / "submission.csv").read_text(encoding="utf-8"),
            )
            self.assertEqual(
                (log_dir / "working_best" / "pred_masks" / "best.txt").read_text(
                    encoding="utf-8"
                ),
                "best mask",
            )
            self.assertEqual(
                (log_dir / "working" / "pred_masks" / "best.txt").read_text(
                    encoding="utf-8"
                ),
                "best mask",
            )

    def test_bootstrap_finishes_root_drafts_before_debug(self) -> None:
        cfg = SimpleNamespace(
            agent=SimpleNamespace(
                search=SimpleNamespace(
                    max_debug_depth=2,
                    max_children_per_node=5,
                    topk_improve=3,
                )
            ),
            exec=SimpleNamespace(
                include_medical_packages=True,
            ),
        )
        journal = Journal()
        agent = BaseAgent("task", cfg, journal)  # type: ignore[arg-type]

        initial_tasks = agent.select_round_tasks()

        self.assertEqual(len(initial_tasks), 5)
        self.assertTrue(all(task.parent is None for task in initial_tasks))

        failed_draft = Node(code="raise RuntimeError('x')")
        failed_draft.step = 0
        failed_draft.is_buggy = True
        journal = Journal(nodes=[failed_draft])
        agent = BaseAgent("task", cfg, journal)  # type: ignore[arg-type]

        followup_tasks = agent.select_round_tasks()

        self.assertEqual(len(followup_tasks), 4)
        self.assertTrue(all(task.parent is None for task in followup_tasks))
        self.assertTrue(
            all(
                "bootstrap selected virtual root draft" in task.selection_reason
                for task in followup_tasks
            )
        )

        filled_bootstrap = Journal(
            nodes=[
                failed_draft,
                Node(code="draft 2"),
                Node(code="draft 3"),
                Node(code="draft 4"),
                Node(code="draft 5"),
            ]
        )
        for idx, node in enumerate(filled_bootstrap.nodes):
            node.step = idx
            node.is_buggy = node is failed_draft
            if node is not failed_draft:
                node.metric = MetricValue(0.5 + idx / 100, maximize=True)
        agent = BaseAgent("task", cfg, filled_bootstrap)  # type: ignore[arg-type]

        post_bootstrap_tasks = agent.select_round_tasks()

        self.assertTrue(
            any(task.parent is failed_draft for task in post_bootstrap_tasks)
        )

    def test_bootstrap_drafts_receive_diverse_plan_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = SimpleNamespace(
                workspace_dir=Path(tmp),
                log_dir=Path(tmp) / "logs",
                exec=SimpleNamespace(include_medical_packages=True),
                agent=SimpleNamespace(
                    search=SimpleNamespace(
                        max_debug_depth=2,
                        max_children_per_node=3,
                        topk_improve=1,
                    ),
                    data_summary=True,
                    expose_prediction=False,
                    enable_node_tools=False,
                    max_node_tool_calls=4,
                    enable_monai_knowledge_base=False,
                    enable_web_search=False,
                    web_search_topk=2,
                    code=SimpleNamespace(model="test-model", temp=0.7),
                    knowledge_base=SimpleNamespace(model="test-model", temp=0.0),
                ),
            )
            agent = ProposalAgent("task", cfg, Journal())  # type: ignore[arg-type]
            agent.data_summary = "data summary"

            with patch(
                "biomaker.agents.proposal_agent.query",
                return_value={
                    "plans": [
                        {"name": "image evidence", "plan": "Use the image inputs."},
                        {"name": "split aware", "plan": "Use the split facts."},
                        {"name": "metric aligned", "plan": "Optimize for the metric."},
                    ]
                },
            ):
                tasks = agent.select_round_tasks()

            self.assertEqual(len(tasks), 3)
            self.assertEqual(
                [task.draft_plan for task in tasks],
                [
                    "image evidence: Use the image inputs.",
                    "split aware: Use the split facts.",
                    "metric aligned: Optimize for the metric.",
                ],
            )
            self.assertTrue(
                all(
                    "bootstrap root draft plan" in task.selection_reason
                    for task in tasks
                )
            )

    def test_draft_plan_batch_prompt_is_text_only_and_no_code_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = SimpleNamespace(
                workspace_dir=Path(tmp),
                log_dir=Path(tmp) / "logs",
                exec=SimpleNamespace(include_medical_packages=True),
                agent=SimpleNamespace(
                    search=SimpleNamespace(
                        max_debug_depth=2,
                        max_children_per_node=3,
                        topk_improve=1,
                    ),
                    data_summary=True,
                    expose_prediction=False,
                    enable_node_tools=False,
                    max_node_tool_calls=4,
                    enable_monai_knowledge_base=False,
                    enable_web_search=False,
                    web_search_topk=2,
                    code=SimpleNamespace(model="test-model", temp=0.7),
                    knowledge_base=SimpleNamespace(model="test-model", temp=0.0),
                ),
            )
            existing = Node(
                code="SECRET_CODE_SHOULD_NOT_REACH_PLANNER",
                plan="Use already observed task-relevant image evidence.",
            )
            existing.step = 0
            existing.is_buggy = False
            existing.metric = MetricValue(0.7, maximize=True)
            agent = ProposalAgent("task", cfg, Journal(nodes=[existing]))  # type: ignore[arg-type]
            agent.data_summary = "data summary"

            with patch(
                "biomaker.agents.proposal_agent.query",
                return_value={
                    "plans": [
                        {"name": "next", "plan": "Use task facts for the next plan."}
                    ]
                },
            ) as query_mock:
                plans = agent.generate_draft_plan_batch(1)

            prompt = query_mock.call_args.kwargs["system_message"]
            prompt_text = json.dumps(prompt, ensure_ascii=False, default=str)

        self.assertEqual(plans, ["next: Use task facts for the next plan."])
        self.assertIn("Existing Root Draft Plan Summaries", prompt)
        self.assertIn("Use already observed task-relevant image evidence.", prompt_text)
        self.assertNotIn("SECRET_CODE_SHOULD_NOT_REACH_PLANNER", prompt_text)
        self.assertNotIn("simple robust baseline", prompt_text)
        self.assertNotIn("fallback strategy", prompt_text)
        self.assertNotIn("learned-model", prompt_text)

    def test_draft_prompt_includes_assigned_plan(self) -> None:
        cfg = SimpleNamespace(
            workspace_dir=Path("."),
            log_dir=Path("."),
            exec=SimpleNamespace(include_medical_packages=True),
            agent=SimpleNamespace(
                data_summary=False,
                expose_prediction=False,
                enable_node_tools=False,
                max_node_tool_calls=4,
                enable_monai_knowledge_base=False,
                enable_web_search=False,
                web_search_topk=2,
                code=SimpleNamespace(model="test-model", temp=0.0),
                knowledge_base=SimpleNamespace(model="test-model", temp=0.0),
            ),
        )
        agent = ProposalAgent("task", cfg, Journal())  # type: ignore[arg-type]
        agent.set_generation_context(
            pending_round_nodes=[],
            draft_plan="Train a small CNN with strict evaluator outputs.",
        )

        prompt = agent.context_builder.build_draft()

        self.assertEqual(
            prompt["Current Selection"]["Assigned draft plan"],
            "Train a small CNN with strict evaluator outputs.",
        )

    def test_task_selection_stops_when_no_frontier_remains(self) -> None:
        cfg = SimpleNamespace(
            agent=SimpleNamespace(
                search=SimpleNamespace(
                    max_debug_depth=0,
                    max_children_per_node=1,
                    topk_improve=1,
                )
            ),
            exec=SimpleNamespace(
                include_medical_packages=True,
            ),
        )
        failed_draft = Node(code="raise RuntimeError('x')")
        failed_draft.step = 0
        failed_draft.is_buggy = True

        tasks = BaseAgent("task", cfg, Journal(nodes=[failed_draft])).select_round_tasks()  # type: ignore[arg-type]

        self.assertEqual(tasks, [])

    def test_tree_records_keep_full_node_code(self) -> None:
        selected_code = "x = 1\n" * 10000
        sibling_code = "y = 2\n" * 10000
        selected = Node(code=selected_code, plan="long selected parent")
        selected.step = 0
        selected.is_buggy = False
        sibling = Node(code=sibling_code, plan="long sibling draft")
        sibling.step = 1
        sibling.is_buggy = False
        journal = Journal(nodes=[selected, sibling])

        records = journal.generate_tree_records(stage="draft")

        self.assertGreater(len(records), 12000)
        self.assertIn(selected_code, records)
        self.assertIn(sibling_code, records)

    def test_code_extraction_accepts_common_python_fence_variants(self) -> None:
        for fence in ("```python", "```Python", "``` py", "```python3"):
            blocks = extract_code_blocks(f"{fence}\nprint('ok')\n```")
            self.assertEqual(len(blocks), 1)
            self.assertIn("print", blocks[0])

    def test_tool_decision_context_includes_full_tree_records(self) -> None:
        cfg = SimpleNamespace(
            workspace_dir=Path("."),
            log_dir=Path("."),
            exec=SimpleNamespace(include_medical_packages=True),
            agent=SimpleNamespace(
                enable_node_tools=False,
                max_node_tool_calls=4,
                data_summary=False,
                expose_prediction=False,
                enable_monai_knowledge_base=False,
                enable_web_search=False,
                web_search_topk=2,
                code=SimpleNamespace(model="test-model", temp=0.0),
                knowledge_base=SimpleNamespace(model="test-model", temp=0.0),
            ),
        )
        agent = ProposalAgent("task", cfg, Journal())  # type: ignore[arg-type]
        agent.data_summary = ""
        tree_records = "TREE\n" + ("x" * 5000)

        context = agent._tool_decision_context(
            "debug",
            {
                "Tree Records": tree_records,
                "Current Selection": {"Selection reason": "debug selected"},
            },
        )

        self.assertEqual(context["Tree Records"], tree_records)
        self.assertEqual(
            context["Current Selection"]["Selection reason"], "debug selected"
        )

    def test_inspect_api_cannot_read_raw_exp_guide(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "input").mkdir()
            (workspace / "exp_guide.json").write_text(
                '{"target": {"hidden_answers": ["class"]}}',
                encoding="utf-8",
            )

            result = python_probe(
                "from pathlib import Path\nPath('exp_guide.json').read_text()",
                cwd=workspace,
            )

        self.assertIn("PermissionError", result)

    def test_inspect_api_can_read_symlinked_input_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            dataset = root / "dataset"
            (workspace / "input").mkdir(parents=True)
            dataset.mkdir()
            (dataset / "train.csv").write_text("case_id,label\n0,1\n", encoding="utf-8")
            (workspace / "input" / "train.csv").symlink_to(dataset / "train.csv")

            result = python_probe(
                "from pathlib import Path\nprint(Path('input/train.csv').read_text())",
                cwd=workspace,
            )

        self.assertIn("exit_code=0", result)
        self.assertIn("case_id,label", result)

    def test_inspect_api_blocks_parent_escape_from_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "input").mkdir()
            (workspace / "exp_guide.json").write_text("secret", encoding="utf-8")

            result = python_probe(
                "from pathlib import Path\nPath('input/../exp_guide.json').read_text()",
                cwd=workspace,
            )

        self.assertIn("PermissionError", result)

    def test_working_path_normalization_keeps_relative_refs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            working = workspace / "working"
            mask = working / "pred_masks" / "case.png"
            mask.parent.mkdir(parents=True, exist_ok=True)
            mask.write_bytes(b"not read here")
            _write_csv(
                working / "submission.csv",
                [{"case_id": "case", "mask_path": str(mask.resolve())}],
            )

            messages = normalize_working_outputs(workspace, "path normalization")
            self.assertTrue(messages)
            with (working / "submission.csv").open(newline="", encoding="utf-8") as f:
                row = next(csv.DictReader(f))
            self.assertEqual(row["mask_path"], "working/pred_masks/case.png")
            self.assertNotIn(str(workspace), row["mask_path"])

    def test_artifact_validation_accepts_nifti_mask_references(self) -> None:
        if importlib.util.find_spec("SimpleITK") is None:
            self.skipTest("SimpleITK is not installed")
        import SimpleITK as sitk

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            split_dir = workspace / "input" / "biomaker_splits"
            working = workspace / "working"
            _write_csv(split_dir / "agent_test.csv", [{"case_id": "case"}])
            _write_json(
                workspace / "evaluator_prediction_spec.json",
                {
                    "test_submission": "./working/submission.csv",
                    "key_columns": ["case_id"],
                    "prediction_columns": ["mask_path"],
                    "file_reference_columns": ["mask_path"],
                    "primary_metric": "binary_dice",
                    "prediction_type": "mask_file_path",
                },
            )
            mask_path = working / "pred_masks" / "case.nii.gz"
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            image = sitk.GetImageFromArray(np.zeros((3, 4, 5), dtype=np.uint8))
            sitk.WriteImage(image, str(mask_path))
            _write_csv(
                working / "submission.csv",
                [{"case_id": "case", "mask_path": str(mask_path)}],
            )

            result = validate_working_outputs(workspace, "segmentation")
            self.assertTrue(result.ok, result.messages)

    def test_prediction_contract_uses_split_manifest_file_references(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            split_dir = workspace / "input" / "biomaker_splits"
            working = workspace / "working"
            _write_csv(split_dir / "agent_test.csv", [{"case_id": "case"}])
            _write_json(
                split_dir / "split_manifest.json",
                {
                    "manifest_version": 1,
                    "schema": "biomaker_split_manifest",
                    "sample_unit": "case",
                    "key_columns": ["case_id"],
                    "task_heads": ["segmentation"],
                    "prediction_contract": {
                        "test_submission": "./working/submission.csv",
                        "prediction_columns": ["mask_path"],
                        "file_reference_columns": ["mask_path"],
                        "prediction_type": "mask_file_path",
                    },
                    "metric_contract": {
                        "primary_metric": "binary_dice",
                        "maximize": True,
                    },
                },
            )
            mask = working / "pred_masks" / "case.png"
            mask.parent.mkdir(parents=True, exist_ok=True)
            mask.write_bytes(b"not a readable mask")
            _write_csv(
                working / "submission.csv",
                [{"case_id": "case", "mask_path": str(mask)}],
            )

            contract = load_prediction_contract(workspace)
            result = validate_working_outputs(workspace, "manifest contract")

            self.assertEqual(contract.file_reference_columns, ["mask_path"])
            self.assertFalse(result.ok)
            self.assertTrue(
                any(
                    "mask file is not readable" in message
                    for message in result.messages
                )
            )

    def test_experiment_contract_owns_hidden_answers_and_legacy_score_alias(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            split_dir = workspace / "input" / "biomaker_splits"
            _write_csv(split_dir / "agent_train.csv", [{"case_id": "a", "label": 0}])
            _write_csv(split_dir / "agent_val.csv", [{"case_id": "b"}])
            _write_csv(split_dir / "agent_test.csv", [{"case_id": "c"}])
            _write_json(
                split_dir / "split_plan.json",
                {
                    "hidden_answers": {
                        "score_val": {
                            "file": "score_val.csv",
                            "answer_columns": ["label"],
                        },
                        "test": {"file": "test.csv", "answer_columns": ["label"]},
                    }
                },
            )
            _write_json(
                workspace / "evaluator_prediction_spec.json",
                {
                    "test_submission": "./working/submission.csv",
                    "key_columns": ["case_id"],
                    "prediction_columns": ["prob_0", "prob_1"],
                    "primary_metric": "accuracy",
                },
            )

            contract = load_experiment_contract(workspace)

            self.assertEqual(
                contract.split_path("score_val"), split_dir / "agent_val.csv"
            )
            self.assertEqual(
                contract.hidden_answer_columns("score_val"),
                ["label"],
            )
            self.assertEqual(contract.prediction.key_columns, ["case_id"])
            self.assertEqual(contract.metric.primary_metric, "accuracy")

    def test_experiment_contract_resolves_split_plan_paths_under_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            split_dir = workspace / "input" / "biomaker_splits"
            _write_csv(
                split_dir / "agent_train.csv",
                [{"Id": "train_1", "class": 0, "path": "input/public/train/1.png"}],
            )
            _write_csv(
                split_dir / "score_val.csv",
                [{"Id": "val_1", "path": "input/public/val/1.png"}],
            )
            _write_csv(
                split_dir / "agent_test.csv",
                [{"Id": "test_1", "path": "input/public/test/1.png"}],
            )
            _write_json(
                split_dir / "split_plan.json",
                {
                    "agent_visible": {
                        "train": "biomaker_splits/agent_train.csv",
                        "score_val": "biomaker_splits/score_val.csv",
                        "test": "biomaker_splits/agent_test.csv",
                    },
                    "hidden_answers": {
                        "score_val": {
                            "file": "score_val.csv",
                            "answer_columns": ["class"],
                        },
                        "test": {"file": "test.csv", "answer_columns": ["class"]},
                    },
                },
            )
            _write_json(
                workspace / "evaluator_prediction_spec.json",
                {
                    "validation_artifact": "./working/validation_predictions.csv",
                    "test_submission": "./working/submission.csv",
                    "key_columns": ["Id"],
                    "prediction_columns": ["prob_0", "prob_1"],
                    "primary_metric": "accuracy",
                    "score_split": "./input/biomaker_splits/score_val.csv",
                },
            )

            contract = load_experiment_contract(workspace)
            guide = build_exp_guide(workspace)

        self.assertEqual(
            contract.split_path("train"),
            split_dir / "agent_train.csv",
        )
        self.assertEqual(
            contract.splits["train"].public_ref,
            "./input/biomaker_splits/agent_train.csv",
        )
        self.assertEqual(
            guide["splits"]["train"]["columns"],
            ["Id", "class", "path"],
        )
        self.assertEqual(
            guide["samples"]["train"]["first_row"]["class"],
            0,
        )

    def test_run_overview_uses_contract_split_refs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            private_dir = Path(tmp) / "private_gt"
            split_dir = workspace / "input" / "biomaker_splits"
            _write_csv(split_dir / "agent_train.csv", [{"case_id": "a", "label": 0}])
            _write_csv(split_dir / "score_val.csv", [{"case_id": "b"}])
            _write_csv(split_dir / "agent_test.csv", [{"case_id": "c"}])
            _write_csv(private_dir / "test.csv", [{"case_id": "c", "label": 1}])
            _write_json(
                split_dir / "split_plan.json",
                {
                    "hidden_answers": {
                        "test": {"file": "test.csv", "answer_columns": ["label"]}
                    }
                },
            )
            _write_json(
                workspace / "evaluator_prediction_spec.json",
                {
                    "test_submission": "./working/submission.csv",
                    "key_columns": ["case_id"],
                    "prediction_columns": ["prob_0", "prob_1"],
                    "primary_metric": "accuracy",
                },
            )

            text, payload = _run_overview_text(
                {"Task description": "classify"},
                None,
                workspace,
                private_dir,
            )

        self.assertEqual(
            payload["train_path"], "./input/biomaker_splits/agent_train.csv"
        )
        self.assertEqual(
            payload["node_selection_val_path"],
            "./input/biomaker_splits/score_val.csv",
        )
        self.assertIn("./input/biomaker_splits/agent_test.csv", text)

    def test_private_multilabel_uses_all_answer_columns(self) -> None:
        from biomaker.utils.evaluator_templates import _FIXED_CLASSIFICATION_EVALUATOR

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            private_dir = Path(tmp) / "private_gt"
            _write_json(
                workspace / "evaluator_prediction_spec.json",
                {
                    "test_submission": "./working/submission.csv",
                    "key_columns": ["Id"],
                    "prediction_columns": ["prob_1", "prob_2"],
                    "primary_metric": "mean_binary_accuracy",
                    "task_type": "multilabel",
                    "evaluator_api_version": 2,
                    "secondary_metrics": [],
                    "maximize": True,
                },
            )
            (workspace / "evaluator.py").write_text(
                "\n".join(
                    [
                        "def validate_submission():",
                        "    return {'valid': True}",
                    ]
                ),
                encoding="utf-8",
            )
            _write_csv(
                workspace / "working" / "submission.csv",
                [
                    {"Id": 1, "prob_1": 0.9, "prob_2": 0.1},
                    {"Id": 2, "prob_1": 0.2, "prob_2": 0.8},
                ],
            )
            _write_csv(
                private_dir / "test.csv",
                [
                    {"Id": 1, "class_1": 1, "class_2": 0},
                    {"Id": 2, "class_1": 0, "class_2": 1},
                ],
            )
            _write_json(
                private_dir / "private_gt.json",
                {
                    "test": {
                        "file": "test.csv",
                        "answer_columns": ["class_1", "class_2"],
                    }
                },
            )

            _write_csv(
                workspace / "input/biomaker_splits/agent_test.csv",
                [{"Id": 1}, {"Id": 2}],
            )
            spec = json.loads(
                (workspace / "evaluator_prediction_spec.json").read_text()
            )
            spec["answer_columns"] = ["class_1", "class_2"]
            (workspace / "evaluator.py").write_text(
                _FIXED_CLASSIFICATION_EVALUATOR.replace("__BIOMAKER_SPEC__", repr(spec))
            )
            result = evaluate_private_outputs(workspace, private_dir)
            self.assertIsNotNone(result)
            private_result = result["results"][0]
            self.assertEqual(
                private_result["metric_name"], "private_mean_binary_accuracy"
            )
            self.assertEqual(private_result["value"], 1.0)

    def test_private_summary_excludes_buggy_entries_from_best(self) -> None:
        history = [
            {
                "node_id": "buggy",
                "is_buggy": True,
                "metric": {"value": 0.1, "maximize": True},
                "private_metrics": {
                    "results": [
                        {"name": "private_test", "value": 0.99, "maximize": True}
                    ]
                },
            },
            {
                "node_id": "good",
                "is_buggy": False,
                "metric": {"value": 0.8, "maximize": True},
                "private_metrics": {
                    "results": [
                        {"name": "private_test", "value": 0.8, "maximize": True}
                    ]
                },
            },
        ]
        summary = _private_metrics_summary(history)
        self.assertEqual(summary["private_test_best"]["node_id"], "good")
        self.assertEqual(summary["eligible_history_count"], 1)

    def test_environment_policy_ignores_comments_and_strings_for_installs(self) -> None:
        safe = "\n".join(
            [
                "# do not run: pip install forbidden",
                "note = 'pip install appears in documentation, not execution'",
                "print(note)",
            ]
        )
        self.assertTrue(validate_code_environment(safe).ok)

        blocked = "import subprocess\nsubprocess.run(['pip', 'install', 'x'])"
        result = validate_code_environment(blocked)
        self.assertFalse(result.ok)
        self.assertEqual(result.error_type, "DependencyPolicyError")

    def test_environment_policy_allows_guarded_allowed_optional_import(self) -> None:
        code = "\n".join(
            [
                "try:",
                "    import lifelines",
                "except ImportError:",
                "    lifelines = None",
                "print(lifelines)",
            ]
        )
        self.assertTrue(validate_code_environment(code).ok)

    def test_environment_policy_rejects_disallowed_dynamic_imports(self) -> None:
        cases = [
            "__import__('cv2')",
            "import importlib\nimportlib.import_module('cv2')",
        ]
        for code in cases:
            with self.subTest(code=code):
                result = validate_code_environment(code)
                self.assertFalse(result.ok)
                self.assertEqual(result.error_type, "DependencyPolicyError")
                self.assertEqual(result.exc_info["package"], "cv2")

    def test_environment_policy_rejects_opaque_dynamic_imports(self) -> None:
        code = "\n".join(
            [
                "name = 'cv2'",
                "__import__(name)",
            ]
        )
        result = validate_code_environment(code)
        self.assertFalse(result.ok)
        self.assertEqual(result.error_type, "DependencyPolicyError")
        self.assertIn("not statically verifiable", result.exc_info["reason"])

    def test_allowed_packages_are_declared_in_requirements(self) -> None:
        requirements_path = Path(__file__).resolve().parents[1] / "requirements.txt"
        declared = set()
        for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            name = re.split(r"[<>=!~\[]", line, maxsplit=1)[0]
            declared.add(name.strip().lower().replace("_", "-"))

        allowed = {
            spec.distribution.lower().replace("_", "-")
            for spec in allowed_package_specs(include_medical=True)
        }

        self.assertFalse(
            allowed - declared,
            f"Allowed packages missing from requirements.txt: {sorted(allowed - declared)}",
        )

    def test_interpreter_missing_working_dir_raises_without_assert(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing"
            with self.assertRaises(FileNotFoundError):
                Interpreter(missing, enforce_environment=False)

    def test_interpreter_uses_ipython_traceback_mode_without_breaking_execution(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            interpreter = Interpreter(
                workspace,
                timeout=10,
                format_tb_ipython=True,
                enforce_environment=False,
            )
            result = interpreter.run("1 / 0")
            self.assertEqual(result.exc_type, "ZeroDivisionError")
            self.assertIn("ZeroDivisionError", "\n".join(result.term_out))


if __name__ == "__main__":
    unittest.main()
