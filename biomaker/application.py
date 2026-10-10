"""Shared run operations for the CLI, WebUI and desktop adapters.

The durable runner remains the only queue writer. Deployment-specific runtime
readiness can be injected as an admission policy, without inspecting HTTP paths.
"""

from __future__ import annotations

from pathlib import Path
from threading import RLock
from typing import Any, Callable

from biomaker.runtime.models import ExperimentInput, RunCreate
from biomaker.runtime.queries import RunQueries, session_run
from biomaker.runtime.runner import merge_options, resolve_run_options


class AdmissionError(RuntimeError):
    def __init__(self, detail: str, *, status_code: int = 422, code: str | None = None):
        self.detail, self.status_code, self.code = detail, status_code, code
        super().__init__(detail)


class RunService:
    """One application boundary over an existing durable RunManager."""

    def __init__(
        self, manager, settings=None, *, admission_policy: Callable | None = None
    ):
        self.manager = manager
        self.settings = settings if settings is not None else manager.settings
        self.admission_policy = admission_policy
        self.admission_lock = RLock()
        self.queries = RunQueries(self.settings, self.manager.store)
        self.observations = self.queries.observations

    def normalize_experiment(self, payload: ExperimentInput) -> dict[str, Any]:
        dataset = Path(payload.dataset_path).expanduser()
        if not dataset.is_absolute():
            dataset = self.settings.workspace_root / dataset
        dataset = dataset.resolve()
        if not dataset.is_dir():
            raise AdmissionError(f"Dataset directory does not exist: {dataset}")
        template = None
        if payload.task_template_path:
            candidate = Path(payload.task_template_path).expanduser()
            if not candidate.is_absolute():
                candidate = self.settings.workspace_root / candidate
            candidate = candidate.resolve()
            if (
                not candidate.is_relative_to(self.settings.experiments_root.resolve())
                or not candidate.is_file()
                or candidate.name != "task.md"
            ):
                raise AdmissionError("Invalid task template path")
            template = str(candidate)
        return {
            "dataset_path": str(dataset),
            "task_template_path": template,
            "task_body": payload.task_body,
            "advanced_defaults": payload.advanced_defaults.model_dump(),
        }

    def _admit(self, operation, experiment=None, options=None):
        if self.admission_policy is not None:
            self.admission_policy(operation, experiment, options or {})

    def submit(self, request: RunCreate | dict) -> dict[str, Any]:
        payload = RunCreate.model_validate(request)
        with self.admission_lock:
            experiment = self.normalize_experiment(payload.experiment)
            options = resolve_run_options(
                self.settings,
                merge_options(experiment["advanced_defaults"], payload.options),
            )
            self._admit("submit", experiment, options)
            return self.manager.enqueue_experiment(
                experiment,
                requested_exp_name=payload.requested_exp_name,
                option_patch=options,
            )

    def submit_command(self, command, **kwargs):
        """CLI retains its complete engine arguments, using the same admission gate."""
        with self.admission_lock:
            self._admit("submit", kwargs.get("project_snapshot"), kwargs.get("options"))
            return self.manager.enqueue_command(command, **kwargs)

    def get(self, run_id):
        return self.manager.store.get_record("runs", run_id)

    def present_run(self, record):
        """Project a mutation receipt without another ledger or log read."""
        return session_run(record)

    def status(self, run_id):
        return self.queries.status(run_id)

    def list_runs(self):
        return self.queries.list_runs()

    def detail(self, run_id):
        return self.queries.detail(run_id)

    def _controlled(self, run_id):
        record = self.get(run_id)
        if record is None:
            raise KeyError(run_id)
        if record.get("read_only") or not record.get("managed", True):
            raise AdmissionError(
                "Imported experiment records are view-only and cannot be controlled.",
                status_code=409,
            )
        return record

    def cancel(self, run_id):
        with self.admission_lock:
            self._controlled(run_id)
            return self.manager.cancel(run_id)

    def stop(self, run_id):
        with self.admission_lock:
            self._controlled(run_id)
            return self.manager.stop(run_id)

    def retry(self, run_id):
        with self.admission_lock:
            record = self._controlled(run_id)
            options = resolve_run_options(self.settings, record.get("options") or {})
            self._admit("retry", record.get("project_snapshot"), options)
            return self.manager.requeue(run_id)

    def resume_queue(self, pause_generation):
        with self.admission_lock:
            self._admit("resume")
            return self.manager.resume_queue(pause_generation)

    def results(self, run_id):
        return self.queries.results(run_id)

    def rescore(
        self,
        *,
        workspace: Path | str,
        prediction: Path | str,
        truth: Path | str,
        expected: Path | str,
        output: Path | str,
        timeout: int = 3600,
        workers: int = 1,
    ) -> dict[str, Any]:
        """Explicitly rescore saved artifacts into a new private receipt."""
        from biomaker.score import rescore_saved_predictions

        return rescore_saved_predictions(
            workspace=workspace,
            prediction=prediction,
            truth=truth,
            expected=expected,
            output=output,
            timeout=timeout,
            workers=workers,
        )
