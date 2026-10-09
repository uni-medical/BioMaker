"""Trusted fitted-state and portable inference probe, run only by Interpreter.

The Python audit/profile guards catch ordinary accidental downloads, retraining,
subprocesses, and dependencies on the old workspace. They are not an OS security
sandbox against adversarial code. The existing Interpreter owns process cleanup,
the execution environment, and timeout.
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import sysconfig
import tempfile


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def _install_guards(artifact_dir: Path):
    active = True
    libraries = {
        Path(value).resolve()
        for name, value in sysconfig.get_paths().items()
        if name in {"stdlib", "platstdlib", "purelib", "platlib"}
    }
    # torch/numpy shared-library discovery can read system identity/loader files.
    system_roots = [Path("/System/Library"), Path("/usr/lib"), Path("/etc")]
    allowed_reads = [artifact_dir, *libraries, *system_roots]

    def audit(event, args):
        if not active:
            return
        if event in {
            "socket.connect",
            "socket.connect_ex",
            "socket.getaddrinfo",
            "subprocess.Popen",
            "os.system",
            "os.posix_spawn",
            "os.fork",
            "os.forkpty",
        }:
            raise RuntimeError(
                "model inference probe forbids network and child processes"
            )
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).resolve()
            if path in {Path("/dev/urandom"), Path("/dev/null")}:
                return
            mode = args[1]
            flags = args[2]
            writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
                isinstance(flags, int)
                and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)
            )
            if writing and not path.is_relative_to(artifact_dir):
                raise RuntimeError(
                    "model inference probe cannot write outside its temporary artifact copy"
                )
            if not writing and not any(
                path.is_relative_to(root) for root in allowed_reads
            ):
                raise RuntimeError(
                    "model inference probe cannot read outside saved artifacts and runtime libraries"
                )

    sys.addaudithook(audit)

    def profile(frame, event, arg):
        if event != "call":
            return
        name = frame.f_code.co_name
        module = str(frame.f_globals.get("__name__", ""))
        if (
            name
            in {
                "fit",
                "partial_fit",
                "fit_transform",
                "backward",
                "fit_generator",
                "train_on_batch",
            }
            or (name == "train" and module.startswith(("lightgbm", "xgboost")))
            or (name == "step" and module.startswith("torch.optim"))
        ):
            raise RuntimeError(
                "model inference probe must reload fitted state without training"
            )

    sys.setprofile(profile)
    try:
        yield
    finally:
        # Python cannot unregister an audit hook. Disable this process-local
        # hook after candidate inference so trusted tempfile cleanup (including
        # dir_fd-relative opens) is not mistaken for external artifact access.
        active = False
        sys.setprofile(None)


def _load_checkpoint(path: Path, format_name: str):
    import numpy as np

    if format_name == "joblib":
        import joblib
        from sklearn.utils.validation import check_is_fitted

        model = joblib.load(path)
        # Supports normal sklearn Pipeline objects and fitted sklearn wrappers,
        # but deliberately rejects an arbitrary pickle or an unfitted estimator.
        check_is_fitted(model)
    elif format_name == "lightgbm":
        import lightgbm

        model = lightgbm.Booster(model_file=str(path))
        if model.num_trees() < 1 or model.num_feature() < 1:
            raise ValueError("LightGBM checkpoint contains no trained trees")
    elif format_name == "torch_state_dict":
        import torch

        state = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(state, dict):
            state = state.get("state_dict", state.get("model_state_dict", state))
        if not isinstance(state, dict) or not state:
            raise ValueError("torch checkpoint must contain a nonempty state_dict")
        tensors = [value for value in state.values() if isinstance(value, torch.Tensor)]
        if not tensors or any(
            value.numel() == 0 or not torch.isfinite(value).all() for value in tensors
        ):
            raise ValueError(
                "torch checkpoint contains empty or non-finite model tensors"
            )
    elif format_name == "xgboost":
        import xgboost

        model = xgboost.Booster()
        model.load_model(path)
        if model.num_boosted_rounds() < 1 or model.num_features() < 1:
            raise ValueError("XGBoost checkpoint contains no trained rounds")
    elif format_name == "npz":
        with np.load(path, allow_pickle=False) as state:
            arrays = [state[key] for key in state.files]
            if not arrays or any(
                value.size == 0
                or not np.issubdtype(value.dtype, np.number)
                or not np.isfinite(value).all()
                for value in arrays
            ):
                raise ValueError(
                    "npz checkpoint must contain nonempty finite numeric parameters"
                )
    else:
        raise ValueError("unsupported model checkpoint format")


def run_probe(working_dir: str, nonce: str) -> None:
    source = Path(working_dir).resolve()
    manifest = json.loads((source / "model_manifest.json").read_text())
    # Copy only declared artifacts to a relocated directory. The inference code
    # cannot accidentally reuse training input, caches, or an unsaved model.
    with tempfile.TemporaryDirectory(prefix="biomaker-model-probe-") as temporary:
        artifacts = Path(temporary).resolve()
        for item in manifest["files"]:
            destination = artifacts / item["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / item["path"], destination)
        # Import numerical dependencies before limiting file reads. They may
        # inspect platform metadata, but no generated code is imported yet.
        import numpy as np

        checkpoint_formats = {
            item.get("format")
            for item in manifest["files"]
            if item["role"] == "checkpoint"
        }
        if "joblib" in checkpoint_formats:
            import joblib
            import sklearn.utils.validation
        if "lightgbm" in checkpoint_formats:
            import lightgbm
        if "xgboost" in checkpoint_formats:
            import xgboost
        if "torch_state_dict" in checkpoint_formats:
            import torch

            torch.set_num_threads(1)
        os.environ.update(
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            HF_HUB_DISABLE_TELEMETRY="1",
            PYTHONDONTWRITEBYTECODE="1",
        )
        # Installed libraries (for example filelock imported by timm) may probe
        # filesystem capabilities during import. Keep their temporary files and
        # empty caches inside this disposable copy instead of weakening the
        # no-training-workspace/no-user-cache boundary.
        scratch = artifacts / ".probe_scratch"
        scratch.mkdir()
        tempfile.tempdir = str(scratch)
        os.environ.update(
            TMPDIR=str(scratch),
            TMP=str(scratch),
            TEMP=str(scratch),
            HOME=str(scratch / "home"),
            USERPROFILE=str(scratch / "home"),
            XDG_CACHE_HOME=str(scratch / "cache"),
            HF_HOME=str(scratch / "huggingface"),
            TORCH_HOME=str(scratch / "torch"),
            MPLCONFIGDIR=str(scratch / "matplotlib"),
        )
        sys.dont_write_bytecode = True
        original_cwd = Path.cwd()
        os.chdir(artifacts)
        # Guards cover checkpoint loading and generated inference. Trusted
        # receipt emission and tempfile cleanup run after that guarded scope.
        with _install_guards(artifacts):
            for item in manifest["files"]:
                if _sha(artifacts / item["path"]) != item["sha256"]:
                    raise ValueError("model artifact changed while copying")
                if item["role"] == "checkpoint":
                    _load_checkpoint(artifacts / item["path"], item["format"])
            inference = manifest["inference"]
            inputs = json.loads((artifacts / inference["input"]).read_text())
            expected = np.asarray(
                json.loads((artifacts / inference["expected_output"]).read_text()),
                dtype=float,
            )
            if (
                expected.size == 0
                or expected.ndim == 0
                or expected.shape[0] != len(inputs)
                or not np.isfinite(expected).all()
            ):
                raise ValueError(
                    "model probe expected output must contain finite predictions for each example"
                )
            spec = importlib.util.spec_from_file_location(
                "biomaker_saved_inference", artifacts / inference["entrypoint"]
            )
            if spec is None or spec.loader is None:
                raise ValueError("inference entrypoint must be a Python module")
            module = importlib.util.module_from_spec(spec)
            sys.path.insert(0, str(artifacts))
            spec.loader.exec_module(module)
            if not callable(getattr(module, "predict", None)):
                raise ValueError(
                    "inference entrypoint must expose predict(inputs, artifact_dir)"
                )
            actual = np.asarray(module.predict(inputs, artifacts), dtype=float)
            if (
                actual.shape != expected.shape
                or not np.isfinite(actual).all()
                or not np.allclose(actual, expected, rtol=1e-4, atol=1e-6)
            ):
                raise ValueError(
                    "reloaded model predictions differ from the saved in-memory predictions"
                )
            hashes = {
                str(Path(item["path"])): _sha(artifacts / item["path"])
                for item in manifest["files"]
            }
            if any(
                hashes[str(Path(item["path"]))] != item["sha256"]
                for item in manifest["files"]
            ):
                raise ValueError("inference modified its saved model artifacts")
            receipt = {
                "ok": True,
                "nonce": nonce,
                "checkpoint_count": sum(
                    item["role"] == "checkpoint" for item in manifest["files"]
                ),
                "probe_examples": len(inputs),
                "output_shape": list(actual.shape),
                "artifact_sha256": hashes,
                "offline": True,
                "relocated": True,
                "training_disabled": True,
                "guard_scope": "python_audit_and_profile_not_os_sandbox",
                "verification_scope": "checkpoint_loadability_and_saved_small_sample_prediction_equivalence; not proof_of_training_or_all_prediction_dependencies",
            }
        # Restore only the cwd before TemporaryDirectory's trusted cleanup. No
        # further candidate code runs, and no original-workspace file is opened.
        os.chdir(original_cwd)
        print("BIOMAKER_MODEL_PROBE " + json.dumps(receipt, sort_keys=True))
