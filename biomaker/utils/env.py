from __future__ import annotations

import os
from pathlib import Path

ENV_FILE_VARIABLE = "BIOMAKER_ENV_FILE"
DEFAULT_METHOD_ROOT = Path(__file__).resolve().parents[2]


def resolve_env_file(method_root: Path | str | None = None) -> Path:
    """Return BioMaker's configured environment file without depending on cwd."""

    root = Path(method_root or DEFAULT_METHOD_ROOT).expanduser().resolve()
    override = os.environ.get(ENV_FILE_VARIABLE)
    path = Path(override).expanduser() if override else root / ".env"
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def load_local_env(
    *, method_root: Path | str | None = None, override: bool = False
) -> Path:
    """Load BioMaker's canonical environment file and return its resolved path."""

    try:
        from dotenv import load_dotenv
    except ImportError:
        return resolve_env_file(method_root)

    path = resolve_env_file(method_root)
    load_dotenv(dotenv_path=path, override=override)
    return path
