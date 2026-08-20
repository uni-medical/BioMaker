from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from biomaker.utils.env import load_local_env, resolve_env_file


class EnvironmentFileTests(unittest.TestCase):
    def test_default_and_relative_override_do_not_depend_on_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "BioMaker"
            elsewhere = Path(tmp) / "elsewhere"
            root.mkdir()
            elsewhere.mkdir()

            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("BIOMAKER_ENV_FILE", None)
                previous_cwd = Path.cwd()
                try:
                    os.chdir(elsewhere)
                    self.assertEqual(resolve_env_file(root), (root / ".env").resolve())
                finally:
                    os.chdir(previous_cwd)

                os.environ["BIOMAKER_ENV_FILE"] = "secrets/provider.env"
                self.assertEqual(
                    resolve_env_file(root),
                    (root / "secrets" / "provider.env").resolve(),
                )

    def test_loader_uses_resolved_file_without_overwriting_shell_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "BioMaker"
            root.mkdir()
            (root / ".env").write_text(
                "BIOMAKER_ENV_TEST_VALUE=from-file\n"
                "BIOMAKER_ENV_TEST_FILE_ONLY=file-only\n",
                encoding="utf-8",
            )

            with patch.dict(
                os.environ,
                {"BIOMAKER_ENV_TEST_VALUE": "from-shell"},
                clear=False,
            ):
                os.environ.pop("BIOMAKER_ENV_FILE", None)
                os.environ.pop("BIOMAKER_ENV_TEST_FILE_ONLY", None)
                path = load_local_env(method_root=root)
                self.assertEqual(path, (root / ".env").resolve())
                self.assertEqual(os.environ["BIOMAKER_ENV_TEST_VALUE"], "from-shell")
                self.assertEqual(os.environ["BIOMAKER_ENV_TEST_FILE_ONLY"], "file-only")


if __name__ == "__main__":
    unittest.main()
