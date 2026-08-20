from __future__ import annotations

import ntpath
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path, PureWindowsPath
from typing import Any

PICKER_TIMEOUT_SECONDS = 120.0
_CANCEL_MARKER = "__BIOMAKER_DIRECTORY_PICKER_CANCELLED__"

_MACOS_SCRIPT = f'''try
    set selectedFolder to choose folder with prompt "Choose a dataset folder"
    return POSIX path of selectedFolder
on error number -128
    return "{_CANCEL_MARKER}"
end try'''

_WINDOWS_SCRIPT = f'''Add-Type -AssemblyName System.Windows.Forms
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$dialog = New-Object System.Windows.Forms.FolderBrowserDialog
$dialog.Description = 'Choose a dataset folder'
$dialog.ShowNewFolderButton = $false
try {{
    if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {{
        [Console]::Out.WriteLine($dialog.SelectedPath)
    }} else {{
        [Console]::Out.WriteLine('{_CANCEL_MARKER}')
    }}
}} finally {{
    $dialog.Dispose()
}}'''


class DirectoryPickerError(RuntimeError):
    """Base error for a failed native directory-picker invocation."""


class DirectoryPickerUnsupported(DirectoryPickerError):
    """Raised when no supported native directory picker is available."""


class DirectoryPickerTimeout(DirectoryPickerError):
    """Raised when the native directory picker remains open for too long."""


class DirectoryPickerBusy(DirectoryPickerError):
    """Raised when another native directory picker is already open."""


_PICKER_LOCK = threading.Lock()


def _is_wsl() -> bool:
    if not sys.platform.startswith("linux"):
        return False
    if os.environ.get("WSL_INTEROP") or os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        release = Path("/proc/sys/kernel/osrelease").read_text(encoding="utf-8")
    except OSError:
        return False
    normalized = release.lower()
    return "microsoft" in normalized or "wsl" in normalized


def _command_for_platform(platform: str) -> tuple[list[str], str]:
    if platform == "darwin":
        executable = shutil.which("osascript")
        if executable:
            return [executable, "-e", _MACOS_SCRIPT], "macos"
        raise DirectoryPickerUnsupported(
            "The macOS folder picker is not available on this computer."
        )

    if platform == "win32":
        executable = shutil.which("powershell.exe") or shutil.which("powershell")
        if executable:
            return [
                executable,
                "-NoProfile",
                "-NonInteractive",
                "-STA",
                "-Command",
                _WINDOWS_SCRIPT,
            ], "windows"
        raise DirectoryPickerUnsupported(
            "The Windows folder picker is not available on this computer."
        )

    if platform.startswith("linux"):
        if _is_wsl():
            executable = shutil.which("powershell.exe")
            converter = shutil.which("wslpath")
            if executable and converter:
                return [
                    executable,
                    "-NoProfile",
                    "-NonInteractive",
                    "-STA",
                    "-Command",
                    _WINDOWS_SCRIPT,
                ], "wsl-windows"
        executable = shutil.which("zenity")
        if executable:
            return [
                executable,
                "--file-selection",
                "--directory",
                "--title=Choose a dataset folder",
            ], "linux"
        executable = shutil.which("kdialog")
        if executable:
            return [
                executable,
                "--getexistingdirectory",
                str(Path.home()),
                "--title",
                "Choose a dataset folder",
            ], "linux"
        raise DirectoryPickerUnsupported(
            "No supported Linux folder picker was found (zenity or kdialog)."
        )

    raise DirectoryPickerUnsupported(
        f"Native folder selection is not supported on {platform or 'this platform'}."
    )


def _absolute_path(raw_path: str, picker_kind: str) -> str:
    if picker_kind in {"windows", "wsl-windows"}:
        if not PureWindowsPath(raw_path).is_absolute():
            raise DirectoryPickerError(
                "The system folder picker returned an invalid path."
            )
        windows_path = ntpath.normpath(raw_path)
        if picker_kind == "windows":
            return windows_path

        converter = shutil.which("wslpath")
        if not converter:
            raise DirectoryPickerUnsupported(
                "The Windows folder path cannot be converted for WSL."
            )
        try:
            completed = subprocess.run(
                [converter, "-u", windows_path],
                capture_output=True,
                text=True,
                timeout=10.0,
                check=False,
                shell=False,
                stdin=subprocess.DEVNULL,
                encoding="utf-8",
                errors="strict",
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise DirectoryPickerError(
                "The selected Windows folder path could not be converted for WSL."
            ) from exc
        except UnicodeError as exc:
            raise DirectoryPickerError(
                "The converted WSL folder path could not be read."
            ) from exc

        converted = completed.stdout.rstrip("\r\n")
        if completed.returncode != 0 or not converted:
            raise DirectoryPickerError(
                "The selected Windows folder path could not be converted for WSL."
            )
        path = Path(converted)
        if not path.is_absolute():
            raise DirectoryPickerError(
                "The system folder picker returned an invalid WSL path."
            )
        return str(path.resolve())

    path = Path(raw_path)
    if not path.is_absolute():
        raise DirectoryPickerError("The system folder picker returned an invalid path.")
    return str(path.resolve())


def _select_directory_unlocked(
    *, timeout: float = PICKER_TIMEOUT_SECONDS
) -> dict[str, Any]:
    command, picker_kind = _command_for_platform(sys.platform)
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            shell=False,
            stdin=subprocess.DEVNULL,
            encoding="utf-8",
            errors="strict",
        )
    except subprocess.TimeoutExpired as exc:
        raise DirectoryPickerTimeout(
            "The system folder picker timed out before a folder was selected."
        ) from exc
    except FileNotFoundError as exc:
        raise DirectoryPickerUnsupported(
            "The system folder picker is no longer available."
        ) from exc
    except OSError as exc:
        raise DirectoryPickerError(
            "The system folder picker could not be opened."
        ) from exc
    except UnicodeError as exc:
        raise DirectoryPickerError(
            "The system folder picker returned an unreadable path."
        ) from exc

    stdout = completed.stdout.rstrip("\r\n")
    stderr = completed.stderr or ""
    cancelled = stdout == _CANCEL_MARKER
    if (
        picker_kind == "linux"
        and completed.returncode == 1
        and not stdout
        and not stderr.strip()
    ):
        cancelled = True
    if picker_kind == "macos" and completed.returncode != 0 and "-128" in stderr:
        cancelled = True
    if cancelled:
        return {"status": "cancelled", "path": None}

    if completed.returncode != 0 or not stdout:
        raise DirectoryPickerError(
            "The system folder picker closed without selecting a folder."
        )

    return {"status": "selected", "path": _absolute_path(stdout, picker_kind)}


def select_directory(
    *, timeout: float = PICKER_TIMEOUT_SECONDS
) -> dict[str, Any]:
    """Open one native folder picker and return its selected path."""

    if not _PICKER_LOCK.acquire(blocking=False):
        raise DirectoryPickerBusy("A system folder picker is already open.")
    try:
        return _select_directory_unlocked(timeout=timeout)
    finally:
        _PICKER_LOCK.release()
