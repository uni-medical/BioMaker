from __future__ import annotations

import subprocess

import pytest

import webui.directory_picker as directory_picker


def test_macos_picker_uses_fixed_osascript_command_and_returns_absolute_path(
    tmp_path, monkeypatch
):
    captured = {}

    monkeypatch.setattr(directory_picker.sys, "platform", "darwin")
    monkeypatch.setattr(
        directory_picker.shutil,
        "which",
        lambda name: "/usr/bin/osascript" if name == "osascript" else None,
    )

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(
            command, 0, stdout=f"{tmp_path}\n", stderr=""
        )

    monkeypatch.setattr(directory_picker.subprocess, "run", fake_run)

    result = directory_picker.select_directory(timeout=2.5)

    assert result == {"status": "selected", "path": str(tmp_path.resolve())}
    assert captured["command"][:2] == ["/usr/bin/osascript", "-e"]
    assert "choose folder" in captured["command"][2]
    assert captured["kwargs"] == {
        "capture_output": True,
        "text": True,
        "timeout": 2.5,
        "check": False,
        "shell": False,
        "stdin": subprocess.DEVNULL,
        "encoding": "utf-8",
        "errors": "strict",
    }


def test_macos_cancel_and_timeout_are_distinct(monkeypatch):
    monkeypatch.setattr(directory_picker.sys, "platform", "darwin")
    monkeypatch.setattr(directory_picker.shutil, "which", lambda _name: "osascript")
    monkeypatch.setattr(
        directory_picker.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command,
            0,
            stdout=f"{directory_picker._CANCEL_MARKER}\n",
            stderr="",
        ),
    )
    assert directory_picker.select_directory() == {
        "status": "cancelled",
        "path": None,
    }

    def time_out(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(directory_picker.subprocess, "run", time_out)
    with pytest.raises(directory_picker.DirectoryPickerTimeout, match="timed out"):
        directory_picker.select_directory(timeout=0.25)


def test_linux_uses_kdialog_fallback_and_treats_user_cancel_as_cancelled(
    monkeypatch,
):
    requested = []

    monkeypatch.setattr(directory_picker.sys, "platform", "linux")
    monkeypatch.setattr(directory_picker, "_is_wsl", lambda: False)

    def fake_which(name):
        requested.append(name)
        return "/usr/bin/kdialog" if name == "kdialog" else None

    monkeypatch.setattr(directory_picker.shutil, "which", fake_which)
    monkeypatch.setattr(
        directory_picker.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 1, stdout="", stderr=""
        ),
    )

    assert directory_picker.select_directory() == {
        "status": "cancelled",
        "path": None,
    }
    assert requested == ["zenity", "kdialog"]


def test_linux_picker_error_is_not_treated_as_cancel(monkeypatch):
    monkeypatch.setattr(directory_picker.sys, "platform", "linux")
    monkeypatch.setattr(directory_picker, "_is_wsl", lambda: False)
    monkeypatch.setattr(directory_picker.shutil, "which", lambda _name: "/usr/bin/zenity")
    monkeypatch.setattr(
        directory_picker.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 1, stdout="", stderr="cannot open display"
        ),
    )

    with pytest.raises(directory_picker.DirectoryPickerError, match="closed"):
        directory_picker.select_directory()


def test_wsl_uses_windows_picker_and_converts_selected_path(tmp_path, monkeypatch):
    captured = []
    selected = "C:\\Users\\研究者\\数据集"
    converted = tmp_path / "数据集"

    monkeypatch.setattr(directory_picker.sys, "platform", "linux")
    monkeypatch.setattr(directory_picker, "_is_wsl", lambda: True)

    def fake_which(name):
        return {
            "powershell.exe": "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe",
            "wslpath": "/usr/bin/wslpath",
        }.get(name)

    def fake_run(command, **kwargs):
        captured.append((command, kwargs))
        if command[0].endswith("powershell.exe"):
            return subprocess.CompletedProcess(
                command, 0, stdout=selected + "\r\n", stderr=""
            )
        assert command == ["/usr/bin/wslpath", "-u", selected]
        return subprocess.CompletedProcess(
            command, 0, stdout=str(converted) + "\n", stderr=""
        )

    monkeypatch.setattr(directory_picker.shutil, "which", fake_which)
    monkeypatch.setattr(directory_picker.subprocess, "run", fake_run)

    assert directory_picker.select_directory()["path"] == str(converted.resolve())
    assert len(captured) == 2
    assert captured[0][0][1:5] == [
        "-NoProfile",
        "-NonInteractive",
        "-STA",
        "-Command",
    ]
    assert "FolderBrowserDialog" in captured[0][0][5]
    assert captured[1][1]["timeout"] == 10.0


def test_windows_uses_fixed_powershell_dialog(monkeypatch):
    captured = {}
    selected = "C:\\Users\\研究者\\数据集"

    monkeypatch.setattr(directory_picker.sys, "platform", "win32")
    monkeypatch.setattr(
        directory_picker.shutil,
        "which",
        lambda name: r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
        if name == "powershell.exe"
        else None,
    )

    def fake_run(command, **_kwargs):
        captured["command"] = command
        return subprocess.CompletedProcess(command, 0, stdout=selected + "\r\n", stderr="")

    monkeypatch.setattr(directory_picker.subprocess, "run", fake_run)

    assert directory_picker.select_directory()["path"] == selected
    assert captured["command"][1:5] == [
        "-NoProfile",
        "-NonInteractive",
        "-STA",
        "-Command",
    ]
    assert "FolderBrowserDialog" in captured["command"][5]
    assert "OutputEncoding" in captured["command"][5]


def test_only_one_picker_can_be_open(monkeypatch):
    monkeypatch.setattr(directory_picker.sys, "platform", "darwin")
    monkeypatch.setattr(directory_picker.shutil, "which", lambda _name: "osascript")

    with directory_picker._PICKER_LOCK:
        with pytest.raises(directory_picker.DirectoryPickerBusy, match="already open"):
            directory_picker.select_directory()


def test_unsupported_platform_and_invalid_picker_output(monkeypatch):
    monkeypatch.setattr(directory_picker.sys, "platform", "freebsd")
    with pytest.raises(directory_picker.DirectoryPickerUnsupported, match="not supported"):
        directory_picker.select_directory()

    monkeypatch.setattr(directory_picker.sys, "platform", "darwin")
    monkeypatch.setattr(directory_picker.shutil, "which", lambda _name: "osascript")
    monkeypatch.setattr(
        directory_picker.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 0, stdout="relative/path\n", stderr=""
        ),
    )
    with pytest.raises(directory_picker.DirectoryPickerError, match="invalid path"):
        directory_picker.select_directory()
