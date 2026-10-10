from __future__ import annotations

import io
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from rich.console import Console
from rich.text import Text

from biomaker import interpreter as module
from biomaker.interpreter import Interpreter, _StreamingLogRedactor
from biomaker.utils.config import RichLogHandler


@pytest.fixture
def live_records(monkeypatch):
    records = []

    class Capture(logging.Handler):
        def emit(self, record):
            if getattr(record, "biomaker_candidate_output", False):
                records.append(record)

    # Resource sampling is unrelated to pipe delivery; retain the real process,
    # process identity, containment, timeout and pipe implementation in these tests.
    monkeypatch.setattr(module, "_sample_cuda_total_gpu_memory_mb", lambda: None)
    monkeypatch.setattr(module, "_sample_process_usage", lambda *a, **kw: None)
    handler = Capture()
    previous_level = module.logger.level
    module.logger.setLevel(logging.INFO)
    module.logger.addHandler(handler)
    yield records
    module.logger.removeHandler(handler)
    module.logger.setLevel(previous_level)


def joined(records, channel="stdout", label=None):
    return "".join(
        record.candidate_output
        for record in records
        if record.candidate_output_stream == channel
        and (label is None or record.candidate_output_label == label)
    )


def test_live_output_arrives_before_process_finishes_without_flush_or_newline(
    tmp_path, live_records
):
    release = tmp_path / "release"
    code = (
        "import pathlib, sys, time\n"
        "print('epoch 1: 训练中', end='')\n"
        "print('stderr progress.', end='', file=sys.stderr)\n"
        f"release = pathlib.Path({str(release)!r})\n"
        "deadline = time.monotonic() + 8\n"
        "while not release.exists() and time.monotonic() < deadline:\n"
        "    time.sleep(0.02)\n"
        "print(' — 完成')\n"
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            Interpreter(
                tmp_path, timeout=10, enforce_environment=False, output_label="node-a"
            ).run,
            code,
        )
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if joined(live_records) and joined(live_records, "stderr"):
                break
            time.sleep(0.02)
        assert not future.done(), "candidate must still be running when logs arrive"
        assert joined(live_records) == "epoch 1: 训练中"
        assert joined(live_records, "stderr") == "stderr progress."
        release.touch()
        result = future.result(timeout=5)
    assert result.exc_type is None
    assert joined(live_records) == result.stdout == "epoch 1: 训练中 — 完成\n"
    assert joined(live_records, "stderr") == result.stderr == "stderr progress."


def test_split_utf8_crlf_and_unterminated_tail_are_not_duplicated(
    tmp_path, live_records
):
    code = (
        "import os, time\n"
        "os.write(1, b'first\\r')\n"
        "time.sleep(0.65)\n"
        "os.write(1, b'\\n\\xe8')\n"
        "time.sleep(0.65)\n"
        "os.write(1, b'\\xae\\xad\\xe7\\xbb\\x83\\rprogress\\n\\nlast')\n"
        "os.write(2, '警告: 末尾'.encode('utf-8'))\n"
    )
    result = Interpreter(tmp_path, timeout=6, enforce_environment=False).run(code)
    assert result.exc_type is None
    assert result.stdout == "first\n训练\nprogress\n\nlast"
    assert joined(live_records) == result.stdout
    assert joined(live_records, "stderr") == result.stderr == "警告: 末尾"


def test_large_dual_pipe_output_does_not_deadlock_or_drop_tail(tmp_path, live_records):
    size = 256 * 1024
    code = f"import os\nos.write(1, b'A' * {size} + b'\\nstdout-tail')\nos.write(2, b'Z' * {size} + b'\\nstderr-tail')"
    result = Interpreter(tmp_path, timeout=15, enforce_environment=False).run(code)
    assert result.exc_type is None
    assert joined(live_records) == result.stdout == "A" * size + "\nstdout-tail"
    assert (
        joined(live_records, "stderr") == result.stderr == "Z" * size + "\nstderr-tail"
    )
    assert max(len(record.candidate_output) for record in live_records) <= 4096


def test_timeout_preserves_delivered_output_once(tmp_path, live_records):
    result = Interpreter(tmp_path, timeout=1, enforce_environment=False).run(
        "import time\nprint('before timeout', end='')\ntime.sleep(10)"
    )
    assert result.exc_type == "TimeoutError"
    assert result.stdout == joined(live_records) == "before timeout"


def test_concurrent_candidates_keep_stream_and_candidate_attribution(
    tmp_path, live_records
):
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {}
        for label in ("node-one", "node-two"):
            directory = tmp_path / label
            directory.mkdir()
            code = f"import sys, time\nprint('{label} start')\ntime.sleep(0.7)\nprint('{label} end', file=sys.stderr)"
            futures[label] = pool.submit(
                Interpreter(
                    directory, timeout=5, enforce_environment=False, output_label=label
                ).run,
                code,
            )
        for label, future in futures.items():
            result = future.result(timeout=8)
            assert result.exc_type is None
            assert (
                joined(live_records, label=label) == result.stdout == f"{label} start\n"
            )
            assert (
                joined(live_records, "stderr", label)
                == result.stderr
                == f"{label} end\n"
            )


def test_real_child_secrets_split_between_reads_are_redacted_only_in_live_log(
    tmp_path, live_records
):
    secret = "synthetic-private-token-value"
    code = (
        "import os, time\n"
        "secret = os.environ['TEST_API_KEY']\n"
        "print(secret[:8], end='')\n"
        "time.sleep(0.65)\n"
        "print(secret[8:])\n"
        "print('sk-', end='')\n"
        "time.sleep(0.65)\n"
        "print('synthetic-never-real-key')\n"
        "print('Bearer synthetic.jwt.token', file=__import__('sys').stderr)\n"
    )
    result = Interpreter(
        tmp_path,
        timeout=6,
        enforce_environment=False,
        environment={"TEST_API_KEY": secret},
    ).run(code)
    assert result.exc_type is None
    assert secret in result.stdout
    assert "sk-synthetic-never-real-key" in result.stdout
    assert joined(live_records) == "[REDACTED]\n[REDACTED]\n"
    assert joined(live_records, "stderr") == "Bearer [REDACTED]\n"
    assert all(secret not in record.getMessage() for record in live_records)


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 7, 31, 4096])
def test_streaming_redaction_handles_every_boundary_and_bounds_token_storage(
    chunk_size,
):
    secret = "arbitrary-known-credential"
    redactor = _StreamingLogRedactor(
        {"SYNTHETIC_API_KEY": secret, "PREFIX_API_KEY": "sk-xxxxxx"}
    )
    raw = f"prefix {secret} / sk-" + "x" * 20000 + "\nBearer abc.def_ghi== end"
    chunks = []
    for index in range(0, len(raw), chunk_size):
        chunks.append(redactor.feed(raw[index : index + chunk_size]))
        assert len(redactor.pending) < len(secret)
    chunks.append(redactor.feed("", final=True))
    assert "".join(chunks) == "prefix [REDACTED] / [REDACTED]\nBearer [REDACTED] end"


@pytest.mark.parametrize(
    "secret",
    [
        "sk-synthetic.part-only",
        "sk-synthetic/key+with=padding",
        "Bearer synthetic phrase-only",
        "synthetic-credential:/punctuation!",
    ],
)
@pytest.mark.parametrize("token_prefix", ["", "sk-wrapper-"])
def test_known_punctuated_credentials_take_priority_at_every_chunk_boundary(
    secret, token_prefix
):
    raw = f"prefix {token_prefix}{secret}\nend"
    expected = "prefix [REDACTED]\nend"
    for boundary in range(len(raw) + 1):
        redactor = _StreamingLogRedactor({"SYNTHETIC_API_KEY": secret})
        rendered = redactor.feed(raw[:boundary])
        rendered += redactor.feed(raw[boundary:])
        rendered += redactor.feed("", final=True)
        assert rendered == expected, f"credential split at {boundary}"
    redactor = _StreamingLogRedactor({"SYNTHETIC_API_KEY": secret})
    chunks = []
    for character in raw:
        chunks.append(redactor.feed(character))
        assert len(redactor.pending) < len(secret)
    chunks.append(redactor.feed("", final=True))
    assert "".join(chunks) == expected


@pytest.mark.parametrize("rich_text", [False, True])
def test_file_log_keeps_long_paths_markup_and_unicode_without_terminal_wrapping(
    rich_text,
):
    stream = io.StringIO()
    console = Console(file=stream, width=30, force_terminal=False)
    handler = RichLogHandler(console=console, markup=True)
    path = "/long/" + "训练目录/" * 30 + "predictions.csv"
    record = logging.LogRecord(
        "test",
        logging.INFO,
        __file__,
        1,
        "[candidate abc stdout] %s [bold]literal",
        (path,),
        None,
    )
    if rich_text:
        record.msg = Text(record.getMessage())
        record.args = ()
    handler.emit(record)
    rendered = stream.getvalue()
    assert rendered.count("\n") == 1
    assert path + " [bold]literal" in rendered
    assert "INFO [candidate abc stdout]" in rendered
