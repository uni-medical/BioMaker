from __future__ import annotations

import re

from biomaker.utils import tree_export


def test_generate_html_inlines_tree_dependencies():
    payload = '{"exp_name":"offline-tree"}'

    html = tree_export.generate_html(payload)

    assert payload in html
    assert re.search(r'(?:src|href)=["\']https?://', html) is None
    assert "/*! p5.js v1.9.0" in html
    assert "Highlight.js v11.9.0" in html
    assert ".hljs{" in html
    assert "__BIOMAKER_VENDOR_" not in html
    assert "__BIOMAKER_LICENSE_" not in html
    assert "GNU LESSER GENERAL PUBLIC LICENSE" in html
    assert "BSD 3-Clause License" in html


def test_generate_html_escapes_script_terminators_from_tree_data():
    payload = (
        '{"code":"</script><script>window.injected=true</script>","line":"\u2028"}'
    )

    html = tree_export.generate_html(payload)

    assert "</script><script>window.injected" not in html
    assert "\\u003c/script>\\u003cscript>window.injected=true\\u003c/script>" in html
    assert "\\u2028" in html


def test_generate_skips_identical_canonical_tree(tmp_path, monkeypatch):
    tree = {"b": 2, "a": {"value": 1}}
    writes: list[str] = []
    real_atomic_write = tree_export._atomic_write_text

    monkeypatch.setattr(
        tree_export,
        "cfg_to_tree_struct",
        lambda _cfg, _journal, live=False: {**tree, "live": live},
    )
    monkeypatch.setattr(tree_export, "generate_html", lambda payload: payload)

    def tracked_write(path, content):
        writes.append(content)
        real_atomic_write(path, content)

    monkeypatch.setattr(tree_export, "_atomic_write_text", tracked_write)
    output = tmp_path / "tree_plot.html"

    first = tree_export.generate(object(), object(), output, live=True)
    second = tree_export.generate(object(), object(), output, live=True)

    assert first["changed"] is True
    assert second == {"changed": False, "revision": first["revision"]}
    assert len(str(first["revision"])) == 64
    assert len(writes) == 1

    tree["a"] = {"value": 3}
    third = tree_export.generate(object(), object(), output, live=True)

    assert third["changed"] is True
    assert third["revision"] != first["revision"]
    assert len(writes) == 2
