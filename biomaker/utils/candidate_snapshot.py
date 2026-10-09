"""Atomic, public-only display snapshots; never a selection or scoring ledger."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time

MAX_NODES = 1500
MAX_BYTES = 12 * 1024 * 1024


def write_candidate_snapshot(log_dir: Path, nodes) -> None:
    nodes = list(nodes)
    public = []
    text_fields = []
    for node in nodes[:MAX_NODES]:
        value = {
            "id": node.id,
            "step": node.step,
            "scheduled_step": node.scheduled_step,
            "parent_id": node.parent.id if node.parent else None,
            "phase": {"generated": "queued", "completed": "evaluating"}.get(node.tag, node.tag),
            "agent_name": node.agent_name,
            "stage": node.node_status,
        }
        fields = {}
        for name, limit in (("plan", 32000), ("selection_reason", 32000), ("code", 256000)):
            text = getattr(node, name, None)
            if isinstance(text, str):
                fields[name] = text[:limit]
                if len(text) > limit:
                    value["content_truncated"] = True
        public.append(value)
        text_fields.append(fields)
    payload = {"version": 1, "updated_at": time.time(), "nodes": public, "total": len(nodes)}
    encode = lambda value: json.dumps(value, ensure_ascii=True, allow_nan=False).encode()
    # Reserve all lifecycle rows first, then spend a bounded budget on text.
    # Never repeatedly encode an unbounded collection of generated programs.
    remaining = MAX_BYTES - len(encode(payload)) - len(public) * 40
    for value, fields in zip(public, text_fields):
        for name, text in fields.items():
            cost = len(encode({name: text})) + 2
            if cost <= remaining:
                value[name] = text
                remaining -= cost
            else:
                value["content_truncated"] = True
    data = encode(payload)
    if len(data) > MAX_BYTES:
        raise ValueError("Candidate lifecycle rows exceed the snapshot limit")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=log_dir, prefix=".candidates_live-", delete=False) as stream:
            temporary = stream.name
            stream.write(data)
        os.replace(temporary, Path(log_dir) / "candidates_live.json")
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
