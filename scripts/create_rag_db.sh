#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${PYTHON:-}" && -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
else
  PYTHON_BIN="${PYTHON:-python3}"
fi

export PYTHON="$PYTHON_BIN"
pushd "$ROOT_DIR/biomaker/monai_rag"
rm -rf docs.monai.io
bash wget_monai_doc.sh # MONAI-1.4.0
"$PYTHON_BIN" create_monai_rag_db.py
popd
