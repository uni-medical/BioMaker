#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${PYTHON:-}" && -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
else
  PYTHON_BIN="${PYTHON:-python3}"
fi

export PYTHON="$PYTHON_BIN"
RAG_MODE="${BIOMAKER_RAG_MODE:-keyword}"
EMBEDDING_MODEL="${BIOMAKER_EMBEDDING_MODEL:-text-embedding-3-large}"
if [[ "$RAG_MODE" != "embedding" && "$RAG_MODE" != "keyword" ]]; then
  echo "BIOMAKER_RAG_MODE must be 'embedding' or 'keyword'" >&2
  exit 2
fi

pushd "$ROOT_DIR/biomaker/monai_rag"
rm -rf docs.monai.io
bash wget_monai_doc.sh # MONAI-1.4.0
"$PYTHON_BIN" create_monai_rag_db.py \
  --mode "$RAG_MODE" \
  --embedding-model "$EMBEDDING_MODEL"
popd
