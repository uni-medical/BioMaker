#!/usr/bin/env bash
set -euo pipefail

uv run python -m biomaker.monai_rag.query_rag_db
