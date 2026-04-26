#!/usr/bin/env bash
set -euo pipefail

downloaded=0

# Mirror the MONAI 1.4.0 docs while preserving directory structure when wget is
# available. If that fails, fall back to a single-page download with Python.
if command -v wget >/dev/null 2>&1; then
  if wget --mirror --convert-links --adjust-extension --page-requisites --no-parent --wait=1 -e robots=off -c https://docs.monai.io/en/1.4.0/api.html; then
    downloaded=1
  else
    echo "Warning: wget failed; falling back to Python download."
  fi
fi

if [[ "$downloaded" -eq 0 ]]; then
  "${PYTHON:-python3}" - <<'PY'
import time
from pathlib import Path

import requests

url = "https://docs.monai.io/en/1.4.0/api.html"
out = Path("docs.monai.io/en/1.4.0/api.html")
out.parent.mkdir(parents=True, exist_ok=True)
last_error = None
for verify in (True, False):
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=60, verify=verify)
            response.raise_for_status()
            last_error = None
            break
        except Exception as exc:
            last_error = exc
            time.sleep(2 * (attempt + 1))
    if last_error is None:
        break
if last_error is not None:
    print(f"Warning: failed to download {url}: {last_error}")
    print("Continuing with installed MONAI package documentation fallback.")
    raise SystemExit(0)
out.write_text(response.text, encoding="utf-8")
print(f"Downloaded {url} -> {out}")
PY
fi

# Remove generated asset folders that are not useful for text indexing.
rm -rf docs.monai.io/en/1.4.0/_*
