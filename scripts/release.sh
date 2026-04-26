#!/usr/bin/env bash
set -euo pipefail

read -r -p "Build and upload BioMaker to PyPI? Type 'release' to continue: " answer
if [[ "$answer" != "release" ]]; then
  echo "Release cancelled."
  exit 1
fi

rm -rf build/ dist/ *.egg-info/
python -m build
python -m twine check dist/*
python -m twine upload dist/*
