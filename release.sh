#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Clean up old builds ---
echo "--- Cleaning up old build artifacts ---"
rm -rf build/ dist/ *.egg-info/

# --- Build the package ---
echo "--- Building the source and wheel distributions ---"
python -m build

# --- Upload to PyPI ---
echo "--- Uploading to PyPI ---"
python -m twine upload dist/*

echo "--- Release process completed ---"
