#!/usr/bin/env bash
# Runs this crate's Python tests; also invoked by CI. Requires uv.
# Forwards extra args to pytest (e.g. `-k temperature -v`).
set -euo pipefail

cd "$(dirname "$0")"

uv run --python 3.12 --with-requirements requirements-dev.txt \
  pytest tests \
  --cov=src/python \
  --cov-report=term-missing \
  --cov-fail-under=55 \
  "$@"
