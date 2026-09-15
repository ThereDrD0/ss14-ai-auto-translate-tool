#!/usr/bin/env bash
set -euo pipefail

script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
tool_root="$(CDPATH= cd -- "$script_dir/../.." && pwd)"

exec python3 "$tool_root/run.py" translate-all \
    --source-culture ru-RU \
    --target-culture en-US \
    "$@"
