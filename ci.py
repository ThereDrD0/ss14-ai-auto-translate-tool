#!/usr/bin/env python3
"""CI entry point. Git operations belong to the host workflow, never to this runner."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Перевод для GitHub Actions без внутренних операций Git")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-json", type=Path)
    args, rest = parser.parse_known_args()
    command = [sys.executable, str(Path(__file__).resolve().parent / "run.py"), "translate-all", "--allow-partial"]
    if args.dry_run:
        command.append("--dry-run")
    if args.report_json:
        command.extend(["--report-json", str(args.report_json)])
    command.extend(rest)
    result = subprocess.run(command)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with Path(summary).open("a", encoding="utf-8") as stream:
            stream.write(f"\nПеревод локализации: код завершения {result.returncode}; пробный запуск {args.dry_run}.\n")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
