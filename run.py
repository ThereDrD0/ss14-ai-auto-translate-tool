#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _load_env(path: Path) -> None:
    """Load a small .env file without adding another dependency."""
    if not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip()
        if value[:1] == value[-1:] and value[:1] in {"'", '"'}:
            value = value[1:-1]
        if name:
            os.environ.setdefault(name, value)


_env_parser = argparse.ArgumentParser(add_help=False)
_env_parser.add_argument("--env-file", type=Path)
_env_args, _ = _env_parser.parse_known_args()
if _env_args.env_file is not None and not _env_args.env_file.is_file():
    _env_parser.error(f"Файл окружения не найден: {_env_args.env_file}")
_load_env(_env_args.env_file or ROOT / ".env")

from ss14_localization.cli import main

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as error:
        print(f"Ошибка настройки или проверки: {error}", file=sys.stderr)
        raise SystemExit(2)
