from __future__ import annotations

import os
from pathlib import Path

TOOL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT_ENV = "SS14_REPO_ROOT"


class RepositoryNotFoundError(RuntimeError):
    pass


def find_repo_root(explicit: Path | None = None) -> Path:
    """Find the Space Station 14 checkout that contains this tool."""
    if explicit is not None:
        return _validate_repo_root(explicit)

    configured = os.environ.get(REPO_ROOT_ENV)
    if configured:
        return _validate_repo_root(Path(configured))

    starts = (TOOL_ROOT, Path.cwd().resolve())
    checked: set[Path] = set()
    for start in starts:
        for candidate in (start, *start.parents):
            candidate = candidate.resolve()
            if candidate in checked:
                continue
            checked.add(candidate)
            if _looks_like_repo_root(candidate):
                return candidate

    raise RepositoryNotFoundError(
        "Не найден корень Space Station 14: ожидается папка Resources/Locale. "
        f"Установите инструмент внутрь репозитория, задайте {REPO_ROOT_ENV} "
        "или передайте --repo-root ПУТЬ."
    )


def resolve_tool_file(path: Path | None, default: Path) -> Path:
    """Resolve bundled files independently from the parent repository layout."""
    if path is None:
        return TOOL_ROOT / default
    if path.is_absolute():
        return path
    return TOOL_ROOT / path


def _validate_repo_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not _looks_like_repo_root(resolved):
        raise RepositoryNotFoundError(
            f"{resolved} не похож на корень Space Station 14: не найдена папка Resources/Locale."
        )
    return resolved


def _looks_like_repo_root(path: Path) -> bool:
    return (path / "Resources" / "Locale").is_dir()
