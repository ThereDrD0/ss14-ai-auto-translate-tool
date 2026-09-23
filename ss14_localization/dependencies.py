from __future__ import annotations

import importlib
import platform
import subprocess
import sys
from pathlib import Path
from types import ModuleType

DEPS_DIR = (
    Path(__file__).resolve().parents[1]
    / ".deps"
    / (
        f"{sys.platform}-py{sys.version_info.major}{sys.version_info.minor}-"
        f"{platform.machine().lower()}"
    )
)


def import_or_install(import_name: str, package_name: str | None = None) -> ModuleType:
    if DEPS_DIR.exists():
        _prepend_deps_dir()

    try:
        return importlib.import_module(import_name)
    except ModuleNotFoundError:
        package = package_name or import_name
        _ensure_pip()
        DEPS_DIR.mkdir(parents=True, exist_ok=True)
        _prepend_deps_dir()
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "--target",
                str(DEPS_DIR),
                package,
            ],
            stdout=sys.stderr,
        )
        importlib.invalidate_caches()
        _purge_module(import_name)
        return importlib.import_module(import_name)


def _ensure_pip() -> None:
    probe = subprocess.run(
        [sys.executable, "-m", "pip", "--version"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if probe.returncode == 0:
        return

    subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade", "--user"])


def _prepend_deps_dir() -> None:
    deps = str(DEPS_DIR)
    if deps not in sys.path:
        sys.path.insert(0, deps)


def _purge_module(import_name: str) -> None:
    root_name = import_name.split(".", 1)[0]
    for module_name in list(sys.modules):
        if module_name == root_name or module_name.startswith(f"{root_name}."):
            del sys.modules[module_name]
