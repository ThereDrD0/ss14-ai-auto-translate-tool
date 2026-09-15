from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any
import re

from .dependencies import import_or_install
from .filesystem import read_text, write_text_if_changed
from .fluent import message_map, parse_messages, render_entity_message, same_message_payload


@dataclass(frozen=True)
class EntityLocalization:
    source_path: Path
    prototype_id: str
    message_id: str
    name: str | None
    description: str | None
    suffix: str | None


def extract_entity_localizations(repo_root: Path, prototypes_root: Path) -> list[EntityLocalization]:
    absolute_root = repo_root / prototypes_root
    entries: list[EntityLocalization] = []
    yaml = _yaml_parser()

    for path in sorted(absolute_root.rglob("*.yml")):
        for item in _read_yaml_items(yaml, path):
            entry = _read_entity_localization(repo_root, path, item)
            if entry is not None:
                entries.append(entry)

    return entries


def build_entity_ftl(
    entries: list[EntityLocalization],
    existing_text: str | None = None,
    source_state: Mapping[str, str] | None = None,
) -> str:
    existing = message_map(existing_text or "")
    entries = _unique_entity_localizations(entries)
    output: list[str] = []
    last_path: Path | None = None
    source_hashes = source_state or {}

    for entry in entries:
        if last_path != entry.source_path:
            if output:
                output.append("")
            output.append(f"### {entry.source_path.as_posix()}")
            last_path = entry.source_path

        output.append("")
        source_text = _render_source_entry(entry)
        source_hash = _source_hash(source_text)
        preserved = existing.get(entry.message_id)
        source_message = parse_messages(source_text)[0]

        if preserved is not None and same_message_payload(source_message, preserved):
            preserved = None

        if preserved is not None and source_hashes.get(entry.message_id, source_hash) == source_hash:
            output.extend(preserved.lines)
            continue

        output.extend(source_text.splitlines())

    return "\n".join(output).strip("\n") + "\n"


def write_entity_ftl(
    repo_root: Path,
    prototypes_root: Path,
    output_path: Path,
    state_path: Path | None = None,
    dry_run: bool = False,
) -> tuple[int, bool]:
    entries = _unique_entity_localizations(extract_entity_localizations(repo_root, prototypes_root))
    absolute_output = repo_root / output_path
    absolute_state = repo_root / state_path if state_path is not None else None
    existing = read_text(absolute_output) if absolute_output.exists() else None
    source_state = _read_source_state(absolute_state) if absolute_state is not None else {}
    text = build_entity_ftl(entries, existing, source_state)
    state_text = _build_source_state_text(entries)
    changed = write_text_if_changed(absolute_output, text, dry_run=dry_run)

    if absolute_state is not None:
        changed = write_text_if_changed(absolute_state, state_text, dry_run=dry_run) or changed

    return len(entries), changed


def _unique_entity_localizations(entries: list[EntityLocalization]) -> list[EntityLocalization]:
    result: list[EntityLocalization] = []
    seen: set[str] = set()

    for entry in entries:
        if entry.message_id in seen:
            continue

        seen.add(entry.message_id)
        result.append(entry)

    return result


def _render_source_entry(entry: EntityLocalization) -> str:
    return render_entity_message(
        entry.message_id,
        entry.name,
        entry.description,
        entry.suffix,
    )


def _build_source_state_text(entries: list[EntityLocalization]) -> str:
    sources = {
        entry.message_id: _source_hash(_render_source_entry(entry))
        for entry in _unique_entity_localizations(entries)
    }
    return json.dumps({"version": 1, "sources": sources}, indent=2, sort_keys=True) + "\n"


def _read_source_state(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}

    try:
        data = json.loads(read_text(path))
    except ValueError:
        return {}

    sources = data.get("sources") if isinstance(data, dict) else None
    if not isinstance(sources, dict):
        return {}

    return {key: value for key, value in sources.items() if isinstance(key, str) and isinstance(value, str)}


def _source_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_entity_localization(repo_root: Path, path: Path, item: dict[str, str]) -> EntityLocalization | None:
    if item.get("type") != "entity":
        return None

    prototype_id = item.get("id")
    if prototype_id is None:
        return None

    name = item.get("name")
    description = item.get("description")
    suffix = item.get("suffix")

    if name is None and description is None and suffix is None:
        return None

    localization_id = item.get("localizationId") or f"ent-{prototype_id}"
    return EntityLocalization(
        source_path=path.relative_to(repo_root),
        prototype_id=prototype_id,
        message_id=localization_id,
        name=name,
        description=description,
        suffix=suffix,
    )


def _yaml_parser() -> Any:
    yaml_module = import_or_install("ruamel.yaml", "ruamel.yaml>=0.18,<1")
    yaml = yaml_module.YAML(typ="rt")
    yaml.allow_duplicate_keys = True
    return yaml


def _read_yaml_items(yaml: Any, path: Path) -> list[dict[str, str]]:
    text = read_text(path)
    try:
        data = yaml.load(text)
    except Exception:
        return _iter_top_level_yaml_items(text)

    if not isinstance(data, list):
        return []

    return [_read_yaml_mapping(item) for item in data if isinstance(item, dict)]


def _read_yaml_mapping(item: dict[Any, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for key in ("type", "id", "name", "description", "suffix", "localizationId"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            result[key] = value.strip()

    return result


FIELD_RE = re.compile(r"^(?P<indent>\s*)(?P<key>type|id|name|description|suffix|localizationId):(?:\s*(?P<value>.*))?$")
FIRST_FIELD_RE = re.compile(r"^-\s+(?P<key>type|id|name|description|suffix|localizationId):(?:\s*(?P<value>.*))?$")


def _iter_top_level_yaml_items(text: str) -> list[dict[str, str]]:
    lines = text.replace("\r\n", "\n").splitlines()
    items: list[list[str]] = []
    current: list[str] = []

    for line in lines:
        if line.startswith("- "):
            if current:
                items.append(current)
            current = [line]
            continue

        if current:
            current.append(line)

    if current:
        items.append(current)

    return [_parse_yaml_item(item) for item in items]


def _parse_yaml_item(lines: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    index = 0

    while index < len(lines):
        line = lines[index]
        first = FIRST_FIELD_RE.match(line)
        field = first or FIELD_RE.match(line)
        if field is None:
            index += 1
            continue

        key = field.group("key")
        value = (field.group("value") or "").rstrip()
        indent = len(field.groupdict().get("indent") or "  ")

        if value in {"|", "|-", "|+", ">", ">-", ">+"}:
            block, index = _read_block_scalar(lines, index + 1, indent + 2, folded=value.startswith(">"))
            result[key] = block.strip("\n")
            continue

        parsed = _parse_scalar(value)
        if parsed is not None:
            result[key] = parsed

        index += 1

    return result


def _read_block_scalar(lines: list[str], start: int, expected_indent: int, folded: bool) -> tuple[str, int]:
    block: list[str] = []
    index = start

    while index < len(lines):
        line = lines[index]
        if line.strip() and len(line) - len(line.lstrip(" ")) < expected_indent:
            break

        block.append(line[expected_indent:] if len(line) >= expected_indent else "")
        index += 1

    separator = " " if folded else "\n"
    return separator.join(block), index


def _parse_scalar(value: str) -> str | None:
    stripped = _strip_inline_comment(value).strip()
    if not stripped or stripped in {"~", "null", "Null", "NULL"}:
        return None

    if stripped[0] == stripped[-1:] and stripped[0] in {"'", '"'}:
        return _unquote(stripped)

    if stripped.startswith(("[", "{", "!", "&", "*")):
        return None

    return stripped


def _strip_inline_comment(value: str) -> str:
    quote: str | None = None
    escaped = False

    for index, char in enumerate(value):
        if escaped:
            escaped = False
            continue

        if char == "\\" and quote == '"':
            escaped = True
            continue

        if char in {"'", '"'}:
            quote = None if quote == char else char if quote is None else quote
            continue

        if char == "#" and quote is None and (index == 0 or value[index - 1].isspace()):
            return value[:index]

    return value


def _unquote(value: str) -> str:
    inner = value[1:-1]
    if value.startswith('"'):
        try:
            decoded = bytes(inner, "utf-8").decode("unicode_escape")
            decoded.encode("utf-8")
            return decoded
        except (UnicodeDecodeError, UnicodeEncodeError, ValueError):
            return inner

    return inner.replace("''", "'")
