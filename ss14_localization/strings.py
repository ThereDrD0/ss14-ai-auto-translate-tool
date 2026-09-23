from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .filesystem import iter_files, read_text, write_text_if_changed
from .fluent import (
    FluentSyntaxError,
    entries,
    entry_id,
    normalize_resource_commas,
    parse_resource,
    serialize_resource,
    syntax,
)


@dataclass(frozen=True)
class SyncResult:
    scanned_files: int
    changed_files: int
    added_messages: int

    def __add__(self, other):
        return SyncResult(
            self.scanned_files + other.scanned_files,
            self.changed_files + other.changed_files,
            self.added_messages + other.added_messages,
        )


@dataclass
class PrepareTargetFilesResult:
    target_files: tuple[Path, ...]
    prepared_files: int
    dry_run_missing_files: int = 0
    added_messages: int = 0
    moved_messages: int = 0
    planned_texts: dict[Path, str] = field(default_factory=dict)
    changed_paths: tuple[Path, ...] = ()


def _merged_entry(source, translated):
    result = source.clone()
    if translated is None:
        return result
    if type(source) is not type(translated):
        raise ValueError(f"Конфликт типов ключа {entry_id(source)}")
    if translated.comment:
        result.comment = translated.comment.clone()
    if translated.value is not None:
        result.value = translated.value.clone()
    existing = {attribute.id.name: attribute for attribute in translated.attributes}
    result.attributes = [
        existing.get(attribute.id.name, attribute).clone() for attribute in source.attributes
    ]
    # Target-only attributes are not silently deleted.
    result.attributes.extend(
        attribute.clone()
        for attribute in translated.attributes
        if attribute.id.name not in {a.id.name for a in source.attributes}
    )
    return result


def _comment_groups(resource):
    ast = syntax().ast
    groups = {}
    pending = []
    for node in resource.body:
        if isinstance(node, ast.BaseComment):
            pending.append(node)
        elif isinstance(node, (ast.Message, ast.Term)):
            groups[entry_id(node)] = pending
            pending = []
    return groups, pending


def _read_resource(path, texts=None):
    try:
        text = read_text(path)
        if texts is not None:
            texts[path] = text
        return parse_resource(text)
    except FluentSyntaxError as error:
        raise FluentSyntaxError(f"{path}: {error}") from error


def _prepare(pairs: dict[Path, Path], target_root: Path, dry_run: bool, on_event=None):
    ast = syntax().ast
    sources = {}
    source_texts = {}
    source_layouts = {}
    owners = {}
    source_origins = {}
    for source_path, target_path in sorted(pairs.items()):
        if on_event:
            on_event("started", source_path, len(sources), len(pairs) + 1)
        resource = _read_resource(source_path, source_texts)
        sources[target_path] = resource
        source_layouts[target_path] = source_texts[source_path]
        if on_event:
            on_event("completed", source_path, len(sources), len(pairs) + 1)
        for key, node in entries(resource).items():
            if key in owners:
                raise ValueError(
                    f"Ключ {key} повторяется в исходной локали: "
                    f"{source_origins[key]} и {source_path}"
                )
            owners[key] = target_path
            source_origins[key] = source_path

    target_paths = set(iter_files(target_root, ".ftl")) | set(pairs.values())
    target_texts = {}
    targets = {
        path: _read_resource(path, target_texts) for path in sorted(target_paths) if path.exists()
    }
    target_comments = {path: _comment_groups(resource) for path, resource in targets.items()}
    known = {}
    origins = {}
    for path, resource in targets.items():
        for key, node in entries(resource).items():
            if key in known and not node.equals(known[key], ignored_fields=["span", "comment"]):
                raise ValueError(
                    f"Разные переводы ключа {key}: {origins[key]} и {path}. "
                    "Разрешите конфликт перед подготовкой."
                )
            if key not in known or owners.get(key) == path:
                known[key], origins[key] = node, path

    plans = {}
    added = moved = 0
    for path, source_resource in sources.items():
        source_comments, source_trailing = _comment_groups(source_resource)
        resource = source_resource.clone()
        body = []
        for node in resource.body:
            if isinstance(node, ast.BaseComment):
                continue
            key = entry_id(node)
            target_group = target_comments[origins[key]][0][key] if key in known else []
            translated_comment = key in known and (known[key].comment or target_group)
            comments = target_group if translated_comment else source_comments[key]
            body.extend(comment.clone() for comment in comments)
            merged = _merged_entry(node, known.get(key))
            if translated_comment and not known[key].comment:
                merged.comment = None
            body.append(merged)
            added += key not in known
            moved += key in origins and origins[key] != path
        if path in targets:
            for node in targets[path].body:
                if isinstance(node, (ast.Message, ast.Term)) and entry_id(node) not in owners:
                    body.extend(
                        comment.clone() for comment in target_comments[path][0][entry_id(node)]
                    )
                    body.append(node.clone())
            body.extend(comment.clone() for comment in target_comments[path][1])
        else:
            body.extend(comment.clone() for comment in source_trailing)
        resource.body = body
        normalize_resource_commas(resource)
        layout = targets.get(path, source_resource)
        original_text = target_texts[path] if path in targets else source_layouts[path]
        plans[path] = serialize_resource(resource, original_text, layout)

    for path, resource in targets.items():
        if path in sources:
            continue
        kept = []
        comments, trailing = target_comments[path]
        for node in resource.body:
            if isinstance(node, (ast.Message, ast.Term)) and entry_id(node) not in owners:
                kept.extend(comment.clone() for comment in comments[entry_id(node)])
                kept.append(node.clone())
        kept.extend(comment.clone() for comment in trailing)
        plan = ast.Resource(kept)
        normalize_resource_commas(plan)
        plans[path] = serialize_resource(plan, target_texts[path], resource)

    # Validate the complete plan before touching any existing files.
    changed = []
    for path, text in plans.items():
        if target_texts.get(path, "") != text:
            parse_resource(text)
            changed.append(path)
            action = (
                "создать" if not path.exists() else ("удалить" if not text.strip() else "обновить")
            )
            if not on_event:
                print(f"Подготовка: {action} {path}")
    if not dry_run:
        for path in changed:
            text = plans[path]
            action = (
                "создать" if not path.exists() else ("удалить" if not text.strip() else "обновить")
            )
            if not text.strip():
                if path.exists():
                    path.unlink()
            else:
                write_text_if_changed(path, text)
            if on_event:
                on_event(action, path, len(pairs), len(pairs) + 1)
    if on_event:
        on_event("finished", target_root, len(pairs) + 1, len(pairs) + 1)
    return PrepareTargetFilesResult(
        tuple(sorted(path for path in sources if plans[path].strip())),
        len(changed),
        dry_run_missing_files=sum(not path.exists() for path in sources) if dry_run else 0,
        added_messages=added,
        moved_messages=moved,
        planned_texts=plans,
        changed_paths=tuple(sorted(changed)),
    )


def prepare_target_files(
    source_culture_root: Path,
    target_culture_root: Path,
    relative_roots: list[Path],
    dry_run: bool = False,
    on_event=None,
):
    source_culture_root = source_culture_root.resolve()
    target_culture_root = target_culture_root.resolve()
    if not source_culture_root.is_dir():
        raise FileNotFoundError(f"Не найдена исходная локаль: {source_culture_root}")
    if (
        source_culture_root == target_culture_root
        or source_culture_root in target_culture_root.parents
        or target_culture_root in source_culture_root.parents
    ):
        raise ValueError("Исходная и целевая локали должны быть разными непересекающимися папками")
    pairs = {}
    for relative in relative_roots:
        root = (source_culture_root / relative).resolve()
        if root != source_culture_root and source_culture_root not in root.parents:
            raise ValueError(f"Путь выходит за пределы исходной локали: {relative}")
        if not root.exists():
            raise FileNotFoundError(root)
        paths = [root] if root.is_file() else iter_files(root, ".ftl")
        for path in paths:
            if path.suffix == ".ftl":
                pairs[path] = target_culture_root / path.relative_to(source_culture_root)
    return _prepare(pairs, target_culture_root, dry_run, on_event)


def sync_locale_strings(source_root: Path, target_root: Path, dry_run: bool = False):
    result = prepare_target_files(source_root, target_root, [Path(".")], dry_run)
    return SyncResult(len(result.target_files), result.prepared_files, result.added_messages)


def write_missing_messages_for_file(
    source_path: Path,
    target_path: Path,
    target_locale_root: Path,
    dry_run: bool = False,
):
    if target_locale_root.resolve() not in target_path.resolve().parents:
        raise ValueError("Целевой файл должен находиться в указанной целевой локали")
    if target_locale_root.resolve() in source_path.resolve().parents:
        raise ValueError("Исходный файл не должен находиться в целевой локали")
    if target_path.resolve() == source_path.resolve():
        raise ValueError("Исходный и целевой файл совпадают")
    result = _prepare(
        {source_path.resolve(): target_path.resolve()},
        target_locale_root.resolve(),
        dry_run,
    )
    return result.added_messages, bool(result.prepared_files)
