from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .constants import (
    DEFAULT_LOCALE_ROOT,
    DEFAULT_PROTOTYPE_OUTPUT,
    DEFAULT_PROTOTYPE_STATE,
    DEFAULT_PROTOTYPES_ROOT,
    DEFAULT_SOURCE_CULTURE,
    DEFAULT_TARGET_CULTURE,
)
from .filesystem import (
    iter_files,
    read_text,
    remove_empty_files_and_dirs,
    write_text_if_changed,
)
from .fluent import normalize_fluent_text
from .paths import find_repo_root, resolve_tool_file
from .prototypes import write_entity_ftl
from .strings import (
    prepare_target_files,
    sync_locale_strings,
    write_missing_messages_for_file,
)
from .validation import validate_locale


def main(argv: list[str] | None = None) -> int:
    if not (sys.argv[1:] if argv is None else argv):
        from .tui import run

        return run()
    parser = argparse.ArgumentParser(prog="ss14-loc")
    parser.add_argument("--env-file", type=Path, help="файл окружения вместо встроенного .env")
    parser.add_argument(
        "--repo-root",
        type=Path,
        help="корень репозитория Space Station 14; обычно определяется автоматически",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    show_root = subparsers.add_parser("repo-root", help="показать найденный корень репозитория")
    show_root.set_defaults(func=_show_repo_root)

    normalize = subparsers.add_parser("normalize")
    normalize.add_argument("--culture", default=DEFAULT_TARGET_CULTURE)
    normalize.add_argument("--locale-root", type=Path, default=DEFAULT_LOCALE_ROOT)
    normalize.add_argument("--dry-run", action="store_true")
    normalize.set_defaults(func=_normalize)

    extract = subparsers.add_parser("extract-prototypes")
    extract.add_argument("--culture", default=DEFAULT_TARGET_CULTURE)
    extract.add_argument("--locale-root", type=Path, default=DEFAULT_LOCALE_ROOT)
    extract.add_argument("--prototypes-root", type=Path, default=DEFAULT_PROTOTYPES_ROOT)
    extract.add_argument("--output", type=Path, default=DEFAULT_PROTOTYPE_OUTPUT)
    extract.add_argument("--state-output", type=Path, default=DEFAULT_PROTOTYPE_STATE)
    extract.add_argument("--dry-run", action="store_true")
    extract.set_defaults(func=_extract_prototypes)

    sync = subparsers.add_parser("sync-strings")
    sync.add_argument("--source-culture", default=DEFAULT_SOURCE_CULTURE)
    sync.add_argument("--target-culture", default=DEFAULT_TARGET_CULTURE)
    sync.add_argument("--locale-root", type=Path, default=DEFAULT_LOCALE_ROOT)
    sync.add_argument("--bidirectional", action="store_true")
    sync.add_argument("--dry-run", action="store_true")
    sync.set_defaults(func=_sync_strings)

    prepare = subparsers.add_parser("prepare-target-file")
    prepare.add_argument("--source-file", type=Path, required=True)
    prepare.add_argument("--target-file", type=Path, required=True)
    prepare.add_argument("--target-locale-root", type=Path, required=True)
    prepare.add_argument("--dry-run", action="store_true")
    prepare.set_defaults(func=_prepare_target_file)

    prepare_many = subparsers.add_parser("prepare-target-files")
    prepare_many.add_argument("--source-culture-root", type=Path, required=True)
    prepare_many.add_argument("--target-culture-root", type=Path, required=True)
    prepare_many.add_argument("--relative-root", type=Path, action="append", required=True)
    prepare_many.add_argument("--report-json", type=Path, required=True)
    prepare_many.add_argument("--dry-run", action="store_true")
    prepare_many.set_defaults(func=_prepare_target_files)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--source-culture", default=DEFAULT_SOURCE_CULTURE)
    validate.add_argument("--target-culture", default=DEFAULT_TARGET_CULTURE)
    validate.add_argument("--locale-root", type=Path, default=DEFAULT_LOCALE_ROOT)
    validate.add_argument("--fail-on-errors", action="store_true")
    validate.add_argument("--source-root", type=Path, default=_env_path("TRANSLATE_SOURCE_ROOT"))
    validate.add_argument("--target-root", type=Path, default=_env_path("TRANSLATE_TARGET_ROOT"))
    validate.add_argument("--pass-list", type=Path, default=_env_path("TRANSLATE_PASS_LIST"))
    validate.add_argument(
        "--language-ratio", type=float, default=_env_float("TRANSLATE_LANGUAGE_RATIO")
    )
    validate.add_argument(
        "--language-profile", type=Path, default=_env_path("TRANSLATE_LANGUAGE_PROFILE")
    )
    validate.set_defaults(func=_validate)

    translate = subparsers.add_parser("translate")
    translate.add_argument("files", nargs="+", type=Path)
    _translation_options(translate)
    translate.set_defaults(func=_translate)

    translate_all = subparsers.add_parser("translate-all")
    _translation_options(translate_all)
    translate_all.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("TRANSLATE_BATCH_SIZE", "0")),
    )
    translate_all.set_defaults(func=_translate_all)

    args = parser.parse_args(argv)
    try:
        args.repo_root = find_repo_root(args.repo_root)
    except RuntimeError as error:
        parser.error(str(error))
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


def _env_path(name):
    return Path(os.environ[name]) if os.environ.get(name) else None


def _env_float(name):
    return float(os.environ[name]) if os.environ.get(name) else None


def _translation_options(parser):
    parser.add_argument("--env-file", type=Path, default=argparse.SUPPRESS)
    parser.add_argument("--repo-root", type=Path, default=argparse.SUPPRESS)
    parser.add_argument("--source-culture", default=DEFAULT_SOURCE_CULTURE)
    parser.add_argument("--target-culture", default=DEFAULT_TARGET_CULTURE)
    parser.add_argument("--locale-root", type=Path, default=DEFAULT_LOCALE_ROOT)
    parser.add_argument("--source-root", type=Path, default=_env_path("TRANSLATE_SOURCE_ROOT"))
    parser.add_argument("--target-root", type=Path, default=_env_path("TRANSLATE_TARGET_ROOT"))
    parser.add_argument("--prompt", type=Path, default=_env_path("TRANSLATE_PROMPT"))
    parser.add_argument("--glossary", type=Path, default=_env_path("TRANSLATE_GLOSSARY"))
    parser.add_argument("--pass-list", type=Path, default=_env_path("TRANSLATE_PASS_LIST"))
    parser.add_argument(
        "--language-ratio", type=float, default=_env_float("TRANSLATE_LANGUAGE_RATIO")
    )
    parser.add_argument(
        "--language-profile", type=Path, default=_env_path("TRANSLATE_LANGUAGE_PROFILE")
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=int(os.environ.get("TRANSLATE_CHUNK_SIZE", "0")),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=int(os.environ.get("TRANSLATE_CONCURRENCY", "2")),
    )
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument(
        "--save-tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="не отправлять готовые переводы как примеры (по умолчанию включено)",
    )
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--dry-run", action="store_true")


def _locale_roots(args):
    root = args.repo_root / args.locale_root
    source = (
        args.repo_root / args.source_root if args.source_root else root / args.source_culture
    ).resolve()
    target = (
        args.repo_root / args.target_root if args.target_root else root / args.target_culture
    ).resolve()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError("Исходная и целевая локали должны быть разными непересекающимися папками")
    return source, target


def _show_repo_root(args: argparse.Namespace) -> int:
    print(args.repo_root)
    return 0


def _normalize(args: argparse.Namespace) -> int:
    root = args.repo_root / args.locale_root / args.culture
    changed = 0

    for path in iter_files(root, ".ftl"):
        normalized = normalize_fluent_text(read_text(path))
        if normalized:
            changed += 1 if write_text_if_changed(path, normalized, dry_run=args.dry_run) else 0
        else:
            changed += 1
            if not args.dry_run:
                path.unlink()

    removed_files, removed_dirs = remove_empty_files_and_dirs(root, dry_run=args.dry_run)
    print(f"normalized={changed} removed_files={removed_files} removed_dirs={removed_dirs}")
    return 0


def _extract_prototypes(args: argparse.Namespace) -> int:
    output = args.locale_root / args.culture / args.output
    state = args.locale_root / args.culture / args.state_output
    count, changed = write_entity_ftl(
        args.repo_root,
        args.prototypes_root,
        output,
        state_path=state,
        dry_run=args.dry_run,
    )
    print(f"entity_messages={count} changed={changed}")
    return 0


def _sync_strings(args: argparse.Namespace) -> int:
    locale_root = args.repo_root / args.locale_root
    result = sync_locale_strings(
        locale_root / args.source_culture,
        locale_root / args.target_culture,
        dry_run=args.dry_run,
    )

    if args.bidirectional:
        reverse = sync_locale_strings(
            locale_root / args.target_culture,
            locale_root / args.source_culture,
            dry_run=args.dry_run,
        )
        result = result + reverse

    print(
        f"scanned_files={result.scanned_files} changed_files={result.changed_files} "
        f"added_messages={result.added_messages}"
    )
    return 0


def _prepare_target_file(args: argparse.Namespace) -> int:
    added, changed = write_missing_messages_for_file(
        args.repo_root / args.source_file,
        args.repo_root / args.target_file,
        args.repo_root / args.target_locale_root,
        dry_run=args.dry_run,
    )
    print(f"added_messages={added} changed={changed}")
    return 0


def _prepare_target_files(args: argparse.Namespace) -> int:
    result = prepare_target_files(
        args.repo_root / args.source_culture_root,
        args.repo_root / args.target_culture_root,
        args.relative_root,
        dry_run=args.dry_run,
    )
    report = {
        "target_files": [str(path) for path in result.target_files],
        "prepared_files": result.prepared_files,
        "dry_run_missing_files": result.dry_run_missing_files,
    }
    report_path = args.repo_root / args.report_json
    if args.dry_run:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"target_files={len(result.target_files)} prepared_files={result.prepared_files} "
        f"dry_run_missing_files={result.dry_run_missing_files}"
    )
    return 0


def _validate(args: argparse.Namespace) -> int:
    from .language import LanguageChecker, load_pass_list

    source_root, target_root = _locale_roots(args)
    pass_path = (
        resolve_tool_file(args.pass_list, Path("pass_list.yml")) if args.pass_list else None
    )
    profile = resolve_tool_file(args.language_profile, Path("")) if args.language_profile else None
    checker = LanguageChecker(
        args.source_culture,
        args.target_culture,
        load_pass_list(args.repo_root, pass_path),
        args.language_ratio,
        profile,
    )
    report = validate_locale(
        source_root,
        target_root,
        checker=checker,
    )

    print(
        f"checked={report.checked_messages} missing={report.missing_messages} "
        f"untranslated={report.untranslated_messages} findings={len(report.findings)}"
    )

    for finding in report.findings[:200]:
        print(f"{finding.level}: {finding.path}:{finding.message_id}: {finding.text}")

    if len(report.findings) > 200:
        print(f"... {len(report.findings) - 200} more findings omitted")

    return 1 if args.fail_on_errors and report.has_errors else 0


def _translation_settings(args):
    from .ai import AiConfig
    from .budget import OutputBudget
    from .language import LanguageChecker, load_pass_list
    from .paths import TOOL_ROOT
    from .translate import build_translation_prompt

    if args.chunk_size < 0 or args.concurrency < 1 or getattr(args, "batch_size", 1) < 0:
        raise ValueError(
            "Размеры блока/группы должны быть неотрицательными, "
            "число параллельных запросов — положительным"
        )
    pass_path = (
        resolve_tool_file(args.pass_list, Path("pass_list.yml")) if args.pass_list else None
    )
    pass_list = load_pass_list(args.repo_root, pass_path)
    profile = resolve_tool_file(args.language_profile, Path("")) if args.language_profile else None
    checker = LanguageChecker(
        args.source_culture,
        args.target_culture,
        pass_list,
        args.language_ratio,
        profile,
    )
    budget = OutputBudget.from_env()
    default_prompt = Path("prompts") / f"{args.target_culture}.md"
    if not (TOOL_ROOT / default_prompt).is_file():
        default_prompt = Path("prompts/default.md")
    prompt_path = resolve_tool_file(args.prompt, default_prompt)
    glossary_path = resolve_tool_file(args.glossary, Path("glossary.md"))
    prompt = build_translation_prompt(
        prompt_path, glossary_path, args.source_culture, args.target_culture, pass_list
    )
    if not args.dry_run:
        AiConfig.from_env()
    return checker, budget, prompt


def _run_translation(args, files, settings, texts=None):
    from .translate import run_translate_files

    checker, budget, prompt = settings
    result = run_translate_files(
        files,
        prompt,
        args.chunk_size,
        target_culture=args.target_culture,
        concurrency=args.concurrency,
        allow_partial=args.allow_partial,
        dry_run=args.dry_run,
        checker=checker,
        budget=budget,
        texts=texts,
        save_tokens=args.save_tokens,
    )
    print(
        f"translated_messages={result.translated_messages} changed_files={result.changed_files} "
        f"failed_files={len(result.failed_files)}"
    )
    for failed in result.failed_details:
        print(f"Ошибка: {failed.path}: {failed.error}")
    return result


def _report(args, result, prepared=None):
    report = {
        "dry_run": args.dry_run,
        "translated_messages": result.translated_messages,
        "changed_files": result.changed_files,
        "failed_files": [
            {
                "path": str(item.path),
                "translated_messages": item.translated_messages,
                "changed": item.changed,
                "error": item.error,
            }
            for item in result.failed_details
        ],
    }
    if prepared:
        report["prepared_paths"] = [str(path) for path in prepared.changed_paths]
        report["added_messages"] = prepared.added_messages
        report["moved_messages"] = prepared.moved_messages
    if args.report_json:
        if args.dry_run:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            path = args.repo_root / args.report_json
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _translate(args):
    settings = _translation_settings(args)
    files = [(args.repo_root / path).resolve() for path in args.files]
    _, target_root = _locale_roots(args)
    for path in files:
        if target_root not in path.parents:
            raise ValueError(f"Файл вне целевой локали: {path}. Укажите правильный --target-root.")
    result = _run_translation(args, files, settings)
    _report(args, result)
    return 1 if result.failed_files else 0


def _translate_all(args):
    from .translate import TranslationRunResult

    settings = _translation_settings(args)
    source_root, target_root = _locale_roots(args)
    prepared = prepare_target_files(source_root, target_root, [Path(".")], args.dry_run)
    print(
        f"target_files={len(prepared.target_files)} prepared_files={prepared.prepared_files} "
        f"added_messages={prepared.added_messages} moved_messages={prepared.moved_messages}"
    )
    files = list(prepared.target_files)
    if not files:
        print("Нет файлов для перевода.")
    translated = changed = 0
    failures = []
    batch_size = args.batch_size or len(files) or 1
    for start in range(0, len(files), batch_size):
        batch = files[start : start + batch_size]
        print(f"Перевод файлов {start + 1}-{start + len(batch)} из {len(files)}...")
        result = _run_translation(
            args, batch, settings, prepared.planned_texts if args.dry_run else None
        )
        translated += result.translated_messages
        changed += result.changed_files
        failures.extend(result.failed_details)
        if result.failed_files and not args.allow_partial:
            break
    result = TranslationRunResult(
        translated, changed, tuple(item.path for item in failures), tuple(failures)
    )
    _report(args, result, prepared)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
