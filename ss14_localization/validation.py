from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .filesystem import iter_files, read_text
from .fluent import (FluentSyntaxError, assert_structure, entries, parse_resource, rich_tags,
                     serialize_entry, visible_parts)


@dataclass(frozen=True)
class ValidationFinding:
    level: str
    path: Path
    message_id: str
    text: str


@dataclass
class ValidationReport:
    findings: list[ValidationFinding] = field(default_factory=list)
    checked_messages: int = 0
    missing_messages: int = 0
    untranslated_messages: int = 0

    @property
    def has_errors(self) -> bool:
        return any(finding.level == "error" for finding in self.findings)

    def add(self, level: str, path: Path, message_id: str, text: str) -> None:
        self.findings.append(ValidationFinding(level, path, message_id, text))


def validate_locale(source_root: Path, target_root: Path, checker=None) -> ValidationReport:
    report = ValidationReport()
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    target_owners = {}
    target_resources = {}
    for path in iter_files(target_root, ".ftl"):
        try:
            resource = parse_resource(read_text(path))
            target_resources[path] = resource
            for key in entries(resource):
                if key in target_owners:
                    report.add("error", path.relative_to(target_root), key, f"duplicate key in {target_owners[key]}")
                target_owners[key] = path
        except ValueError as error:
            report.add("error", path.relative_to(target_root), "", str(error))

    for source_path in iter_files(source_root, ".ftl"):
        relative = source_path.relative_to(source_root)
        target_path = target_root / relative
        try:
            source_messages = entries(parse_resource(read_text(source_path)))
        except ValueError as error:
            report.add("error", relative, "", str(error))
            continue
        target_messages = entries(target_resources[target_path]) if target_path in target_resources else {}
        actual_order = [key for key in target_messages if key in source_messages]
        expected_order = [key for key in source_messages if key in target_messages]
        if actual_order != expected_order:
            report.add("error", relative, "", "target key order differs from source")

        for message_id, source_message in source_messages.items():
            report.checked_messages += 1
            target_message = target_messages.get(message_id)
            if target_message is None:
                report.missing_messages += 1
                elsewhere = target_owners.get(message_id)
                report.add("error", relative, message_id, f"missing target message; existing location={elsewhere}")
                continue

            if checker and checker.needs_translation(target_message) and target_path.name not in checker.pass_list.ignored_files:
                report.untranslated_messages += 1
                report.add("error", relative, message_id, "target language ratio is below threshold")

            source_copy, target_copy = source_message.clone(), target_message.clone()
            source_copy.comment = target_copy.comment = None
            try:
                assert_structure(source_copy, target_copy)
            except FluentSyntaxError as error:
                report.add("error", relative, message_id, str(error))
            source_text = "\n".join(visible_parts(source_message))
            target_text = "\n".join(visible_parts(target_message))
            source_tags = rich_tags(source_text)
            target_tags = rich_tags(target_text)
            if source_tags != target_tags:
                report.add(
                    "error",
                    relative,
                    message_id,
                    f"rich-text tag mismatch: source={dict(source_tags)} target={dict(target_tags)}",
                )

            if checker:
                try:
                    checker.pass_list.assert_preserved(source_text, target_text)
                except ValueError as error:
                    report.add("error", relative, message_id, str(error))

    return report
