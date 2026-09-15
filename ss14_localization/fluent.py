from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from functools import lru_cache
from dataclasses import dataclass
import re

from .constants import ZERO_WIDTH_SPACE
from .dependencies import import_or_install


class FluentSyntaxError(ValueError):
    pass


@lru_cache(maxsize=1)
def syntax():
    return import_or_install("fluent.syntax", "fluent.syntax>=0.19,<1")


def parse_resource(text: str):
    module = syntax()
    resource = module.FluentParser(with_spans=True).parse(text.replace("\r\n", "\n"))
    seen = set()
    for entry in resource.body:
        if isinstance(entry, module.ast.Junk):
            details = "; ".join(f"{a.code}: {a.message}" for a in entry.annotations)
            line = text[:entry.span.start].count("\n") + 1
            raise FluentSyntaxError(f"FTL, строка {line}: {details}")
        if isinstance(entry, (module.ast.Message, module.ast.Term)):
            key = entry_id(entry)
            if key in seen:
                raise FluentSyntaxError(f"Повторяющийся ключ FTL: {key}")
            seen.add(key)
            attributes_seen = set()
            for attribute in entry.attributes:
                if attribute.id.name in attributes_seen:
                    raise FluentSyntaxError(f"Повторяющийся атрибут {key}.{attribute.id.name}")
                attributes_seen.add(attribute.id.name)
    return resource


def entry_id(entry) -> str:
    prefix = "-" if isinstance(entry, syntax().ast.Term) else ""
    return prefix + entry.id.name


def entries(resource) -> dict:
    ast = syntax().ast
    return {entry_id(node): node for node in resource.body if isinstance(node, (ast.Message, ast.Term))}


def serialize_resource(resource) -> str:
    return syntax().FluentSerializer().serialize(resource)


def serialize_entry(entry) -> str:
    return serialize_resource(syntax().ast.Resource([entry]))


def visible_parts(entry) -> list[str]:
    return [pattern_text(pattern) for pattern in ([entry.value] if entry.value else []) +
            [attribute.value for attribute in entry.attributes]]


def pattern_text(node) -> str:
    ast = syntax().ast
    if isinstance(node, ast.StringLiteral):
        return node.parse()["value"]
    if isinstance(node, ast.TextElement):
        return node.value
    if isinstance(node, ast.Pattern):
        return "".join(pattern_text(element) for element in node.elements)
    if isinstance(node, ast.Placeable):
        return pattern_text(node.expression)
    if isinstance(node, ast.SelectExpression):
        return "\n".join(pattern_text(variant.value) for variant in node.variants)
    return " "


def structural_signature(node, technical: bool = False):
    """Keep syntax, references, arguments, variants and comments; mask visible prose only."""
    ast = syntax().ast
    if isinstance(node, ast.TextElement):
        return ("TextElement",)
    if isinstance(node, ast.StringLiteral) and not technical:
        value = node.parse()["value"]
        return ("StringLiteral", "<text>" if any(character.isalpha() for character in value) else value)
    if isinstance(node, list):
        return [structural_signature(value, technical) for value in node]
    if isinstance(node, ast.BaseNode):
        return (type(node).__name__, {key: structural_signature(value, technical or key == "arguments")
                for key, value in vars(node).items() if key != "span"})
    return node


def assert_structure(source, target) -> None:
    if structural_signature(source) != structural_signature(target):
        raise FluentSyntaxError("ИИ изменил структуру FTL, ключи, атрибуты, ссылки, аргументы или комментарии")


MESSAGE_START_RE = re.compile(r"^(?P<id>-?[A-Za-z][A-Za-z0-9_-]*)\s*=")
VARIABLE_RE = re.compile(r"\{\s*\$([A-Za-z][A-Za-z0-9_-]*)")
ATTRIBUTE_RE = re.compile(r"^\s+\.([A-Za-z][A-Za-z0-9_-]*)\s*=", re.MULTILINE)
FUNCTION_RE = re.compile(r"\{\s*([A-Z][A-Z0-9_-]*)\s*\(")
RICH_TAG_RE = re.compile(r"(?<!\\)\[(\/?)([A-Za-z][A-Za-z0-9_-]*)(?:[^\]]*)\]")
RICH_TAG_NAMES = {
    "bold",
    "bolditalic",
    "bullet",
    "center",
    "cmdlink",
    "color",
    "emoji",
    "font",
    "head",
    "italic",
    "keybind",
    "mono",
    "protodata",
    "scramble",
    "textlink",
}


@dataclass(frozen=True)
class FluentMessage:
    id: str
    start: int
    end: int
    lines: tuple[str, ...]

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def parse_messages(text: str) -> list[FluentMessage]:
    text = text.replace("\r\n", "\n")
    resource = parse_resource(text)
    result = []
    for key, node in entries(resource).items():
        start = text[:node.span.start].count("\n")
        end = text[:node.span.end].count("\n") + 1
        result.append(FluentMessage(key, start, end, tuple(text[node.span.start:node.span.end].splitlines())))
    return result


def message_map(text: str) -> dict[str, FluentMessage]:
    return {message.id: message for message in parse_messages(text)}


def normalize_fluent_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n")
    lines = normalized.split("\n")
    if normalized.endswith("\n"):
        lines = lines[:-1]

    lines = _trim_outer_blank_lines(lines)
    if not any(line.strip() for line in lines):
        return ""

    result = "\n".join(escape_leading_multiline_markup_lines(lines)) + "\n"
    parse_resource(result)
    return result


def _trim_outer_blank_lines(lines: list[str]) -> list[str]:
    start = 0
    while start < len(lines) and not lines[start].strip():
        start += 1

    end = len(lines)
    while end > start and not lines[end - 1].strip():
        end -= 1

    leading = [""] if start > 0 else []
    trailing = [""] if end < len(lines) else []
    return leading + lines[start:end] + trailing


def escape_leading_multiline_markup_lines(lines: list[str]) -> list[str]:
    output = list(lines)
    in_multiline_pattern = False

    for index, line in enumerate(output):
        if _is_pattern_assignment(line):
            in_multiline_pattern = True
            continue

        if not in_multiline_pattern:
            continue

        if not line.strip():
            continue

        output[index] = escape_leading_markup_line(line)

    return output


def escape_leading_markup_line(line: str) -> str:
    stripped = line.lstrip()
    if stripped.startswith(ZERO_WIDTH_SPACE):
        return line

    match = RICH_TAG_RE.match(stripped)
    if match is None or match.group(2).lower() not in RICH_TAG_NAMES:
        return line

    leading_len = len(line) - len(stripped)
    return line[:leading_len] + ZERO_WIDTH_SPACE + stripped


def _is_pattern_assignment(line: str) -> bool:
    return MESSAGE_START_RE.match(line) is not None or ATTRIBUTE_RE.match(line) is not None


def variables(text: str) -> set[str]:
    return set(VARIABLE_RE.findall(text))


def attributes(text: str) -> set[str]:
    return set(ATTRIBUTE_RE.findall(text))


def functions(text: str) -> set[str]:
    return set(FUNCTION_RE.findall(text))


def rich_tags(text: str) -> Counter[str]:
    return Counter(match.group(0) for match in RICH_TAG_RE.finditer(text)
                   if match.group(2).lower() in RICH_TAG_NAMES)


def strip_rich_tags(text: str) -> str:
    return RICH_TAG_RE.sub(" ", text)


def same_message_payload(left: FluentMessage, right: FluentMessage) -> bool:
    return _canonical_message(left.text) == _canonical_message(right.text)


def _canonical_message(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def render_pattern(prefix: str, value: str) -> list[str]:
    prepared = escape_leading_multiline_markup(value)
    lines = prepared.splitlines() or [""]
    if len(lines) == 1:
        return [f"{prefix} {lines[0]}".rstrip()]

    return [prefix] + [f"    {line}" for line in lines]


def render_entity_message(message_id: str, name: str | None, description: str | None, suffix: str | None) -> str:
    lines: list[str] = []
    lines.extend(render_pattern(f"{message_id} =", name or ""))

    if description:
        lines.extend(render_pattern("  .desc =", description))

    if suffix:
        lines.extend(render_pattern("  .suffix =", suffix))

    return "\n".join(lines)


def escape_leading_multiline_markup(value: str) -> str:
    if "\n" not in value:
        return value

    return "\n".join(escape_leading_markup_line(line) for line in value.split("\n"))


def normalize_entity_message_style(text: str) -> str:
    parsed = parse_messages(text)
    if len(parsed) != 1 or not parsed[0].id.startswith("ent-"):
        return text

    lines = list(parsed[0].lines)
    if lines:
        lines[0] = _replace_assignment_value(lines[0], _entity_name)

    for index, line in enumerate(lines):
        if re.match(r"^\s+\.(?:desc|suffix)\s*=", line):
            lines[index] = _replace_assignment_value(line, _capitalize_value)

    return "\n".join(lines)


def _replace_assignment_value(line: str, transform: Callable[[str], str]) -> str:
    if "=" not in line:
        return line

    prefix, value = line.split("=", 1)
    separator = " " if value.startswith(" ") else ""
    stripped = value[1:] if value.startswith(" ") else value
    return f"{prefix}={separator}{transform(stripped)}".rstrip()


def _entity_name(value: str) -> str:
    if _starts_with_fluent_syntax(value):
        return value

    return value.lower()


def _capitalize_value(value: str) -> str:
    if _starts_with_fluent_syntax(value):
        return value

    for index, char in enumerate(value):
        if char.isalpha():
            return value[:index] + char.upper() + value[index + 1:]

    return value


def _starts_with_fluent_syntax(value: str) -> bool:
    return value.lstrip().startswith(("{", "["))
