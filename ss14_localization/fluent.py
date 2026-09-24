from __future__ import annotations

import importlib
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

from .constants import ZERO_WIDTH_SPACE
from .dependencies import import_or_install


class FluentSyntaxError(ValueError):
    pass


@lru_cache(maxsize=1)
def syntax():
    return import_or_install("fluent.syntax", "fluent.syntax==0.19.0")


@lru_cache(maxsize=1)
def _parser_class():
    """Adapt Mozilla's parser to the non-experimental Linguini 0.8 grammar used by SS14."""
    module = syntax()
    parser_module = importlib.import_module("fluent.syntax.parser")
    stream_module = importlib.import_module("fluent.syntax.stream")
    ast = module.ast

    class LinguiniCompatibleStream(module.FluentParserStream):
        def is_next_line_comment(self, level: int = -1) -> bool:
            if self.current_char != stream_module.EOL:
                return False

            offset = self.index + (
                2 if self.get(self.index) == "\r" and self.get(self.index + 1) == "\n" else 1
            )
            count = 0
            while self.get(offset + count) == "#":
                count += 1
            next_level = min(count, 3) - 1
            return next_level == level

    class LinguiniCompatibleParser(module.FluentParser):
        def parse(self, source: str):
            ps = LinguiniCompatibleStream(source)
            ps.skip_blank_block()
            body = []
            last_comment = None

            while ps.current_char:
                entry = self.get_entry_or_junk(ps)
                blank_lines = ps.skip_blank_block()
                if isinstance(entry, ast.Comment) and not blank_lines and ps.current_char:
                    last_comment = entry
                    continue
                if last_comment is not None:
                    if isinstance(entry, (ast.Message, ast.Term)):
                        entry.comment = last_comment
                        if self.with_spans:
                            entry.span.start = entry.comment.span.start
                    else:
                        body.append(last_comment)
                    last_comment = None
                body.append(entry)

            resource = ast.Resource(body)
            if self.with_spans:
                resource.add_span(0, ps.index)
            return resource

        def parse_entry(self, source: str):
            ps = LinguiniCompatibleStream(source)
            ps.skip_blank_block()
            while ps.current_char == "#":
                skipped = self.get_entry_or_junk(ps)
                if isinstance(skipped, ast.Junk):
                    return skipped
                ps.skip_blank_block()
            return self.get_entry_or_junk(ps)

        @parser_module.with_span
        def get_comment(self, ps):
            level = -1
            content = ""
            while True:
                current_level = -1
                while ps.current_char == "#" and current_level < (2 if level == -1 else level):
                    ps.next()
                    current_level += 1
                if level == -1:
                    level = current_level

                if ps.current_char != stream_module.EOL:
                    if ps.current_char == " ":
                        ps.next()
                    char = ps.take_char(lambda value: value != stream_module.EOL)
                    while char:
                        content += char
                        char = ps.take_char(lambda value: value != stream_module.EOL)

                if ps.is_next_line_comment(level):
                    content += stream_module.EOL
                    ps.next()
                else:
                    break

            comment_type = (ast.Comment, ast.GroupComment, ast.ResourceComment)[level]
            return comment_type(content)

        @parser_module.with_span
        def get_call_argument(self, ps):
            expression = self.get_inline_expression(ps)
            ps.skip_blank()
            if ps.current_char != ":":
                return expression
            if isinstance(expression, ast.MessageReference) and expression.attribute is None:
                ps.next()
                ps.skip_blank()
                return ast.NamedArgument(expression.id, self.get_inline_expression(ps))
            raise module.ParseError("E0009")

        def get_escape_sequence(self, ps):
            if ps.current_char == "{":
                ps.next()
                return r"\{"
            return super().get_escape_sequence(ps)

        def get_inline_expression(self, ps):
            expression = super().get_inline_expression(ps)
            if (
                isinstance(expression, ast.MessageReference)
                and expression.attribute is None
                and ps.current_peek == "."
            ):
                ps.skip_to_peek()
                ps.next()
                expression.attribute = self.get_identifier(ps)
                if self.with_spans:
                    expression.span.end = ps.index
            return expression

    return LinguiniCompatibleParser


def parse_resource(text: str):
    module = syntax()
    normalized = text.replace("\r\n", "\n")
    resource = _parser_class()(with_spans=True).parse(normalized)
    seen = set()
    for entry in resource.body:
        if isinstance(entry, module.ast.Junk):
            details = "; ".join(f"{a.code}: {a.message}" for a in entry.annotations)
            line = normalized[: entry.span.start].count("\n") + 1
            raise FluentSyntaxError(f"FTL, строка {line}: {details}")
        if isinstance(entry, (module.ast.Message, module.ast.Term)):
            key = entry_id(entry)
            if key in seen:
                raise FluentSyntaxError(f"Повторяющийся ключ FTL: {key}")
            seen.add(key)
    return resource


def entry_id(entry) -> str:
    prefix = "-" if isinstance(entry, syntax().ast.Term) else ""
    return prefix + entry.id.name


def entries(resource) -> dict:
    ast = syntax().ast
    return {
        entry_id(node): node for node in resource.body if isinstance(node, (ast.Message, ast.Term))
    }


def serialize_resource(resource, original_text: str | None = None, original_resource=None) -> str:
    text = syntax().FluentSerializer().serialize(resource)
    if not original_text or "\n\n" not in original_text:
        return text

    ast = syntax().ast
    original_resource = original_resource or parse_resource(original_text)
    blank_lines = {}
    previous_end = 0
    for node in original_resource.body:
        if isinstance(node, (ast.Message, ast.Term)):
            gap = original_text[previous_end : node.span.start]
            blank_lines[entry_id(node)] = max(0, gap.count("\n") - (previous_end != 0))
        previous_end = node.span.end

    insertions = []
    previous_end = 0
    for node in parse_resource(text).body:
        if isinstance(node, (ast.Message, ast.Term)):
            gap = text[previous_end : node.span.start]
            existing = max(0, gap.count("\n") - (previous_end != 0))
            missing = blank_lines.get(entry_id(node), 0) - existing
            if missing > 0:
                insertions.append((node.span.start, missing))
        previous_end = node.span.end
    for offset, count in reversed(insertions):
        text = text[:offset] + "\n" * count + text[offset:]
    return text


def serialize_entry(entry) -> str:
    return serialize_resource(syntax().ast.Resource([entry]))


def normalize_commas(text: str) -> str:
    def space_comma(match: re.Match[str]) -> str:
        before, after = match.string[match.start() - 1], match.string[match.end()]
        return match.group() if before.isdecimal() and after.isdecimal() else ", "

    parts = TECHNICAL_TEXT_RE.split(text)
    return "".join(
        part if index % 2 else COMMA_SPACING_RE.sub(space_comma, part)
        for index, part in enumerate(parts)
    )


def normalize_resource_commas(node) -> None:
    ast = syntax().ast
    if isinstance(node, ast.TextElement):
        node.value = normalize_commas(node.value)
    elif isinstance(node, ast.BaseNode):
        for key, value in vars(node).items():
            if key not in {"span", "comment", "arguments"}:
                normalize_resource_commas(value)
    elif isinstance(node, list):
        for value in node:
            normalize_resource_commas(value)


def _visible_text(value: str) -> str:
    return TECHNICAL_TEXT_RE.sub(lambda match: " " * len(match.group()), value)


def _capitalize_pattern(pattern, upper: bool) -> None:
    ast = syntax().ast
    for element in pattern.elements:
        if not isinstance(element, ast.TextElement):
            return  # Начальное выражение Fluent может задавать первую букву само.
        visible = _visible_text(element.value)
        for index, char in enumerate(visible):
            if char.isspace() or char in ZERO_WIDTH_SPACE + "\"'«“([{—-":
                continue
            if char.isalpha():
                if not upper:
                    word = re.match(r"[\w.-]+", visible[index:])
                    if word and len(word.group()) > 1 and word.group().isupper():
                        return
                changed = char.upper() if upper else char.lower()
                element.value = element.value[:index] + changed + element.value[index + 1 :]
            return


def _normalize_pattern_end(pattern, description: bool) -> None:
    ast = syntax().ast
    for element in reversed(pattern.elements):
        if not isinstance(element, ast.TextElement):
            return  # Неизвестно, чем закончится подставляемое значение.
        visible = _visible_text(element.value)
        for index in range(len(visible) - 1, -1, -1):
            char = visible[index]
            if char.isspace() or char in "\"'»”)]}":
                continue
            if description:
                if char not in ".,!?;:…":
                    tail = pattern.elements[-1]
                    trailing = len(tail.value) - len(tail.value.rstrip())
                    offset = len(tail.value) - trailing
                    tail.value = tail.value[:offset] + "." + tail.value[offset:]
            elif char in ".,!?;:…":
                if char == "." and re.search(r"(?:[^\W\d_]\.){2,}$", visible[: index + 1]):
                    return
                element.value = element.value[:index] + element.value[index + 1 :]
                continue
            return


def normalize_entity_message(node) -> None:
    ast = syntax().ast
    if not isinstance(node, ast.Message) or not node.id.name.startswith("ent-"):
        return
    if node.value is not None:
        _capitalize_pattern(node.value, upper=False)
        _normalize_pattern_end(node.value, description=False)
    for attribute in node.attributes:
        if attribute.id.name in {"desc", "suffix"}:
            _capitalize_pattern(attribute.value, upper=True)
            if attribute.id.name == "desc":
                _normalize_pattern_end(attribute.value, description=True)


def visible_parts(entry) -> list[str]:
    return [
        pattern_text(pattern)
        for pattern in ([entry.value] if entry.value else [])
        + [attribute.value for attribute in entry.attributes]
    ]


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
        return (
            "StringLiteral",
            "<text>" if any(character.isalpha() for character in value) else value,
        )
    if isinstance(node, list):
        return [structural_signature(value, technical) for value in node]
    if isinstance(node, ast.BaseNode):
        return (
            type(node).__name__,
            {
                key: structural_signature(value, technical or key == "arguments")
                for key, value in vars(node).items()
                if key != "span"
            },
        )
    return node


def assert_structure(source, target) -> None:
    if structural_signature(source) != structural_signature(target):
        raise FluentSyntaxError(
            "ИИ изменил структуру FTL, ключи, атрибуты, ссылки, аргументы или комментарии"
        )


MESSAGE_START_RE = re.compile(r"^(?P<id>-?[A-Za-z][A-Za-z0-9_-]*)[ \t]*=", re.MULTILINE)
ATTRIBUTE_RE = re.compile(r"^\s+\.([A-Za-z][A-Za-z0-9_-]*)\s*=", re.MULTILINE)
RICH_TAG_RE = re.compile(r"(?<!\\)\[(\/?)([A-Za-z][A-Za-z0-9_-]*)(?:[^\]]*)\]")
COMMA_SPACING_RE = re.compile(r"(?<=\w)[ \t]*,[ \t]*(?=\w)")
TECHNICAL_TEXT_RE = re.compile(r"(https?://\S+|<[^>\n]+>|\[[^\]\n]+\]|\{[^{}\n]+\})")
RICH_TAG_NAMES = {
    "bold",
    "bolditalic",
    "bullet",
    "bubblecontent",
    "bubbleheader",
    "center",
    "cmdlink",
    "color",
    "emoji",
    "enttex",
    "font",
    "head",
    "italic",
    "italics",
    "keybind",
    "mono",
    "name",
    "protodata",
    "radicon",
    "scramble",
    "textlink",
    "tex",
    "tutkeybind",
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


def parse_messages(text: str, resource=None) -> list[FluentMessage]:
    text = text.replace("\r\n", "\n")
    if resource is None:
        resource = parse_resource(text)
    result = []
    for key, node in entries(resource).items():
        start = text[: node.span.start].count("\n")
        end = text[: node.span.end].count("\n") + 1
        result.append(
            FluentMessage(
                key,
                start,
                end,
                tuple(text[node.span.start : node.span.end].splitlines()),
            )
        )
    return result


def message_map(text: str, resource=None) -> dict[str, FluentMessage]:
    return {message.id: message for message in parse_messages(text, resource)}


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


def rich_tags(text: str) -> Counter[str]:
    return Counter(
        match.group(0)
        for match in RICH_TAG_RE.finditer(text)
        if match.group(2).lower() in RICH_TAG_NAMES
    )


def same_message_payload(left: FluentMessage, right: FluentMessage) -> bool:
    return _canonical_message(left.text) == _canonical_message(right.text)


def _canonical_message(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def render_pattern(prefix: str, value: str) -> list[str]:
    prepared = escape_leading_multiline_markup(normalize_commas(value))
    lines = prepared.splitlines() or [""]
    if len(lines) == 1:
        return [f"{prefix} {lines[0]}".rstrip()]

    return [prefix] + [f"    {line}" for line in lines]


def render_entity_message(
    message_id: str, name: str | None, description: str | None, suffix: str | None
) -> str:
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
