from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
import re

from .dependencies import import_or_install
from .fluent import RICH_TAG_RE, RICH_TAG_NAMES, visible_parts
from .paths import TOOL_ROOT


TAG_RE = re.compile(r"</?[^>]+>|https?://\S+")


@dataclass(frozen=True)
class PassList:
    terms: tuple[str, ...] = ()
    ignored_files: tuple[str, ...] = ()

    @property
    def pattern(self):
        return _pass_pattern(self.terms)

    def strip(self, text: str) -> str:
        text = RICH_TAG_RE.sub(lambda match: " " if match.group(2).lower() in RICH_TAG_NAMES else match.group(0), text)
        return self.pattern.sub(" ", TAG_RE.sub(" ", text))

    def occurrences(self, text: str):
        return Counter(match.group(0) for match in self.pattern.finditer(text))

    def assert_preserved(self, source: str, target: str):
        if self.occurrences(source) != self.occurrences(target):
            raise ValueError("ИИ изменил слово или название из pass-листа")


@lru_cache(maxsize=16)
def _pass_pattern(terms):
    alternatives = "|".join(re.escape(term) for term in sorted(terms, key=len, reverse=True))
    return re.compile(r"(?<!\w)(?:" + alternatives + r")(?!\w)" if alternatives else r"(?!)", re.IGNORECASE)


def load_pass_list(repo_root: Path | None = None, path: Path | None = None) -> PassList:
    candidates = [path] if path else [TOOL_ROOT / "pass_list.yml"]
    if repo_root is not None and path is None:
        candidates.append(repo_root / "Tools" / "_sunrise" / "Schemas" / "ignore_list.yml")
    terms = set()
    ignored_files = set()
    for candidate in candidates:
        if not candidate.is_file():
            if path:
                raise FileNotFoundError(candidate)
            continue
        module = import_or_install("ruamel.yaml", "ruamel.yaml>=0.18,<1")
        data = module.YAML(typ="safe").load(candidate.read_text(encoding="utf-8-sig")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Pass-лист должен быть YAML-словарём: {candidate}")
        values = data.get("ignore_list", data.get("terms", []))
        if not isinstance(values, list) or any(not isinstance(item, str) for item in values):
            raise ValueError(f"ignore_list должен содержать список строк: {candidate}")
        terms.update(value.strip() for value in values if value.strip())
        # Whole-file exemptions are opt-in, never imported implicitly from the old validator.
        if os.environ.get("TRANSLATE_PASS_IGNORE_FILES", "false").lower() == "true":
            ignored_files.update(data.get("ignore_files", []))
    return PassList(tuple(sorted(terms)), tuple(sorted(ignored_files)))


def language_code(culture: str) -> str:
    module = import_or_install("langcodes", "langcodes>=3.4,<4")
    code = module.Language.get(culture.replace("_", "-")).language
    return {"no": "nb", "iw": "he"}.get(code, code)


@lru_cache(maxsize=16)
def _detector(source: str, target: str):
    module = import_or_install("lingua", "lingua-language-detector>=2.0,<3")
    supported = {language.iso_code_639_1.name.lower(): language for language in module.Language.all()}
    if target not in supported:
        raise ValueError(f"Язык {target} не поддерживается встроенным определителем. "
                         "Укажите TRANSLATE_LANGUAGE_PROFILE с собственным словарём языка; "
                         "неподдерживаемые языки не считаются переведёнными автоматически.")
    selected = {supported[target], module.Language.ENGLISH}
    if source in supported:
        selected.add(supported[source])
    return module.LanguageDetectorBuilder.from_languages(*selected).build(), supported[target]


@dataclass
class LanguageChecker:
    source_culture: str
    target_culture: str
    pass_list: PassList
    minimum_ratio: float = 0.8
    profile: Path | None = None

    def __post_init__(self):
        if not 0 < self.minimum_ratio <= 1:
            raise ValueError("Порог доли целевого языка должен быть в пределах (0, 1]")
        self.source_code = language_code(os.environ.get("TRANSLATE_DETECT_SOURCE", self.source_culture))
        self.target_code = language_code(os.environ.get("TRANSLATE_DETECT_TARGET", self.target_culture))
        if self.source_code == self.target_code:
            raise ValueError("Исходный и целевой языки совпадают")
        self.profile_pattern = None
        if self.profile:
            module = import_or_install("ruamel.yaml", "ruamel.yaml>=0.18,<1")
            data = module.YAML(typ="safe").load(self.profile.read_text(encoding="utf-8-sig"))
            if not isinstance(data, dict) or language_code(data.get("language", "")) != self.target_code:
                raise ValueError("Язык пользовательского профиля не совпадает с целевым языком")
            words = data.get("words")
            if not isinstance(words, list) or not words or any(not isinstance(word, str) or not word for word in words):
                raise ValueError("Пользовательский профиль должен содержать непустой список words")
            self.profile_pattern = re.compile(_pass_pattern(tuple(words)).pattern, re.IGNORECASE)
            self.detector = self.target_language = None
        else:
            self.detector, self.target_language = _detector(self.source_code, self.target_code)
        codes = import_or_install("langcodes", "langcodes>=3.4,<4")
        self.target_script = codes.Language.get(self.target_culture).maximize().script

    def ratio(self, text: str) -> float:
        text = self.pass_list.strip(text)
        total = sum(character.isalpha() for character in text)
        if not total:
            return 1.0
        if self.profile_pattern:
            return sum(sum(character.isalpha() for character in match.group(0))
                       for match in self.profile_pattern.finditer(text)) / total
        if total < 40:
            accepted = total if self.detector.detect_language_of(text) == self.target_language else 0
        else:
            sections = self.detector.detect_multiple_languages_of(text)
            accepted = sum(sum(character.isalpha() for character in text[section.start_index:section.end_index])
                           for section in sections if section.language == self.target_language)
        # A section classified as Russian must not make nearby Latin prose count as Russian.
        regex = import_or_install("regex", "regex>=2024.5")
        scripts = {"Hans": ["Han"], "Hant": ["Han"], "Jpan": ["Han", "Hiragana", "Katakana"],
                   "Kore": ["Hangul", "Han"], "Hrkt": ["Hiragana", "Katakana"]}.get(self.target_script, [self.target_script])
        alphabet = regex.compile("|".join(r"\p{Script=" + script + "}" for script in scripts))
        script_letters = sum(character.isalpha() and bool(alphabet.fullmatch(character)) for character in text)
        accepted = min(accepted, script_letters)
        return accepted / total

    def needs_translation(self, node) -> bool:
        return any(self.ratio(part) < self.minimum_ratio for part in visible_parts(node))

    def validate(self, node):
        for index, part in enumerate(visible_parts(node)):
            ratio = self.ratio(part)
            if ratio < self.minimum_ratio:
                raise ValueError(f"Поле {index + 1}: доля {self.target_culture} составляет {ratio:.0%}, "
                                 f"требуется не менее {self.minimum_ratio:.0%} (без pass-листа и разметки)")
