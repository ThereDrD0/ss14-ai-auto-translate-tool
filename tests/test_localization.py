from __future__ import annotations

import asyncio
import io
import os
import tempfile
import unittest
from collections import Counter
from collections.abc import Callable
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ss14_localization.ai import (
    AiConfig,
    AiEndpoint,
    OpenAICompatibleClient,
    ResponseTruncatedError,
)
from ss14_localization.budget import OutputBudget
from ss14_localization.dependencies import import_or_install
from ss14_localization.fluent import (
    FluentSyntaxError,
    entries,
    message_map,
    normalize_commas,
    parse_resource,
    pattern_text,
    render_entity_message,
    serialize_entry,
    serialize_resource,
)
from ss14_localization.language import LanguageChecker, PassList, load_pass_list
from ss14_localization.strings import prepare_target_files
from ss14_localization.translate import (
    _chunks,
    _parse_partial_response,
    _parse_translation_response,
    _replace_messages,
    _safe_chunk,
    _translate_chunk,
    translate_file,
    translate_files,
)
from ss14_localization.validation import validate_locale


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "Resources/Locale/en-US"
        self.target = self.root / "Resources/Locale/ru-RU"
        self.source.mkdir(parents=True)
        self.target.mkdir(parents=True)

    def write(self, path, text):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def prepare(self, dry_run=False):
        with redirect_stdout(io.StringIO()):
            return prepare_target_files(self.source, self.target, [Path(".")], dry_run)


class PreparationTests(Fixture):
    def test_normalizes_commas_in_existing_translation(self):
        self.write(self.source / "a.ftl", "a = CardBox ,Empty\n")
        self.write(self.target / "a.ftl", "a = Коробка ,пустая\n")

        self.prepare()

        self.assertEqual(
            (self.target / "a.ftl").read_text(encoding="utf-8"), "a = Коробка, пустая\n"
        )
        self.assertEqual(self.prepare().prepared_files, 0)

    def test_preserves_decimal_commas_during_preparation(self):
        self.write(self.source / "a.ftl", "ent-BoxMagazineRifleM52 = magazine box\n")
        translated = (
            "ent-BoxMagazineRifleM52 = набор магазинов М-52\n"
            "    .desc = Коробка с магазинами 5,56мм для М-52 каждого стандартного типа.\n"
        )
        self.write(self.target / "a.ftl", translated)

        self.assertEqual(self.prepare().prepared_files, 0)
        self.assertEqual((self.target / "a.ftl").read_text(encoding="utf-8"), translated)

    def test_formats_entity_translation_and_preserves_other_messages(self):
        self.write(
            self.source / "a.ftl",
            "ent-Box = Box\n    .desc = A box.\n    .suffix = Filled\n"
            "ent-Alert = Alert\n    .desc = Danger!\nother = Label\n",
        )
        self.write(
            self.target / "a.ftl",
            "ent-Box = Коробка!!!\n"
            "    .desc = [color=red]маленькая коробка[/color]\n"
            "    .suffix = «особая»\n"
            "ent-Alert = Сигнал?\n    .desc = опасно!\nother = Как Есть!\n",
        )

        self.prepare()

        self.assertEqual(
            (self.target / "a.ftl").read_text(encoding="utf-8"),
            "ent-Box = коробка\n"
            "    .desc = [color=red]Маленькая коробка[/color].\n"
            "    .suffix = «Особая»\n"
            "ent-Alert = сигнал\n    .desc = Опасно!\nother = Как Есть!\n",
        )
        self.assertEqual(self.prepare().prepared_files, 0)

    def test_entity_references_are_not_rewritten(self):
        text = (
            "ent-Child = { ent-BaseItem }\n"
            "    .desc = { ent-BaseItem.desc }\n"
            "    .suffix = { ent-BaseItem.suffix }\n"
        )
        self.write(self.source / "a.ftl", text)
        self.write(self.target / "a.ftl", text)

        self.assertEqual(self.prepare().prepared_files, 0)
        self.assertEqual((self.target / "a.ftl").read_text(encoding="utf-8"), text)

    def test_comment_translation_survives_different_spacing(self):
        self.write(
            self.source / "a.ftl",
            "# English attached\na = Hello\n\n# English separate\n\nb = World\n",
        )
        self.write(
            self.target / "a.ftl",
            "# Русский отдельный\n\na = Привет\n\n# Русский прикреплённый\nb = Мир\n",
        )

        self.prepare()

        result = (self.target / "a.ftl").read_text(encoding="utf-8")
        self.assertIn("# Русский отдельный", result)
        self.assertIn("# Русский прикреплённый", result)
        self.assertNotIn("# English", result)
        self.assertEqual(self.prepare().prepared_files, 0)

    def test_preserves_translated_attached_and_section_comments(self):
        self.write(
            self.source / "a.ftl",
            "# English title\na = Hello\n\n# Service\n\nb = World\nc = Bye\n",
        )
        self.write(
            self.target / "a.ftl",
            "# Русский заголовок\na = Привет\n\n# Сервис\n\nb = Мир\n",
        )

        self.prepare()

        result = (self.target / "a.ftl").read_text(encoding="utf-8")
        self.assertIn("# Русский заголовок", result)
        self.assertIn("# Сервис", result)
        self.assertNotIn("# English title", result)
        self.assertNotIn("# Service", result)
        self.assertIn("c = Bye", result)
        self.assertEqual(self.prepare().prepared_files, 0)

    def test_preserves_target_blank_lines_when_adding_key(self):
        self.write(self.source / "a.ftl", "a = Hello\nb = World\nc = Bye\n")
        original = "a = Привет\n\n\nb = Мир\n"
        self.write(self.target / "a.ftl", original)

        self.prepare()

        self.assertEqual(
            (self.target / "a.ftl").read_text(encoding="utf-8"),
            original + "c = Bye\n",
        )
        self.assertEqual(self.prepare().prepared_files, 0)

    def test_empty_source_file_is_not_sent_to_translation(self):
        self.write(self.source / "empty.ftl", "")
        self.write(self.target / "empty.ftl", "filled = Привет\n")
        self.write(self.source / "filled.ftl", "filled = Hello\n")
        result = self.prepare()
        self.assertEqual(result.target_files, (self.target / "filled.ftl",))
        self.assertFalse((self.target / "empty.ftl").exists())

    def test_moves_reorders_and_preserves_translation(self):
        self.write(self.source / "a.ftl", "# Source comment\na = Hello\nb = World\nc = Bye\n")
        self.write(self.target / "a.ftl", "c = Пока\na = Привет\n")
        self.write(self.target / "wrong/path.ftl", "b = Мир\n")
        result = self.prepare()
        text = (self.target / "a.ftl").read_text(encoding="utf-8")
        self.assertEqual(list(message_map(text)), ["a", "b", "c"])
        self.assertIn("b = Мир", text)
        self.assertIn("# Source comment", text)
        self.assertFalse((self.target / "wrong/path.ftl").exists())
        self.assertEqual(result.moved_messages, 1)
        self.assertEqual(self.prepare().prepared_files, 0)

    def test_adds_attributes_in_source_order_and_keeps_target_only_keys(self):
        self.write(
            self.source / "a.ftl",
            "a = Hello\n    .desc = Description\n    .suffix = Label\n-term = Term\n",
        )
        self.write(self.target / "a.ftl", "a = Привет\n    .suffix = Метка\ncustom = Особое\n")
        self.prepare()
        resource = entries(parse_resource((self.target / "a.ftl").read_text(encoding="utf-8")))
        self.assertEqual(list(resource), ["a", "-term", "custom"])
        self.assertEqual([a.id.name for a in resource["a"].attributes], ["desc", "suffix"])
        self.assertEqual(pattern_text(resource["a"].attributes[1].value), "Метка")

    def test_conflicting_duplicates_fail_without_writes(self):
        self.write(self.source / "a.ftl", "a = Hello\n")
        self.write(self.target / "a.ftl", "a = Привет\n")
        self.write(self.target / "b.ftl", "a = Иначе\n")
        before = {p: p.read_bytes() for p in self.target.rglob("*.ftl")}
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertEqual(before, {p: p.read_bytes() for p in self.target.rglob("*.ftl")})

    def test_identical_duplicates_collapse(self):
        self.write(self.source / "a.ftl", "a = Hello\n")
        self.write(self.target / "a.ftl", "a = Привет\n")
        self.write(self.target / "b.ftl", "a = Привет\n")
        self.prepare()
        self.assertFalse((self.target / "b.ftl").exists())

    def test_dry_run_keeps_complete_virtual_new_files(self):
        self.write(self.source / "nested/a.ftl", "a = Hello\n")
        result = self.prepare(True)
        target = self.target / "nested/a.ftl"
        self.assertIn(target, result.target_files)
        self.assertIn("a = Hello", result.planned_texts[target])
        self.assertFalse(target.exists())

    def test_invalid_source_fails_before_writes(self):
        self.write(self.source / "a.ftl", "a = Hello\n")
        self.write(self.source / "b.ftl", "b = Hello {\n")
        with self.assertRaises(FluentSyntaxError):
            self.prepare()
        self.assertEqual(list(self.target.rglob("*.ftl")), [])

    def test_source_duplicates_and_overlapping_roots_fail(self):
        self.write(self.source / "a.ftl", "a = Hello\n")
        self.write(self.source / "b.ftl", "a = Again\n")
        with self.assertRaises(ValueError):
            self.prepare()
        with self.assertRaises(ValueError):
            prepare_target_files(self.source, self.source, [Path(".")])
        with self.assertRaises(ValueError):
            prepare_target_files(self.source, self.target, [Path("../")])

    def test_header_comments_and_selects_mirror(self):
        self.write(
            self.source / "a.ftl",
            "### Group\n# Comment\na = { $count ->\n    [one] One\n   *[other] Many\n}\n",
        )
        self.prepare()
        self.assertIn("### Group", (self.target / "a.ftl").read_text(encoding="utf-8"))
        parse_resource((self.target / "a.ftl").read_text(encoding="utf-8"))


class SyntaxTests(unittest.TestCase):
    def test_comma_spacing_skips_markup_urls_and_formats_prototypes(self):
        self.assertEqual(
            normalize_commas(
                '[color="a,b"] CardBox ,Filled ,54 https://x.test/a,b '
                "{ NUMBER($count,minimumFractionDigits: 2) } 5,56мм 7, 62мм"
            ),
            '[color="a,b"] CardBox, Filled, 54 https://x.test/a,b '
            "{ NUMBER($count,minimumFractionDigits: 2) } 5,56мм 7, 62мм",
        )
        self.assertIn(
            ".suffix = CardBox, Filled, 54",
            render_entity_message("box", "Box", None, "CardBox ,Filled ,54"),
        )

    def check(self, source, target):
        return _parse_translation_response(target, message_map(source))

    def test_plain_ftl_accepted_json_rejected(self):
        self.assertEqual(set(self.check("a = Hello", "a = Привет")), {"a"})
        with self.assertRaises(ValueError):
            self.check("a = Hello", '[{"id":"a","text":"a = Привет"}]')

    def test_invalid_syntax_and_duplicates_rejected(self):
        for text in ("a = Привет {", "a = First\na = Second"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.check("a = Hello", text)

    def test_linguini_syntax_differences_are_supported(self):
        parse_resource('#comment without a space\n#### comment with extra hashes\na = { "\\{" }\n')
        parse_resource("a = { message .attribute }\n")
        parse_resource(
            "a = { -term(variable: $value, message: other, term: -other, "
            "function: FUNC(), nested: { $value }) }\n"
        )
        duplicate_attributes = parse_resource("a = Foo\n    .desc = A\n    .desc = B\n")
        self.assertEqual(len(entries(duplicate_attributes)["a"].attributes), 2)

    def test_unterminated_string_is_rejected_instead_of_hanging_like_linguini(self):
        with self.assertRaises(FluentSyntaxError):
            parse_resource('a = { "unterminated }')

    def test_missing_extra_and_reference_changes_rejected(self):
        for source, target in (
            ("a = Hello\nb = World", "a = Привет"),
            ("a = Hello", "a = Привет\nb = Лишнее"),
            ("a = { original }", "a = { changed }"),
            (
                "a = { NUMBER($n, minimumFractionDigits: 2) }",
                "a = { NUMBER($n, minimumFractionDigits: 3) }",
            ),
            ("a = [color=red]Hello[/color]", "a = [color=blue]Привет[/color]"),
            ("# Same\na = Hello", "# Changed\na = Привет"),
        ):
            with self.subTest(source=source), self.assertRaises(ValueError):
                self.check(source, target)

    def test_visible_literal_can_be_translated_technical_literal_cannot(self):
        self.check('a = { "Hello" }', 'a = { "Привет" }')
        with self.assertRaises(ValueError):
            self.check(
                'a = { GENDER($user, form: "male") }',
                'a = { GENDER($user, form: "мужской") }',
            )

    def test_terms_and_select_variants(self):
        self.check("-term = Hello", "-term = Привет")
        with self.assertRaises(ValueError):
            self.check(
                "a = { $n ->\n [one] Hello\n *[other] World\n}",
                "a = { $n ->\n [few] Привет\n *[other] Мир\n}",
            )


class LanguageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.passed = load_pass_list()
        cls.ru = LanguageChecker("en-US", "ru-RU", cls.passed)
        cls.fr = LanguageChecker("en-US", "fr-FR", cls.passed)

    def test_any_supported_locale_not_just_ru_en(self):
        self.assertEqual(self.ru.minimum_ratio, 0.15)
        self.assertEqual(self.fr.minimum_ratio, 0.15)
        self.assertEqual(self.fr.ratio("Bonjour tout le monde"), 1)
        self.assertEqual(self.fr.ratio("Hello world"), 0)
        de = LanguageChecker("fr-FR", "de-DE", self.passed)
        self.assertEqual(de.ratio("Hallo Welt"), 1)

    def test_short_uppercase_and_wrong_script_not_ignored(self):
        for value in ("On", "HELLO WORLD", "你好世界"):
            with self.subTest(value=value):
                self.assertLess(self.ru.ratio(value), 0.8)

    def test_pass_word_not_whole_line(self):
        self.assertEqual(self.ru.ratio("Desert Eagle"), 1)
        self.assertEqual(self.ru.ratio("M1 Garand"), 1)
        self.assertEqual(self.ru.ratio("AI APC DNA GPS PDA UI NT IDs"), 1)
        self.assertLess(self.ru.ratio("Desert Eagle Hello world"), 0.8)
        self.assertEqual(self.ru.ratio("Desert Eagle — мощное оружие"), 1)
        self.assertLess(self.ru.ratio("[Hello World]"), 0.8)

    def test_language_share_is_combined_across_fields(self):
        text = (
            "ent-WeaponSubMachineGunSP91RC = SP-91-RC\n"
            "    .desc = Компактный пистолет-пулемёт для контроля беспорядков.\n"
        )
        node = entries(parse_resource(text))["ent-WeaponSubMachineGunSP91RC"]
        self.assertEqual(self.ru.ratio("SP-91-RC"), 1)
        self.ru.validate(node)
        self.assertFalse(self.ru.needs_translation(node))
        english = entries(
            parse_resource("name = SP-91-RC\n    .desc = Compact submachine gun.\n")
        )["name"]
        self.assertTrue(self.ru.needs_translation(english))
        self.assertLess(self.ru.ratio("HELLO WORLD"), 0.5)

    def test_model_code_exclusion_does_not_eat_prose_or_partial_codes(self):
        self.assertEqual(self.ru.ratio("LSE-400b"), 1)
        self.assertIn("LSE-400bc", self.ru.pass_list.strip("LSE-400bc"))  # noqa: B005 — метод PassList.
        self.assertIn("Hello2", self.ru.pass_list.strip("Hello2"))  # noqa: B005 — метод PassList.
        self.assertLess(self.ru.ratio("SP-91-RC Compact submachine gun"), 0.5)
        self.assertLess(self.ru.ratio("LSE-400b English description"), 0.5)

    def test_response_language_share_is_combined_across_messages(self):
        source = "code = SP-91-RC\ndescription = Compact submachine gun.\n"
        translated = (
            "code = SP-91-RC\n"
            "description = Компактный пистолет-пулемёт для контроля беспорядков.\n"
        )
        self.assertEqual(
            set(_parse_translation_response(translated, message_map(source), checker=self.ru)),
            {"code", "description"},
        )
        with self.assertRaises(ValueError):
            _parse_translation_response(source, message_map(source), checker=self.ru)

    def test_pass_boundaries_case_and_preservation(self):
        passed = PassList(("ID", "Desert Eagle"))
        self.assertEqual(passed.strip("identity"), "identity")  # noqa: B005 — метод PassList.
        with self.assertRaises(ValueError):
            passed.assert_preserved("Desert Eagle", "desert eagle")

    def test_new_target_language_letter_does_not_break_pass_list(self):
        passed = PassList(("F",))
        passed.assert_preserved("Это «Р» или «Ф»?", "Is it «R» or «F»?")
        with self.assertRaisesRegex(ValueError, "исчезли.*F"):
            passed.assert_preserved("Press F", "Нажмите Ф")

    def test_english_plural_of_protected_term_requires_singular(self):
        passed = PassList(("ID", "APC"))
        self.assertEqual(passed.required("IDs APCs"), Counter({"ID": 1, "APC": 1}))
        self.assertTrue(passed.needs_normalization("IDs"))
        passed.assert_preserved("IDs APCs", "ID APC")
        with self.assertRaises(ValueError):
            passed.assert_preserved("IDs", "IDs")
        with self.assertRaises(ValueError):
            passed.assert_preserved("IDs", "идентификаторы")
        self.assertEqual(
            PassList(("ID", "red")).normalize_plurals("IDs [color=reds]текст[/color]"),
            "ID [color=reds]текст[/color]",
        )
        source = "a = IDs"
        self.assertTrue(self.ru.needs_translation(entries(parse_resource(source))["a"]))
        self.assertIn(
            "ID",
            _parse_translation_response("a = ID", message_map(source), checker=self.ru)["a"],
        )
        mixed = "a = Возвращает IDs всех контейнеров сущности."
        self.assertTrue(self.ru.needs_translation(entries(parse_resource(mixed))["a"]))
        self.assertIn(
            "Возвращает ID",
            _parse_translation_response(
                "a = Возвращает ID всех контейнеров сущности.",
                message_map(mixed),
                checker=self.ru,
            )["a"],
        )
        with self.assertRaisesRegex(ValueError, "переписал уже переведённое"):
            _parse_translation_response(
                "a = Показывает ID контейнеров.", message_map(mixed), checker=self.ru
            )

    def test_pass_list_ignores_contractions_and_markup(self):
        passed = PassList(("T", "red", "green"))
        passed.assert_preserved(
            "It isn't [color=red]ready[/color].", "Он [color=red]готов[/color]."
        )
        with self.assertRaisesRegex(ValueError, "исчезли.*T"):
            passed.assert_preserved("T is ready", "Готово")

    def test_select_variant_prose_is_checked(self):
        node = entries(parse_resource("a = { $n ->\n [one] Hello\n *[other] World\n}"))["a"]
        self.assertTrue(self.ru.needs_translation(node))

    def test_already_translated_attribute_cannot_be_rewritten(self):
        source = "a = Hello\n    .desc = Описание предмета\n"
        with self.assertRaises(ValueError):
            _parse_translation_response(
                "a = Привет\n    .desc = Новое описание\n",
                message_map(source),
                checker=self.ru,
            )

    def test_unsupported_language_errors_instead_of_silent_success(self):
        with self.assertRaises(ValueError):
            LanguageChecker("en-US", "zz", self.passed)

    def test_custom_profile_supports_other_languages(self):
        with tempfile.TemporaryDirectory() as temporary:
            profile = Path(temporary) / "language.yml"
            profile.write_text("language: tlh\nwords: [Qapla, nuqneh]\n", encoding="utf-8")
            checker = LanguageChecker("en-US", "tlh", PassList(), profile=profile)
            self.assertEqual(checker.ratio("Qapla nuqneh"), 1)
            self.assertEqual(checker.ratio("Hello world"), 0)

    def test_russian_ratio_and_markup_exclusions(self):
        mixed = "Привет привет Hello world"
        self.assertGreaterEqual(self.ru.ratio(mixed), 0.5)
        self.assertLess(self.ru.ratio(mixed), 0.8)
        self.assertFalse(
            self.ru.needs_translation(
                entries(parse_resource(f"english-key-name = {mixed}"))["english-key-name"]
            )
        )
        decorated = (
            "[bold][BubbleHeader]Привет привет[/BubbleHeader][/bold] "
            '[tutkeybind="UIClick"] Hello world 123 !?'
        )
        self.assertEqual(self.ru.ratio(decorated), self.ru.ratio(mixed))
        self.assertEqual(
            self.ru.pass_list.strip(  # noqa: B005 — метод PassList.
                "[Name]Привет[/Name] [BubbleContent]мир[/BubbleContent]"
            ).split(),
            ["Привет", "мир"],
        )

    def test_mixed_scripts_ratio_is_independent_of_threshold(self):
        self.assertGreaterEqual(
            self.ru.ratio("Это русское описание игрового предмета на космической станции Hello"),
            0.8,
        )
        self.assertLess(self.ru.ratio("Привет Hello world this is an English description"), 0.8)

    def test_short_russian_text_with_long_command_is_already_translated(self):
        for text in (
            "Использование: clearnetworklinkoverlays",
            "Использование: bloodcult_addtarget <ckey>",
        ):
            with self.subTest(text=text):
                self.assertGreaterEqual(self.ru.ratio(text), 0.15)
                self.assertFalse(
                    self.ru.needs_translation(entries(parse_resource(f"help = {text}"))["help"])
                )

    def test_english_sentence_with_russian_word_still_needs_translation(self):
        for text in (
            "Привет, the device is ready",
            "The device is ready. Привет",
            "Использование: press the button",
        ):
            with self.subTest(text=text):
                self.assertLess(self.ru.ratio(text), 0.15)
                self.assertTrue(
                    self.ru.needs_translation(entries(parse_resource(f"help = {text}"))["help"])
                )


class BudgetTests(unittest.TestCase):
    def test_automatic_chunks_follow_budget_without_key_count_limit(self):
        messages = list(message_map("\n".join(f"key-{i} = Hello" for i in range(101))).values())
        self.assertEqual([len(chunk) for chunk in _chunks(messages, 0)], [101])

    def test_chunks_use_manual_output_budget_with_margin(self):
        budget = OutputBudget(192, 0.65, 3, 16)
        messages = list(
            message_map("a = Hello world\nb = Hello world\nc = Hello world\n").values()
        )
        chunks = _chunks(messages, 10000, budget)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(
                budget.estimated_output("\n\n".join(m.text for m in chunk)),
                budget.capacity,
            )
        self.assertGreater(budget.tokens("Привет"), len("Привет"))

    def test_long_text_splits_without_losing_source(self):
        budget = OutputBudget(256, 0.65, 2, 16)
        render = lambda value: serialize_entry(
            __import__("ss14_localization.translate", fromlist=["_fragment_node"])._fragment_node(
                value
            )
        )
        text = "Hello world " * 30
        parts = budget.split_text(text, render)
        self.assertGreater(len(parts), 1)
        self.assertEqual("".join(parts), text)
        self.assertTrue(all(budget.fits(render(part)) for part in parts))

    def test_invalid_budget_rejected(self):
        for create in (
            lambda: OutputBudget(max_tokens=0),
            lambda: OutputBudget(safety=1),
            lambda: OutputBudget(expansion=0.5),
            lambda: OutputBudget(reserve=100000),
        ):
            with self.subTest(create=create), self.assertRaises(ValueError):
                create()


class FakeClient:
    def __init__(self, invalid_first=False, truncated_first=False):
        self.calls = []
        self.invalid_first = invalid_first
        self.truncated_first = truncated_first
        self._on_retry: Callable[..., object] | None = None

    async def chat(self, messages):
        self.calls.append(messages)
        if len(self.calls) == 1 and self.truncated_first:
            raise ResponseTruncatedError("length")
        if len(self.calls) == 1 and self.invalid_first:
            return "a = Привет {"
        payload = (
            messages[-2]["content"]
            if messages[-1]["content"].startswith(("Предыдущая попытка", "Исправьте ошибки"))
            else messages[-1]["content"]
        )
        resource = parse_resource(payload)
        from ss14_localization.translate import _text_slots

        for node in entries(resource).values():
            for slot in _text_slots(node):
                slot.value = slot.value.replace("Hello", "Привет").replace("World", "Мир")
        return serialize_resource(resource)


class TranslationTests(Fixture, unittest.IsolatedAsyncioTestCase):
    def test_partial_response_keeps_valid_keys_after_bad_pass_term(self):
        source = message_map("a = Hello\nb = NanoTrasen device\nc = World\n")
        checker = LanguageChecker("en-US", "ru-RU", PassList(("NanoTrasen",)))
        valid, invalid = _parse_partial_response(
            "a = Привет\nb = Nanotrasen устройство\nc = Мир", source, checker=checker
        )
        self.assertEqual(set(valid), {"a", "c"})
        self.assertEqual(set(invalid), {"b"})

    def test_partial_response_keeps_keys_after_malformed_entry(self):
        source = message_map("a = Hello\nb = World\nc = Hello\n")
        valid, invalid = _parse_partial_response(
            "a = Привет\nb = {\nc = Привет", source, checker=self.checker
        )
        self.assertEqual(set(valid), {"a", "c"})
        self.assertEqual(set(invalid), {"b"})

    def test_replacement_formats_entity_translation(self):
        original = "ent-Box = Box\n    .desc = A box.\n    .suffix = Filled\n"
        replacement = "ent-Box = Коробка.\n    .desc = маленькая коробка\n    .suffix = особая"

        self.assertEqual(
            _replace_messages(original, {"ent-Box": replacement}),
            "ent-Box = коробка\n    .desc = Маленькая коробка.\n    .suffix = Особая\n",
        )

    def test_replacement_normalizes_commas_without_touching_syntax(self):
        original = "a = CardBox ,Empty { NUMBER($count, minimumFractionDigits: 2) }\n"
        replacement = "a = Коробка ,пустая { NUMBER($count, minimumFractionDigits: 2) }"

        self.assertEqual(
            _replace_messages(original, {"a": replacement}),
            "a = Коробка, пустая { NUMBER($count, minimumFractionDigits: 2) }\n",
        )

    def setUp(self):
        super().setUp()
        self.checker = LanguageChecker("en-US", "ru-RU", load_pass_list())

    def test_retranslation_keeps_existing_comment(self):
        original = "# Русский комментарий\na = Привет\n\n# Сервис\n\nb = Мир\n"
        updated = _replace_messages(original, {"a": "# English comment\na = Здравствуй"})
        self.assertEqual(
            updated,
            "# Русский комментарий\na = Здравствуй\n\n# Сервис\n\nb = Мир\n",
        )

    async def test_translation_preserves_blank_lines_between_keys(self):
        path = self.target / "a.ftl"
        self.write(path, "a = Hello\n\n\nb = World\n")

        await translate_file(
            path, FakeClient(), "Prompt", 4000, target_culture="ru-RU", checker=self.checker
        )

        self.assertEqual(path.read_text(encoding="utf-8"), "a = Привет\n\n\nb = Мир\n")

    async def test_token_saving_only_sends_untranslated_keys(self):
        for save_tokens in (False, True):
            with self.subTest(save_tokens=save_tokens):
                path = self.target / "mixed.ftl"
                self.write(path, "ready = Привет\nmissing = Hello\n")
                client = FakeClient()
                count, changed = await translate_file(
                    path,
                    client,
                    "Prompt",
                    4000,
                    target_culture="ru-RU",
                    checker=self.checker,
                    save_tokens=save_tokens,
                )
                self.assertEqual((count, changed), (1, True))
                self.assertEqual(
                    path.read_text(encoding="utf-8"),
                    "ready = Привет\nmissing = Привет\n",
                )
                sent = [item["content"] for item in client.calls[0] if item["role"] == "user"]
                self.assertEqual(sent[-1], "missing = Hello")
                self.assertEqual(any("ready = Привет" in item for item in sent), not save_tokens)

    async def test_think_preamble_is_removed_before_budget_and_file_write(self):
        class ThinkingClient:
            async def chat(self, messages):
                return (
                    "<think>" + "рассуждение " * 1000 + "</think>a = Привет\n    .desc = Описание"
                )

        path = self.target / "a.ftl"
        self.write(path, "a = Hello\n    .desc = Description\n")
        count, changed = await translate_file(
            path,
            ThinkingClient(),
            "Prompt",
            4000,
            target_culture="ru-RU",
            checker=self.checker,
        )
        self.assertEqual((count, changed), (1, True))
        self.assertEqual(path.read_text(encoding="utf-8"), "a = Привет\n    .desc = Описание\n")

    async def test_raw_request_and_retry_feedback(self):
        client = FakeClient(invalid_first=True)
        retries = []
        client._on_retry = lambda *items: retries.append(items)
        result = await _translate_chunk(
            client,
            "Prompt",
            list(message_map("a = Hello").values()),
            "ru-RU",
            self.checker,
        )
        self.assertIn("Привет", result["a"])
        self.assertEqual(client.calls[0][1]["content"], "a = Hello")
        self.assertEqual(len(client.calls[1]), 3)
        self.assertIn("Предыдущая попытка", client.calls[1][-1]["content"])
        self.assertEqual(retries[0][-2:], ("a = Hello", "a = Привет {"))

    async def test_request_names_exact_pass_terms_for_each_key(self):
        class TermsClient:
            async def chat(self, messages):
                self.messages = messages
                return "a = Получает ID всех контейнеров.\nb = Получает строковый id контейнеров."

        checker = LanguageChecker("en-US", "ru-RU", PassList(("ID", "NT")))
        client = TermsClient()
        chunk = list(
            message_map(
                "a = Gets the IDs of all containers.\nb = Gets the string id of containers."
            ).values()
        )
        result = await _translate_chunk(client, "Prompt", chunk, "ru-RU", checker)
        self.assertEqual(set(result), {"a", "b"})
        self.assertIn("a: 'ID'", client.messages[0]["content"])
        self.assertIn("b: 'id'", client.messages[0]["content"])
        self.assertNotIn("'NT'", client.messages[0]["content"])

    async def test_truncation_reduces_chunk(self):
        client = FakeClient(truncated_first=True)
        retries = []
        client._on_retry = lambda *items: retries.append(items)
        result = await _safe_chunk(
            client,
            "Prompt",
            list(message_map("a = Hello\nb = World").values()),
            self.checker,
            OutputBudget(),
        )
        self.assertEqual(set(result), {"a", "b"})
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(retries[0][0], "split")

    async def test_single_message_truncation_uses_text_fragments(self):
        client = FakeClient(truncated_first=True)
        result = await _safe_chunk(
            client,
            "Prompt",
            list(message_map("a = Hello").values()),
            self.checker,
            OutputBudget(),
        )
        self.assertIn("Привет", result["a"])
        self.assertIn("translation-part", client.calls[1][1]["content"])

    async def test_large_single_message_fragmented_every_request_fits(self):
        path = self.target / "a.ftl"
        self.write(path, "a = " + "Hello " * 50 + "\n")
        client = FakeClient()
        budget = OutputBudget(512, 0.65, 2, 16)
        count, changed = await translate_file(
            path,
            client,
            "Prompt",
            10000,
            target_culture="ru-RU",
            checker=self.checker,
            budget=budget,
        )
        self.assertEqual(count, 1)
        self.assertTrue(changed)
        self.assertGreater(len(client.calls), 1)
        self.assertTrue(all(budget.fits(call[1]["content"], "Prompt") for call in client.calls))
        self.assertNotIn("Hello", path.read_text(encoding="utf-8"))
        parse_resource(path.read_text(encoding="utf-8"))

    async def test_dry_run_no_api_no_write_including_virtual_file(self):
        path = self.target / "new.ftl"
        with (
            patch("ss14_localization.translate.OpenAICompatibleClient") as client,
            redirect_stdout(io.StringIO()),
        ):
            result = await translate_files(
                [path],
                "Prompt",
                4000,
                target_culture="ru-RU",
                dry_run=True,
                checker=self.checker,
                texts={path: "a = Hello\n"},
            )
        client.assert_not_called()
        self.assertFalse(path.exists())
        self.assertEqual(result.failed_files, ())

    async def test_large_visible_string_literal_preserves_literal_braces(self):
        path = self.target / "a.ftl"
        self.write(path, 'a = { "' + "Hello { " * 40 + '" }\n')
        client = FakeClient()
        count, changed = await translate_file(
            path,
            client,
            "Prompt",
            10000,
            target_culture="ru-RU",
            checker=self.checker,
            budget=OutputBudget(512, 0.65, 2, 16),
        )
        self.assertEqual(count, 1)
        self.assertTrue(changed)
        result = entries(parse_resource(path.read_text(encoding="utf-8")))["a"]
        self.assertEqual(pattern_text(result.value).count("{"), 40)
        self.assertNotIn("Hello", pattern_text(result.value))

    async def test_async_concurrency_preserved(self):
        active = peak = 0
        checker = self.checker

        class ConcurrentClient(FakeClient):
            async def chat(self, messages):
                nonlocal active, peak
                active += 1
                peak = max(peak, active)
                await asyncio.sleep(0.02)
                result = await super().chat(messages)
                active -= 1
                return result

        paths = [self.target / f"{i}.ftl" for i in range(4)]
        for i, path in enumerate(paths):
            self.write(path, f"a{i} = Hello\n")
        with patch(
            "ss14_localization.translate.OpenAICompatibleClient",
            return_value=ConcurrentClient(),
        ):
            result = await translate_files(
                paths,
                "Prompt",
                10,
                target_culture="ru-RU",
                concurrency=2,
                checker=checker,
            )
        self.assertEqual(peak, 2)
        self.assertEqual(result.changed_files, 4)

    async def test_finished_file_is_written_while_another_request_waits(self):
        release = asyncio.Event()
        fast_done = asyncio.Event()

        class SlowClient(FakeClient):
            async def chat(self, messages):
                if "Hello" in messages[-1]["content"]:
                    await release.wait()
                return await super().chat(messages)

        slow, fast = self.target / "slow.ftl", self.target / "fast.ftl"
        self.write(slow, "slow = Hello\n")
        self.write(fast, "fast = World\n")

        def on_event(kind, path, _payload):
            if kind == "completed" and path == fast:
                fast_done.set()

        with patch(
            "ss14_localization.translate.OpenAICompatibleClient", return_value=SlowClient()
        ):
            task = asyncio.create_task(
                translate_files(
                    [slow, fast], "Prompt", 10, checker=self.checker, on_event=on_event
                )
            )
            try:
                await asyncio.wait_for(fast_done.wait(), 3)
                self.assertEqual(fast.read_text(encoding="utf-8"), "fast = Мир\n")
                self.assertEqual(slow.read_text(encoding="utf-8"), "slow = Hello\n")
            finally:
                release.set()
                result = await task
        self.assertEqual(result.changed_files, 2)

    async def test_file_waits_for_all_its_chunks_before_writing(self):
        release = asyncio.Event()
        fast_reply = asyncio.Event()

        class SlowClient(FakeClient):
            async def chat(self, messages):
                if "Hello" in messages[-1]["content"]:
                    await release.wait()
                result = await super().chat(messages)
                fast_reply.set()
                return result

        path = self.target / "two-chunks.ftl"
        original = "a = Hello\nb = World\n"
        self.write(path, original)
        with patch(
            "ss14_localization.translate.OpenAICompatibleClient", return_value=SlowClient()
        ):
            task = asyncio.create_task(translate_files([path], "Prompt", 10, checker=self.checker))
            try:
                await asyncio.wait_for(fast_reply.wait(), 3)
                await asyncio.sleep(0)
                self.assertEqual(path.read_text(encoding="utf-8"), original)
            finally:
                release.set()
                result = await task
        self.assertEqual(result.changed_files, 1)
        self.assertEqual(path.read_text(encoding="utf-8"), "a = Привет\nb = Мир\n")

    async def test_valid_keys_are_saved_and_bad_keys_join_new_work(self):
        retry_started = asyncio.Event()
        release = asyncio.Event()

        class PartialClient:
            def __init__(self):
                self.calls = []
                self._on_retry = None

            async def chat(self, messages):
                payload = messages[-2]["content"] if len(messages) > 2 else messages[-1]["content"]
                keys = list(message_map(payload))
                self.calls.append(keys)
                if len(self.calls) == 1:
                    return f"{keys[0]} = Привет\n{keys[1]} = {{"
                retry_started.set()
                await release.wait()
                return "\n".join(
                    f"{key} = {'Мир' if key.endswith('-b') else 'Привет'}" for key in keys
                )

        first, second = self.target / "first.ftl", self.target / "second.ftl"
        self.write(first, "a = Hello\nb = World\n")
        self.write(second, "c = Hello\n")
        client = PartialClient()
        with patch("ss14_localization.translate.OpenAICompatibleClient", return_value=client):
            task = asyncio.create_task(
                translate_files([first, second], "Prompt", 70, checker=self.checker, concurrency=1)
            )
            try:
                await asyncio.wait_for(retry_started.wait(), 3)
                self.assertEqual(first.read_text(encoding="utf-8"), "a = Привет\nb = World\n")
                self.assertEqual(second.read_text(encoding="utf-8"), "c = Hello\n")
                self.assertEqual([key[-2:] for key in client.calls[1]], ["-b", "-c"])
            finally:
                release.set()
                result = await task
        self.assertEqual((result.translated_messages, result.changed_files), (3, 2))
        self.assertEqual(first.read_text(encoding="utf-8"), "a = Привет\nb = Мир\n")
        self.assertEqual(second.read_text(encoding="utf-8"), "c = Привет\n")

    async def test_truncated_response_keeps_completed_keys(self):
        class TruncatedClient:
            def __init__(self):
                self.calls = []
                self._on_retry = None

            async def chat(self, messages):
                payload = messages[-2]["content"] if len(messages) > 2 else messages[-1]["content"]
                keys = list(message_map(payload))
                self.calls.append(keys)
                if len(self.calls) == 1:
                    raise ResponseTruncatedError("length", f"{keys[0]} = Привет\n{keys[1]} = {{")
                return f"{keys[0]} = Мир"

        path = self.target / "truncated.ftl"
        self.write(path, "a = Hello\nb = World\n")
        client = TruncatedClient()
        with patch("ss14_localization.translate.OpenAICompatibleClient", return_value=client):
            result = await translate_files([path], "Prompt", 0, checker=self.checker)
        self.assertEqual((result.translated_messages, result.changed_files), (2, 1))
        self.assertEqual([len(keys) for keys in client.calls], [2, 1])
        self.assertEqual(path.read_text(encoding="utf-8"), "a = Привет\nb = Мир\n")

    async def test_exhausted_retry_does_not_erase_valid_key(self):
        class BrokenClient:
            _on_retry = None

            async def chat(self, messages):
                payload = messages[-2]["content"] if len(messages) > 2 else messages[-1]["content"]
                return "\n".join(
                    f"{key} = {'Привет' if key.endswith('-a') else '{'}"
                    for key in message_map(payload)
                )

        path = self.target / "partly-valid.ftl"
        self.write(path, "a = Hello\nb = World\n")
        with patch(
            "ss14_localization.translate.OpenAICompatibleClient", return_value=BrokenClient()
        ):
            result = await translate_files([path], "Prompt", 0, checker=self.checker)
        self.assertEqual(result.translated_messages, 1)
        self.assertEqual(result.changed_files, 1)
        self.assertEqual(result.failed_files, (path,))
        self.assertEqual(path.read_text(encoding="utf-8"), "a = Привет\nb = World\n")

    async def test_invalid_multi_message_response_splits_without_full_retry(self):
        paths = [self.target / "one.ftl", self.target / "two.ftl"]
        self.write(paths[0], "a = Hello\n")
        self.write(paths[1], "b = World\n")
        client = FakeClient(invalid_first=True)
        with patch("ss14_localization.translate.OpenAICompatibleClient", return_value=client):
            result = await translate_files(paths, "Prompt", 0, checker=self.checker)
        self.assertEqual(result.changed_files, 2)
        self.assertEqual(len(client.calls), 3)

    async def test_same_keys_in_different_files_share_one_request(self):
        paths = [self.target / "one.ftl", self.target / "two.ftl"]
        self.write(paths[0], "ready = Привет\nsame = Hello\n")
        self.write(paths[1], "same = World\n")
        for save_tokens in (False, True):
            with self.subTest(save_tokens=save_tokens):
                client = FakeClient()
                with patch(
                    "ss14_localization.translate.OpenAICompatibleClient",
                    return_value=client,
                ):
                    result = await translate_files(
                        paths,
                        "Prompt",
                        0,
                        target_culture="ru-RU",
                        checker=self.checker,
                        save_tokens=save_tokens,
                    )
                self.assertEqual((result.translated_messages, result.changed_files), (2, 2))
                self.assertEqual(len(client.calls), 1)
                payload = client.calls[0][-1]["content"]
                self.assertIn("translation-batch-0-same = Hello", payload)
                self.assertIn("translation-batch-1-same = World", payload)
                self.assertEqual(
                    paths[0].read_text(encoding="utf-8"),
                    "ready = Привет\nsame = Привет\n",
                )
                self.assertEqual(paths[1].read_text(encoding="utf-8"), "same = Мир\n")
                self.assertEqual(
                    any("ready = Привет" in item["content"] for item in client.calls[0]),
                    not save_tokens,
                )
                self.write(paths[0], "ready = Привет\nsame = Hello\n")
                self.write(paths[1], "same = World\n")

    async def test_term_returns_to_its_original_file(self):
        path = self.target / "terms.ftl"
        self.write(path, "-item = Hello\n")
        client = FakeClient()
        with patch("ss14_localization.translate.OpenAICompatibleClient", return_value=client):
            result = await translate_files(
                [path],
                "Prompt",
                0,
                target_culture="ru-RU",
                checker=self.checker,
                save_tokens=True,
            )
        self.assertEqual((result.translated_messages, result.changed_files), (1, 1))
        self.assertIn("-translation-batch-0-item = Hello", client.calls[0][-1]["content"])
        self.assertEqual(path.read_text(encoding="utf-8"), "-item = Привет\n")

    async def test_partial_chunks_preserved_and_failure_reported(self):
        class FailSecond(FakeClient):
            async def chat(self, messages):
                if self.calls:
                    raise RuntimeError("provider down")
                return await super().chat(messages)

        path = self.target / "a.ftl"
        self.write(path, "a = Hello\nb = World\n")
        from ss14_localization.translate import TranslationFileError

        with self.assertRaises(TranslationFileError):
            await translate_file(
                path,
                FailSecond(),
                "Prompt",
                10,
                target_culture="ru-RU",
                checker=self.checker,
                allow_partial=True,
            )
        text = path.read_text(encoding="utf-8")
        self.assertIn("a = Привет", text)
        self.assertIn("b = World", text)

    def test_validation_detects_wrong_path_order_and_language(self):
        self.write(self.source / "a.ftl", "a = Hello\nb = World\n")
        self.write(self.target / "a.ftl", "b = World\na = Привет\n")
        result = validate_locale(self.source, self.target, self.checker)
        self.assertTrue(result.has_errors)
        self.assertGreater(result.untranslated_messages, 0)


class ApiTests(unittest.IsolatedAsyncioTestCase):
    def client(self, handler, attempts=3, on_retry=None):
        httpx = import_or_install("httpx", "httpx>=0.27,<1")
        client = OpenAICompatibleClient(
            AiConfig(
                (AiEndpoint("http://test/v1", "manual-model", "test-key"),),
                max_attempts=attempts,
                cooldown_seconds=0,
            ),
            on_retry=on_retry,
        )
        transport = httpx.MockTransport(handler)
        client.__dict__["_httpx"] = SimpleNamespace(
            HTTPError=httpx.HTTPError,
            AsyncClient=lambda **kwargs: httpx.AsyncClient(transport=transport, **kwargs),
        )
        return client

    async def test_raw_http_content_and_manual_output_limit(self):
        httpx = import_or_install("httpx", "httpx>=0.27,<1")
        import json

        captured = []

        def handler(request):
            payload = json.loads(request.content)
            captured.append(payload)
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "a = Привет"}, "finish_reason": "stop"}]
                },
            )

        result = await self.client(handler).chat([{"role": "user", "content": "a = Hello"}])
        self.assertEqual(result, "a = Привет")
        self.assertEqual(captured[0]["messages"][0]["content"], "a = Hello")
        self.assertEqual(captured[0]["max_tokens"], 16384)

    async def test_luna_uses_completion_limit_without_temperature(self):
        httpx = import_or_install("httpx", "httpx>=0.27,<1")
        import json

        captured = []
        client = self.client(
            lambda request: (
                captured.append(json.loads(request.content))
                or httpx.Response(200, json={"choices": [{"message": {"content": "a = Привет"}}]})
            )
        )
        client._config.endpoints[0].model = "gpt-5.6-luna"
        await client.chat([{"role": "user", "content": "a = Hello"}])
        self.assertEqual(captured[0]["max_completion_tokens"], 16384)
        self.assertEqual(captured[0]["reasoning_effort"], "none")
        self.assertNotIn("temperature", captured[0])

    async def test_rate_limit_and_temporary_failures_retry(self):
        httpx = import_or_install("httpx", "httpx>=0.27,<1")
        calls = []
        retries = []

        def handler(request):
            calls.append(request)
            if len(calls) < 3:
                return httpx.Response(429 if len(calls) == 1 else 503)
            return httpx.Response(200, json={"choices": [{"message": {"content": "a = Привет"}}]})

        self.assertEqual(
            await self.client(handler, on_retry=lambda *items: retries.append(items)).chat([]),
            "a = Привет",
        )
        self.assertEqual(len(calls), 3)
        self.assertEqual([item[1] for item in retries], [1, 2])
        self.assertTrue(all(item[5] for item in retries))

    async def test_truncated_response_is_not_saved_or_retried_at_same_size(self):
        httpx = import_or_install("httpx", "httpx>=0.27,<1")
        client = self.client(
            lambda request: httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"content": "a = incomplete"},
                            "finish_reason": "length",
                        }
                    ]
                },
            )
        )
        with self.assertRaises(ResponseTruncatedError) as caught:
            await client.chat([])
        self.assertEqual(caught.exception.partial_response, "a = incomplete")

    async def test_authentication_error_is_terminal(self):
        httpx = import_or_install("httpx", "httpx>=0.27,<1")
        calls = []

        def handler(request):
            calls.append(request)
            return httpx.Response(401)

        with self.assertRaises(RuntimeError):
            await self.client(handler).chat([])
        self.assertEqual(len(calls), 1)


class ConfigurationTests(Fixture):
    def test_token_saving_is_default_and_can_be_disabled(self):
        import argparse

        from ss14_localization.cli import _translation_options

        parser = argparse.ArgumentParser()
        _translation_options(parser)
        self.assertTrue(parser.parse_args([]).save_tokens)
        self.assertFalse(parser.parse_args(["--no-save-tokens"]).save_tokens)

    def test_direct_translation_cannot_overwrite_source_locale(self):
        import subprocess
        import sys

        path = self.source / "a.ftl"
        self.write(path, "a = Hello\n")
        before = path.read_bytes()
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                "run.py",
                "--repo-root",
                str(self.root),
                "translate",
                str(path),
                "--dry-run",
            ],
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(path.read_bytes(), before)

    def test_env_languages_and_cli_dry_run_are_read_without_network(self):
        import subprocess
        import sys

        self.write(self.source / "a.ftl", "a = Hello world\n")
        env_file = self.root / "test.env"
        env_file.write_text(
            "TRANSLATE_SOURCE_CULTURE=en-US\nTRANSLATE_TARGET_CULTURE=fr-FR\n",
            encoding="utf-8",
        )
        environment = {
            key: value for key, value in os.environ.items() if not key.startswith("TRANSLATE_")
        }
        report = self.root / "report.json"
        command = [
            sys.executable,
            "-B",
            "run.py",
            "--env-file",
            str(env_file),
            "--repo-root",
            str(self.root),
            "translate-all",
            "--target-root",
            str(self.target),
            "--dry-run",
            "--report-json",
            str(report),
        ]
        result = subprocess.run(
            command,
            env=environment,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("a", result.stdout)
        self.assertFalse(report.exists())
        self.assertFalse((self.target / "a.ftl").exists())

    def test_cli_real_http_transport_with_local_stub(self):
        import json
        import subprocess
        import sys
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        self.write(
            self.source / "a.ftl",
            "# Keep\na = Hello { $user }\n    .desc = Hello World\ngun = Desert Eagle\n",
        )
        original = (self.source / "a.ftl").read_bytes()
        requests = []
        contexts = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                text = payload["messages"][-1]["content"]
                requests.append(text)
                contexts.append("\n".join(item["content"] for item in payload["messages"][1:-1]))
                translated = text.replace("Hello", "Привет").replace("World", "Мир")
                body = json.dumps(
                    {
                        "choices": [
                            {
                                "message": {"content": translated},
                                "finish_reason": "stop",
                            }
                        ]
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            environment = {
                key: value for key, value in os.environ.items() if not key.startswith("TRANSLATE_")
            }
            environment.update(
                TRANSLATE_AI_BASE_URL=f"http://127.0.0.1:{server.server_port}/v1",
                TRANSLATE_AI_MODEL="local-test-only",
                TRANSLATE_AI_KEYS="local",
                TRANSLATE_SOURCE_CULTURE="en-US",
                TRANSLATE_TARGET_CULTURE="ru-RU",
                TRANSLATE_AI_MAX_ATTEMPTS="1",
                TRANSLATE_AI_RESPONSE_MAX_ATTEMPTS="1",
                PYTHONUTF8="1",
            )
            report = self.root / "result.json"
            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "run.py",
                    "--repo-root",
                    str(self.root),
                    "translate-all",
                    "--report-json",
                    str(report),
                ],
                env=environment,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual((self.source / "a.ftl").read_bytes(), original)
            target = (self.target / "a.ftl").read_text(encoding="utf-8")
            self.assertIn("Привет", target)
            self.assertIn("Desert Eagle", target)
            self.assertIn("{ $user }", target)
            self.assertGreater(len(requests), 0)
            self.assertTrue(all(not text.startswith("[") for text in requests))
            self.assertFalse(any("gun = Desert Eagle" in context for context in contexts))
            self.assertEqual(json.loads(report.read_text(encoding="utf-8"))["failed_files"], [])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_platform_launchers_dry_run(self):
        import subprocess
        import sys

        self.write(self.source / "a.ftl", "a = Hello\n")
        environment = os.environ.copy()
        environment["SS14_REPO_ROOT"] = str(self.root)
        scripts = (
            [
                (str(Path("scripts/windows/translate.cmd")), "--dry-run"),
                (str(Path("scripts/windows/translate-all-ru.cmd")), "-DryRun"),
            ]
            if sys.platform == "win32"
            else [
                ("bash", "scripts/linux/translate.sh", "--dry-run"),
                ("bash", "scripts/linux/translate-all-ru.sh", "--dry-run"),
            ]
        )
        for command in scripts:
            result = subprocess.run(
                command,
                env=environment,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertFalse((self.target / "a.ftl").exists())


class ArtifactTests(unittest.TestCase):
    def test_workflow_yaml_and_bash_are_valid_and_dry_run_cannot_publish(self):
        import subprocess

        module = import_or_install("ruamel.yaml", "ruamel.yaml>=0.18,<1")
        workflow = module.YAML(typ="safe").load(
            Path("examples/github-actions/auto-translate.yml").read_text(encoding="utf-8")
        )
        self.assertIn("schedule", workflow["on"])
        self.assertTrue(workflow["on"]["workflow_dispatch"]["inputs"]["dry_run"]["default"])
        steps = workflow["jobs"]["translate"]["steps"]
        for step in steps:
            if "run" in step:
                result = subprocess.run(
                    ["bash", "-n", "-c", step["run"]],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
        publisher = next(step for step in steps if "create-pull-request" in step.get("uses", ""))
        self.assertIn("DRY_RUN", publisher["if"])
        self.assertEqual(publisher["with"]["add-paths"], "Resources/Locale")

    def test_documentation_local_links_exist(self):
        import re

        for path in Path(".").glob("*.md"):
            for destination in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
                if not destination.startswith(("http:", "https:", "#")):
                    self.assertTrue(
                        (path.parent / destination.split("#")[0]).exists(),
                        f"{path}: {destination}",
                    )


if __name__ == "__main__":
    unittest.main()
