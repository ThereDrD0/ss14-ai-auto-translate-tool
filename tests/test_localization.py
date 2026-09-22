from __future__ import annotations

import asyncio
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ss14_localization.ai import AiConfig, AiEndpoint, OpenAICompatibleClient, ResponseTruncatedError
from ss14_localization.dependencies import import_or_install
from ss14_localization.budget import OutputBudget
from ss14_localization.fluent import (FluentSyntaxError, entries, message_map, parse_resource,
                                     pattern_text, serialize_entry, serialize_resource, syntax)
from ss14_localization.language import LanguageChecker, PassList, load_pass_list
from ss14_localization.strings import prepare_target_files
from ss14_localization.translate import (_chunks, _parse_translation_response, _safe_chunk,
                                         _translate_chunk, translate_file, translate_files)
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
        self.write(self.source / "a.ftl", "a = Hello\n    .desc = Description\n    .suffix = Label\n-term = Term\n")
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
        self.write(self.source / "a.ftl", '### Group\n# Comment\na = { $count ->\n    [one] One\n   *[other] Many\n}\n')
        self.prepare()
        self.assertIn("### Group", (self.target / "a.ftl").read_text(encoding="utf-8"))
        parse_resource((self.target / "a.ftl").read_text(encoding="utf-8"))


class SyntaxTests(unittest.TestCase):
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
        parse_resource("#comment without a space\n#### comment with extra hashes\na = { \"\\{\" }\n")
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
        for source, target in (("a = Hello\nb = World", "a = Привет"),
                               ("a = Hello", "a = Привет\nb = Лишнее"),
                               ("a = { original }", "a = { changed }"),
                               ('a = { NUMBER($n, minimumFractionDigits: 2) }',
                                'a = { NUMBER($n, minimumFractionDigits: 3) }'),
                               ("a = [color=red]Hello[/color]", "a = [color=blue]Привет[/color]"),
                               ("# Same\na = Hello", "# Changed\na = Привет")):
            with self.subTest(source=source), self.assertRaises(ValueError):
                self.check(source, target)

    def test_visible_literal_can_be_translated_technical_literal_cannot(self):
        self.check('a = { "Hello" }', 'a = { "Привет" }')
        with self.assertRaises(ValueError):
            self.check('a = { GENDER($user, form: "male") }', 'a = { GENDER($user, form: "мужской") }')

    def test_terms_and_select_variants(self):
        self.check("-term = Hello", "-term = Привет")
        with self.assertRaises(ValueError):
            self.check("a = { $n ->\n [one] Hello\n *[other] World\n}",
                       "a = { $n ->\n [few] Привет\n *[other] Мир\n}")


class LanguageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.passed = load_pass_list()
        cls.ru = LanguageChecker("en-US", "ru-RU", cls.passed)
        cls.fr = LanguageChecker("en-US", "fr-FR", cls.passed)

    def test_any_supported_locale_not_just_ru_en(self):
        self.assertEqual(self.ru.minimum_ratio, .5)
        self.assertEqual(self.fr.minimum_ratio, .8)
        self.assertEqual(self.fr.ratio("Bonjour tout le monde"), 1)
        self.assertEqual(self.fr.ratio("Hello world"), 0)
        de = LanguageChecker("fr-FR", "de-DE", self.passed)
        self.assertEqual(de.ratio("Hallo Welt"), 1)

    def test_short_uppercase_and_wrong_script_not_ignored(self):
        for value in ("On", "HELLO WORLD", "你好世界"):
            with self.subTest(value=value):
                self.assertLess(self.ru.ratio(value), .8)

    def test_pass_word_not_whole_line(self):
        self.assertEqual(self.ru.ratio("Desert Eagle"), 1)
        self.assertEqual(self.ru.ratio("M1 Garand"), 1)
        self.assertEqual(self.ru.ratio("AI APC DNA GPS PDA UI NT IDs"), 1)
        self.assertLess(self.ru.ratio("Desert Eagle Hello world"), .8)
        self.assertEqual(self.ru.ratio("Desert Eagle — мощное оружие"), 1)
        self.assertLess(self.ru.ratio("[Hello World]"), .8)

    def test_language_share_is_combined_across_fields(self):
        text = ("ent-WeaponSubMachineGunSP91RC = SP-91-RC\n"
                "    .desc = Компактный пистолет-пулемёт для контроля беспорядков.\n")
        node = entries(parse_resource(text))["ent-WeaponSubMachineGunSP91RC"]
        self.assertEqual(self.ru.ratio("SP-91-RC"), 1)
        self.ru.validate(node)
        self.assertFalse(self.ru.needs_translation(node))
        english = entries(parse_resource("name = SP-91-RC\n    .desc = Compact submachine gun.\n"))["name"]
        self.assertTrue(self.ru.needs_translation(english))
        self.assertLess(self.ru.ratio("HELLO WORLD"), .5)

    def test_model_code_exclusion_does_not_eat_prose_or_partial_codes(self):
        self.assertEqual(self.ru.ratio("LSE-400b"), 1)
        self.assertIn("LSE-400bc", self.ru.pass_list.strip("LSE-400bc"))
        self.assertIn("Hello2", self.ru.pass_list.strip("Hello2"))
        self.assertLess(self.ru.ratio("SP-91-RC Compact submachine gun"), .5)
        self.assertLess(self.ru.ratio("LSE-400b English description"), .5)

    def test_response_language_share_is_combined_across_messages(self):
        source = "code = SP-91-RC\ndescription = Compact submachine gun.\n"
        translated = "code = SP-91-RC\ndescription = Компактный пистолет-пулемёт для контроля беспорядков.\n"
        self.assertEqual(set(_parse_translation_response(translated, message_map(source), checker=self.ru)),
                         {"code", "description"})
        with self.assertRaises(ValueError):
            _parse_translation_response(source, message_map(source), checker=self.ru)

    def test_pass_boundaries_case_and_preservation(self):
        passed = PassList(("ID", "Desert Eagle"))
        self.assertEqual(passed.strip("identity"), "identity")
        with self.assertRaises(ValueError):
            passed.assert_preserved("Desert Eagle", "desert eagle")

    def test_select_variant_prose_is_checked(self):
        node = entries(parse_resource("a = { $n ->\n [one] Hello\n *[other] World\n}"))["a"]
        self.assertTrue(self.ru.needs_translation(node))

    def test_already_translated_attribute_cannot_be_rewritten(self):
        source = "a = Hello\n    .desc = Описание предмета\n"
        with self.assertRaises(ValueError):
            _parse_translation_response("a = Привет\n    .desc = Новое описание\n",
                                        message_map(source), checker=self.ru)

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

    def test_fifty_percent_russian_threshold_and_markup_exclusions(self):
        mixed = "Привет привет Hello world"
        self.assertGreaterEqual(self.ru.ratio(mixed), .5)
        self.assertLess(self.ru.ratio(mixed), .8)
        self.assertFalse(self.ru.needs_translation(entries(parse_resource(f"english-key-name = {mixed}"))["english-key-name"]))
        decorated = '[bold][BubbleHeader]Привет привет[/BubbleHeader][/bold] [tutkeybind="UIClick"] Hello world 123 !?'
        self.assertEqual(self.ru.ratio(decorated), self.ru.ratio(mixed))
        self.assertEqual(self.ru.pass_list.strip('[Name]Привет[/Name] [BubbleContent]мир[/BubbleContent]').split(),
                         ["Привет", "мир"])

    def test_mixed_scripts_ratio_is_independent_of_threshold(self):
        self.assertGreaterEqual(self.ru.ratio("Это русское описание игрового предмета на космической станции Hello"), .8)
        self.assertLess(self.ru.ratio("Привет Hello world this is an English description"), .8)


class BudgetTests(unittest.TestCase):
    def test_chunks_use_manual_output_budget_with_margin(self):
        budget = OutputBudget(192, .65, 3, 16)
        messages = list(message_map("a = Hello world\nb = Hello world\nc = Hello world\n").values())
        chunks = _chunks(messages, 10000, budget)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(budget.estimated_output("\n\n".join(m.text for m in chunk)), budget.capacity)
        self.assertGreater(budget.tokens("Привет"), len("Привет"))

    def test_long_text_splits_without_losing_source(self):
        budget = OutputBudget(256, .65, 2, 16)
        render = lambda value: serialize_entry(__import__("ss14_localization.translate", fromlist=["_fragment_node"])._fragment_node(value))
        text = "Hello world " * 30
        parts = budget.split_text(text, render)
        self.assertGreater(len(parts), 1)
        self.assertEqual("".join(parts), text)
        self.assertTrue(all(budget.fits(render(part)) for part in parts))

    def test_invalid_budget_rejected(self):
        for kwargs in ({"max_tokens": 0}, {"safety": 1}, {"expansion": .5}, {"reserve": 10000}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                OutputBudget(**kwargs)


class FakeClient:
    def __init__(self, invalid_first=False, truncated_first=False):
        self.calls = []
        self.invalid_first = invalid_first
        self.truncated_first = truncated_first

    async def chat(self, messages):
        self.calls.append(messages)
        if len(self.calls) == 1 and self.truncated_first:
            raise ResponseTruncatedError("length")
        if len(self.calls) == 1 and self.invalid_first:
            return "a = Привет {"
        payload = messages[-2]["content"] if messages[-1]["content"].startswith("Предыдущая попытка") else messages[-1]["content"]
        resource = parse_resource(payload)
        ast = syntax().ast
        from ss14_localization.translate import _text_slots
        for node in entries(resource).values():
            for slot in _text_slots(node):
                slot.value = slot.value.replace("Hello", "Привет").replace("World", "Мир")
        return serialize_resource(resource)


class TranslationTests(Fixture, unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        super().setUp()
        self.checker = LanguageChecker("en-US", "ru-RU", load_pass_list())

    async def test_token_saving_only_sends_untranslated_keys(self):
        for save_tokens in (False, True):
            with self.subTest(save_tokens=save_tokens):
                path = self.target / "mixed.ftl"
                self.write(path, "ready = Привет\nmissing = Hello\n")
                client = FakeClient()
                count, changed = await translate_file(path, client, "Prompt", 4000,
                                                      target_culture="ru-RU", checker=self.checker,
                                                      save_tokens=save_tokens)
                self.assertEqual((count, changed), (1, True))
                self.assertEqual(path.read_text(encoding="utf-8"), "ready = Привет\nmissing = Привет\n")
                sent = [item["content"] for item in client.calls[0] if item["role"] == "user"]
                self.assertEqual(sent[-1], "missing = Hello")
                self.assertEqual(any("ready = Привет" in item for item in sent), not save_tokens)

    async def test_think_preamble_is_removed_before_budget_and_file_write(self):
        class ThinkingClient:
            async def chat(self, messages):
                return "<think>" + "рассуждение " * 1000 + "</think>a = Привет\n    .desc = Описание"

        path = self.target / "a.ftl"
        self.write(path, "a = Hello\n    .desc = Description\n")
        count, changed = await translate_file(path, ThinkingClient(), "Prompt", 4000,
                                              target_culture="ru-RU", checker=self.checker)
        self.assertEqual((count, changed), (1, True))
        self.assertEqual(path.read_text(encoding="utf-8"), "a = Привет\n    .desc = Описание\n")

    async def test_raw_request_and_retry_feedback(self):
        client = FakeClient(invalid_first=True)
        result = await _translate_chunk(client, "Prompt", list(message_map("a = Hello").values()),
                                        "ru-RU", self.checker)
        self.assertIn("Привет", result["a"])
        self.assertEqual(client.calls[0][1]["content"], "a = Hello")
        self.assertEqual(len(client.calls[1]), 3)
        self.assertIn("Предыдущая попытка", client.calls[1][-1]["content"])

    async def test_truncation_reduces_chunk(self):
        client = FakeClient(truncated_first=True)
        result = await _safe_chunk(client, "Prompt", list(message_map("a = Hello\nb = World").values()),
                                   self.checker, OutputBudget())
        self.assertEqual(set(result), {"a", "b"})
        self.assertEqual(len(client.calls), 3)

    async def test_single_message_truncation_uses_text_fragments(self):
        client = FakeClient(truncated_first=True)
        result = await _safe_chunk(client, "Prompt", list(message_map("a = Hello").values()),
                                   self.checker, OutputBudget())
        self.assertIn("Привет", result["a"])
        self.assertIn("translation-part", client.calls[1][1]["content"])

    async def test_large_single_message_fragmented_every_request_fits(self):
        path = self.target / "a.ftl"
        self.write(path, "a = " + "Hello " * 50 + "\n")
        client = FakeClient()
        budget = OutputBudget(512, .65, 2, 16)
        count, changed = await translate_file(path, client, "Prompt", 10000, target_culture="ru-RU",
                                              checker=self.checker, budget=budget)
        self.assertEqual(count, 1)
        self.assertTrue(changed)
        self.assertGreater(len(client.calls), 1)
        self.assertTrue(all(budget.fits(call[1]["content"], "Prompt") for call in client.calls))
        self.assertNotIn("Hello", path.read_text(encoding="utf-8"))
        parse_resource(path.read_text(encoding="utf-8"))

    async def test_dry_run_no_api_no_write_including_virtual_file(self):
        path = self.target / "new.ftl"
        with patch("ss14_localization.translate.OpenAICompatibleClient") as client, redirect_stdout(io.StringIO()):
            result = await translate_files([path], "Prompt", 4000, target_culture="ru-RU", dry_run=True,
                                           checker=self.checker, texts={path: "a = Hello\n"})
        client.assert_not_called()
        self.assertFalse(path.exists())
        self.assertEqual(result.failed_files, ())

    async def test_large_visible_string_literal_preserves_literal_braces(self):
        path = self.target / "a.ftl"
        self.write(path, 'a = { "' + "Hello { " * 40 + '" }\n')
        client = FakeClient()
        count, changed = await translate_file(path, client, "Prompt", 10000, target_culture="ru-RU",
                                              checker=self.checker, budget=OutputBudget(512, .65, 2, 16))
        self.assertEqual(count, 1)
        self.assertTrue(changed)
        result = entries(parse_resource(path.read_text(encoding="utf-8")))["a"]
        self.assertEqual(pattern_text(result.value).count("{"), 40)
        self.assertNotIn("Hello", pattern_text(result.value))

    async def test_async_concurrency_preserved(self):
        active = peak = 0
        checker = self.checker
        class ConcurrentClient(FakeClient):
            async def chat(inner, messages):
                nonlocal active, peak
                active += 1
                peak = max(peak, active)
                await asyncio.sleep(.02)
                result = await super().chat(messages)
                active -= 1
                return result
        paths = [self.target / f"{i}.ftl" for i in range(4)]
        for i, path in enumerate(paths):
            self.write(path, f"a{i} = Hello\n")
        with patch("ss14_localization.translate.OpenAICompatibleClient", return_value=ConcurrentClient()):
            result = await translate_files(paths, "Prompt", 4000, target_culture="ru-RU",
                                           concurrency=2, checker=checker)
        self.assertEqual(peak, 2)
        self.assertEqual(result.changed_files, 4)

    async def test_partial_chunks_preserved_and_failure_reported(self):
        class FailSecond(FakeClient):
            async def chat(inner, messages):
                if inner.calls:
                    raise RuntimeError("provider down")
                return await super().chat(messages)
        path = self.target / "a.ftl"
        self.write(path, "a = Hello\nb = World\n")
        from ss14_localization.translate import TranslationFileError
        with self.assertRaises(TranslationFileError):
            await translate_file(path, FailSecond(), "Prompt", 10, target_culture="ru-RU",
                                 checker=self.checker, allow_partial=True)
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
        client = OpenAICompatibleClient(AiConfig((AiEndpoint("http://test/v1", "manual-model", "test-key"),),
                                                max_attempts=attempts, cooldown_seconds=0), on_retry=on_retry)
        transport = httpx.MockTransport(handler)
        client._httpx = SimpleNamespace(HTTPError=httpx.HTTPError,
                                      AsyncClient=lambda **kwargs: httpx.AsyncClient(transport=transport, **kwargs))
        return client

    async def test_raw_http_content_and_manual_output_limit(self):
        httpx = import_or_install("httpx", "httpx>=0.27,<1")
        import json
        captured = []
        def handler(request):
            payload = json.loads(request.content)
            captured.append(payload)
            return httpx.Response(200, json={"choices": [{"message": {"content": "a = Привет"}, "finish_reason": "stop"}]})
        result = await self.client(handler).chat([{"role": "user", "content": "a = Hello"}])
        self.assertEqual(result, "a = Привет")
        self.assertEqual(captured[0]["messages"][0]["content"], "a = Hello")
        self.assertEqual(captured[0]["max_tokens"], 8192)

    async def test_rate_limit_and_temporary_failures_retry(self):
        httpx = import_or_install("httpx", "httpx>=0.27,<1")
        calls = []
        retries = []
        def handler(request):
            calls.append(request)
            if len(calls) < 3:
                return httpx.Response(429 if len(calls) == 1 else 503)
            return httpx.Response(200, json={"choices": [{"message": {"content": "a = Привет"}}]})
        self.assertEqual(await self.client(handler, on_retry=lambda *items: retries.append(items)).chat([]), "a = Привет")
        self.assertEqual(len(calls), 3)
        self.assertEqual([item[1] for item in retries], [1, 2])
        self.assertTrue(all(item[5] for item in retries))

    async def test_truncated_response_is_not_saved_or_retried_at_same_size(self):
        httpx = import_or_install("httpx", "httpx>=0.27,<1")
        client = self.client(lambda request: httpx.Response(200, json={"choices": [
            {"message": {"content": "a = incomplete"}, "finish_reason": "length"}]}))
        with self.assertRaises(ResponseTruncatedError):
            await client.chat([])

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
    def test_direct_translation_cannot_overwrite_source_locale(self):
        import subprocess
        import sys
        path = self.source / "a.ftl"
        self.write(path, "a = Hello\n")
        before = path.read_bytes()
        result = subprocess.run([sys.executable, "-B", "run.py", "--repo-root", str(self.root),
                                 "translate", str(path), "--dry-run"], capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(path.read_bytes(), before)

    def test_env_languages_and_cli_dry_run_are_read_without_network(self):
        import subprocess
        import sys
        self.write(self.source / "a.ftl", "a = Hello world\n")
        env_file = self.root / "test.env"
        env_file.write_text("TRANSLATE_SOURCE_CULTURE=en-US\nTRANSLATE_TARGET_CULTURE=fr-FR\n", encoding="utf-8")
        environment = {key: value for key, value in os.environ.items() if not key.startswith("TRANSLATE_")}
        report = self.root / "report.json"
        command = [sys.executable, "-B", "run.py", "--env-file", str(env_file), "--repo-root", str(self.root),
                   "translate-all", "--target-root", str(self.target), "--dry-run", "--report-json", str(report)]
        result = subprocess.run(command, env=environment, capture_output=True, encoding="utf-8", errors="replace")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("a", result.stdout)
        self.assertFalse(report.exists())
        self.assertFalse((self.target / "a.ftl").exists())

    def test_cli_real_http_transport_with_local_stub(self):
        import json
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        import subprocess
        import sys
        import threading
        self.write(self.source / "a.ftl", "# Keep\na = Hello { $user }\n    .desc = Hello World\ngun = Desert Eagle\n")
        original = (self.source / "a.ftl").read_bytes()
        requests = []
        contexts = []
        class Handler(BaseHTTPRequestHandler):
            def do_POST(inner):
                payload = json.loads(inner.rfile.read(int(inner.headers["Content-Length"])))
                text = payload["messages"][-1]["content"]
                requests.append(text)
                contexts.append("\n".join(item["content"] for item in payload["messages"][1:-1]))
                translated = text.replace("Hello", "Привет").replace("World", "Мир")
                body = json.dumps({"choices": [{"message": {"content": translated}, "finish_reason": "stop"}]}).encode()
                inner.send_response(200)
                inner.send_header("Content-Type", "application/json")
                inner.send_header("Content-Length", str(len(body)))
                inner.end_headers()
                inner.wfile.write(body)
            def log_message(inner, *args):
                pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            environment = {key: value for key, value in os.environ.items() if not key.startswith("TRANSLATE_")}
            environment.update(TRANSLATE_AI_BASE_URL=f"http://127.0.0.1:{server.server_port}/v1",
                               TRANSLATE_AI_MODEL="local-test-only", TRANSLATE_AI_KEYS="local",
                               TRANSLATE_SOURCE_CULTURE="en-US", TRANSLATE_TARGET_CULTURE="ru-RU",
                               TRANSLATE_AI_MAX_ATTEMPTS="1", TRANSLATE_AI_RESPONSE_MAX_ATTEMPTS="1",
                               PYTHONUTF8="1")
            report = self.root / "result.json"
            result = subprocess.run([sys.executable, "-B", "run.py", "--repo-root", str(self.root),
                                     "translate-all", "--report-json", str(report)],
                                    env=environment, capture_output=True, encoding="utf-8", errors="replace", timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual((self.source / "a.ftl").read_bytes(), original)
            target = (self.target / "a.ftl").read_text(encoding="utf-8")
            self.assertIn("Привет", target)
            self.assertIn("Desert Eagle", target)
            self.assertIn("{ $user }", target)
            self.assertGreater(len(requests), 0)
            self.assertTrue(all(not text.startswith("[") for text in requests))
            self.assertTrue(any("gun = Desert Eagle" in context for context in contexts))
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
        scripts = ([(str(Path("scripts/windows/translate.cmd")), "--dry-run"),
                    (str(Path("scripts/windows/translate-all-ru.cmd")), "-DryRun")]
                   if sys.platform == "win32" else
                   [("bash", "scripts/linux/translate.sh", "--dry-run"),
                    ("bash", "scripts/linux/translate-all-ru.sh", "--dry-run")])
        for command in scripts:
            result = subprocess.run(command, env=environment, capture_output=True, encoding="utf-8", errors="replace", timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertFalse((self.target / "a.ftl").exists())


class ArtifactTests(unittest.TestCase):
    def test_workflow_yaml_and_bash_are_valid_and_dry_run_cannot_publish(self):
        import subprocess
        module = import_or_install("ruamel.yaml", "ruamel.yaml>=0.18,<1")
        workflow = module.YAML(typ="safe").load(Path("examples/github-actions/auto-translate.yml").read_text(encoding="utf-8"))
        self.assertIn("schedule", workflow["on"])
        self.assertTrue(workflow["on"]["workflow_dispatch"]["inputs"]["dry_run"]["default"])
        steps = workflow["jobs"]["translate"]["steps"]
        for step in steps:
            if "run" in step:
                result = subprocess.run(["bash", "-n", "-c", step["run"]], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
        publisher = next(step for step in steps if "create-pull-request" in step.get("uses", ""))
        self.assertIn("DRY_RUN", publisher["if"])
        self.assertEqual(publisher["with"]["add-paths"], "Resources/Locale")

    def test_documentation_local_links_exist(self):
        import re
        for path in Path(".").glob("*.md"):
            for destination in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
                if not destination.startswith(("http:", "https:", "#")):
                    self.assertTrue((path.parent / destination.split("#")[0]).exists(), f"{path}: {destination}")


if __name__ == "__main__":
    unittest.main()
