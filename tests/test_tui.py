from __future__ import annotations

import json
import os
import tempfile
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from unittest.mock import patch

from textual.color import Color
from textual.widgets import Checkbox, Input, OptionList, RichLog, Static

from ss14_localization.tui import (
    TokenEta,
    _cache_path,
    _diff_lines,
    _inventory,
    _load_cache,
    create_app,
    summary_counts,
)


class TuiTests(unittest.IsolatedAsyncioTestCase):
    def test_eta_uses_finished_file_time_and_remaining_token_sizes(self):
        small, large, larger = (Path(name) for name in ("small.ftl", "large.ftl", "larger.ftl"))
        eta = TokenEta({small: 10, large: 100, larger: 100}, concurrency=2)
        self.assertIsNone(eta.remaining(0))
        eta.start(small, 0)
        eta.start(large, 0)
        eta.finish(small, 4)
        eta.start(larger, 4)
        remaining = eta.remaining(6)
        assert remaining is not None
        self.assertAlmostEqual(remaining, 36)
        remaining = eta.remaining(100)
        assert remaining is not None
        self.assertGreater(remaining, 0)
        eta.finish(large, 8)
        eta.finish(larger, 9)
        self.assertEqual(eta.remaining(9), 0)

    async def test_preparation_eta_uses_completed_file_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            source = repo / "Resources" / "Locale" / "en-US"
            source.mkdir(parents=True)
            (source / "a.ftl").write_text("a = Hello\n", encoding="utf-8")
            app = create_app(repo)
            async with app.run_test():
                app.phase = "work"
                app._new_stage("Подготовка", 10)
                app.done = 2
                app.stage_started = 5
                with patch("ss14_localization.tui.monotonic", return_value=10):
                    app._tick()
                self.assertIn("Осталось: ~20 с", str(app.query_one("#eta", Static).render()))

    def test_no_arguments_open_tui(self):
        from ss14_localization.cli import main

        with patch("ss14_localization.tui.run", return_value=0) as launch:
            self.assertEqual(main([]), 0)
        launch.assert_called_once_with()

    def test_inventory_detects_same_size_edit(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "en-US"
            target = Path(temporary) / "ru-RU"
            source.mkdir()
            target.mkdir()
            (source / "a.ftl").write_text("a = Hello\n", encoding="utf-8")
            file = target / "a.ftl"
            file.write_text("a = Привет\n", encoding="utf-8")
            before = _inventory(source, target)
            file.write_text("a = Прощай\n", encoding="utf-8")
            after = _inventory(source, target)
            self.assertNotEqual(before[0], after[0])
            self.assertNotEqual(before[3]["a.ftl"], after[3]["a.ftl"])

    def test_old_cache_is_invalidated_for_entity_formatting(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cache.json"
            path.write_text('{"version": 4, "prepared": "old"}', encoding="utf-8")
            self.assertEqual(_load_cache(path), {})

    def test_diff_keeps_line_numbers_indent_and_marks_changed_words(self):
        rows = _diff_lines(
            "a = Hello\n    .desc = Old word\n", "a = Привет\n    .desc = New word\n"
        )
        self.assertEqual(
            [row.plain.split()[-1] for row in rows], ["Hello", "Привет", "word", "word"]
        )
        self.assertTrue(rows[2].plain.startswith("   2 -     .desc"))
        self.assertTrue(rows[3].plain.startswith("   2 +     .desc"))
        self.assertIn("    .desc", rows[2].plain)
        for row, word in zip(rows, ("Hello", "Привет", "Old", "New")):
            highlighted = [span for span in row.spans if " on " in str(span.style)]
            self.assertEqual(len(highlighted), 1)
            self.assertEqual(row.plain[highlighted[0].start : highlighted[0].end], word)

    def test_diff_does_not_highlight_shifted_line_numbers(self):
        rows = _diff_lines("a\nb\nc\n", "a\nnew\nb\nc\n")
        self.assertEqual([row.plain[:6] for row in rows], ["   1  ", "   2 +", "   3  ", "   4  "])
        self.assertTrue(
            any(span.start == 0 and " on " in str(span.style) for span in rows[1].spans)
        )
        self.assertTrue(
            all(" on " not in str(span.style) for row in rows[2:] for span in row.spans)
        )

    async def test_autoscroll_can_be_paused_while_log_grows(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            source = repo / "Resources" / "Locale" / "en-US"
            source.mkdir(parents=True)
            (source / "a.ftl").write_text("a = Hello\n", encoding="utf-8")
            app = create_app(repo)
            async with app.run_test(size=(80, 25)) as pilot:
                app.phase = "work"
                app.query_one("#choose").display = False
                app.query_one("#work").display = True
                app._new_stage("Перевод", 1)
                log = app.query_one("#log", RichLog)
                log.focus()
                for number in range(80):
                    app._log("ПРОВЕРЕН", detail=f"строка {number}")
                await pilot.pause()
                self.assertGreater(log.scroll_y, 0)
                await pilot.press("f2", "home")
                await pilot.pause(0.5)
                self.assertFalse(log.auto_scroll)
                self.assertEqual(log.scroll_y, 0)
                app._log("ПРОВЕРЕН", detail="новая строка")
                await pilot.pause()
                self.assertEqual(log.scroll_y, 0)
                await pilot.press("f2")
                await pilot.pause()
                self.assertTrue(log.auto_scroll)
                self.assertGreater(log.scroll_y, 0)
                await pilot.press("ctrl+q")
                self.assertTrue(app.is_running)

    async def test_log_diff_opens_by_keyboard_and_mouse(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            source = repo / "Resources" / "Locale" / "en-US"
            source.mkdir(parents=True)
            (source / "a.ftl").write_text("a = Hello\n", encoding="utf-8")
            app = create_app(repo)
            async with app.run_test(size=(80, 25)) as pilot:
                app.phase = "work"
                app.query_one("#choose").display = False
                app.query_one("#work").display = True
                await pilot.pause()
                app._log(
                    "ГОТОВО",
                    source / "a.ftl",
                    before="a = Hello\n",
                    after="a = Привет\n",
                )
                log = app.query_one("#log", RichLog)
                log.focus()
                self.assertIn("▶ изменения", log.lines[0].text)
                await pilot.press("down", "enter")
                self.assertTrue(app.log_items[0][5])
                self.assertIn("Привет", "\n".join(line.text for line in log.lines))
                await pilot.click("#log", offset=(5, 1))
                self.assertFalse(app.log_items[0][5])

    async def test_review_changes_with_file_selection_and_done_button(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            source = repo / "Resources" / "Locale" / "en-US"
            target = source.parent / "ru-RU"
            source.mkdir(parents=True)
            target.mkdir()
            for name in ("a", "b"):
                (source / f"{name}.ftl").write_text(f"{name} = Hello\n", encoding="utf-8")
                (target / f"{name}.ftl").write_text(
                    f"{name} = {'Привет ' * 25}\n", encoding="utf-8"
                )
            app = create_app(repo)
            app.review_files = [target / "a.ftl", target / "b.ftl"]
            app.review_before = {path: f"{path.stem} = Hello\n" for path in app.review_files}
            async with app.run_test(size=(80, 25)) as pilot:
                app._show_review()
                await pilot.pause()
                view = app.query_one("#review-diff", RichLog)
                self.assertTrue(view.wrap)
                self.assertGreater(len(view.lines), 4)
                self.assertIn("a.ftl", "\n".join(line.text for line in view.lines))
                await pilot.press("down")
                self.assertIn("b.ftl", "\n".join(line.text for line in view.lines))
                await pilot.press("tab")
                assert app.focused is not None
                self.assertEqual(app.focused.id, "review-diff")
                await pilot.press("tab")
                assert app.focused is not None
                self.assertEqual(app.focused.id, "review-log")
                await pilot.click("#review-log")
                self.assertEqual(app.phase, "log")
                await pilot.press("f4")
                await pilot.pause()
                self.assertEqual(app.phase, "review")
                await pilot.click("#review-done")
                self.assertEqual(app.phase, "summary")

    async def test_model_screen_token_setting_defaults_on(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            source = repo / "Resources" / "Locale" / "en-US"
            source.mkdir(parents=True)
            (source / "a.ftl").write_text("a = Hello\n", encoding="utf-8")
            app = create_app(repo)
            async with app.run_test() as pilot:
                app.phase = "models"
                app.query_one("#choose").display = False
                app.query_one("#models").display = True
                checkbox = app.query_one("#save-tokens", Checkbox)
                self.assertTrue(checkbox.value)
                self.assertIn("▐✓▌", checkbox.render().plain)
                self.assertEqual(checkbox.styles.background, Color.parse("#111821"))
                await pilot.press("f3")
                self.assertFalse(checkbox.value)
                self.assertIn("▐ ▌", checkbox.render().plain)
                await pilot.press("f3")
                self.assertTrue(checkbox.value)
                self.assertIn("▐✓▌", checkbox.render().plain)

    async def test_selection_translation_retries_and_summary(self):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.assert_path("/v1/models")
                self.reply({"data": [{"id": "first"}, {"id": "test-model"}]})

            def do_POST(self):
                self.assert_path("/v1/chat/completions")
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append(body)
                source = body["messages"][1]["content"]
                import re

                ids = re.findall(r"^(translation-batch-\d+-[ab])\s*=", source, re.MULTILINE)
                content = (
                    "\n".join(f"{key} = Привет" for key in ids)
                    if ids and all(key.endswith("-a") for key in ids)
                    else "broken = {"
                )
                self.reply(
                    {
                        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 13, "completion_tokens": 5},
                    }
                )

            def assert_path(self, expected):
                if self.path != expected:
                    raise AssertionError(self.path)

            def reply(self, data):
                raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, format, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with (
                tempfile.TemporaryDirectory() as temporary,
                patch.dict(
                    os.environ,
                    {
                        "TRANSLATE_AI_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
                        "TRANSLATE_AI_MODEL": "ignored",
                        "TRANSLATE_AI_RESPONSE_MAX_ATTEMPTS": "2",
                        "TRANSLATE_AI_RESPONSE_COOLDOWN_SECONDS": "0",
                        "TRANSLATE_CONCURRENCY": "2",
                    },
                ),
            ):
                repo = Path(temporary)
                source = repo / "Resources" / "Locale" / "en-US"
                source.mkdir(parents=True)
                (source / "a.ftl").write_text("a = Hello\n", encoding="utf-8")
                (source / "b.ftl").write_text("b = Hello\n", encoding="utf-8")
                (source / "c.ftl").write_text("c = Hello\n", encoding="utf-8")
                (source / "empty.ftl").write_text("", encoding="utf-8")
                other = source.parent / "nl-NL"
                other.mkdir()
                (other / "other.ftl").write_text("other = Hallo\n", encoding="utf-8")
                app = create_app(repo)
                target = source.parent / "ru-RU"
                target.mkdir()
                (target / "c.ftl").write_text("c = Привет\n", encoding="utf-8")
                app.error_log_path = repo / "translation-errors.log"
                async with app.run_test() as pilot:
                    await pilot.pause()
                    await pilot.press("down")
                    await pilot.pause()
                    self.assertEqual(app.source, "nl-NL")
                    self.assertNotIn("nl-NL", app.target_options)
                    await pilot.press("up", "tab", "enter")
                    for _ in range(50):
                        await pilot.pause(0.1)
                        if app.model_names:
                            break
                    self.assertEqual(app.model_names, ["first", "test-model"])
                    self.assertEqual(app.query_one("#model-list", OptionList).highlighted, 0)
                    await pilot.press("down", "enter")
                    for _ in range(100):
                        await pilot.pause(0.1)
                        if app.phase in {"review", "failed"}:
                            break
                    await pilot.pause()
                    self.assertEqual(app.phase, "review")
                    self.assertEqual(len(app.review_files), 1)
                    self.assertEqual(app.review_files[0].name, "a.ftl")
                    self.assertTrue(app.query_one("#review-diff", RichLog).wrap)
                    self.assertIn(
                        "Привет",
                        "\n".join(
                            line.text for line in app.query_one("#review-diff", RichLog).lines
                        ),
                    )
                    self.assertTrue(
                        any(item[3] is not None and not item[5] for item in app.log_items)
                    )
                    await pilot.press("f4")
                    self.assertEqual(app.phase, "log")
                    diff_index = next(
                        index
                        for index, item in enumerate(app.log_items)
                        if item[0] == "ГОТОВО" and item[3] != item[4]
                    )
                    app._toggle_log(diff_index)
                    log_text = "\n".join(
                        line.text for line in app.query_one("#log", RichLog).lines
                    )
                    self.assertRegex(log_text, r"1\s+- a = Hello")
                    self.assertIn("1 + a = Привет", log_text)
                    await pilot.press("f4")
                    await pilot.pause()
                    self.assertEqual(app.phase, "review")
                    before_retry = len(requests)
                    await pilot.press("space")
                    await pilot.pause()
                    self.assertEqual(type(app.screen).__name__, "RetranslatePrompt")
                    await pilot.press("space")
                    for _ in range(100):
                        await pilot.pause(0.1)
                        if len(requests) > before_retry and app.phase == "review":
                            break
                    self.assertEqual(app.phase, "review")
                    self.assertGreater(len(requests), before_retry)
                    before_retry = len(requests)
                    await pilot.press("space")
                    await pilot.pause()
                    note = app.screen.query_one("#prompt-input", Input)
                    note.value = "Сделай точнее"
                    note.focus()
                    await pilot.press("enter")
                    for _ in range(100):
                        await pilot.pause(0.1)
                        if len(requests) > before_retry and app.phase == "review":
                            break
                    self.assertEqual(app.phase, "review")
                    self.assertIn("Сделай точнее", requests[-1]["messages"][0]["content"])
                    await pilot.press("ctrl+enter")
                    self.assertEqual(app.phase, "summary")
                    self.assertEqual(app.success, 1)
                    self.assertEqual(app.skipped, 1)
                    self.assertEqual(app.total, 2)
                    self.assertEqual(len(app.failures), 1)
                    self.assertEqual(app.failures[0].path.name, "b.ftl")
                    self.assertEqual(app.prompt_tokens + app.completion_tokens, 126)
                    self.assertEqual(app.retry_tokens, 36)
                    self.assertIn(
                        "ПОВТОР",
                        "\n".join(line.text for line in app.query_one("#log", RichLog).lines),
                    )
                    self.assertEqual(
                        summary_counts(
                            app.success,
                            app.failures,
                            app.skipped,
                            app.prompt_tokens,
                            app.completion_tokens,
                            app.retry_tokens,
                        )["success_percent"],
                        50.0,
                    )
                    self.assertEqual(
                        (source.parent / "ru-RU" / "a.ftl").read_text(encoding="utf-8"),
                        "a = Привет\n",
                    )
                    self.assertEqual(
                        (source.parent / "ru-RU" / "b.ftl").read_text(encoding="utf-8"),
                        "b = Hello\n",
                    )
                    self.assertFalse((source.parent / "ru-RU" / "empty.ftl").exists())
                    self.assertTrue(all(body["model"] == "test-model" for body in requests))
                    self.assertFalse(
                        any(
                            "c =" in item["content"]
                            for body in requests
                            for item in body["messages"]
                        )
                    )
                    error_log = app.error_log_path.read_text(encoding="utf-8")
                    self.assertIn("[ПОВТОР]", error_log)
                    self.assertIn("[ОШИБКА]", error_log)
                    self.assertIn("b.ftl", error_log)
                    self.assertIn("Исходный текст:", error_log)
                    self.assertIn("Исходный блок:", error_log)
                    self.assertIn("Ответ ИИ:\nbroken = {", error_log)
                    await pilot.press("q")
                    self.assertEqual(app.phase, "summary")
                cache = _load_cache(_cache_path(repo, "en-US", "ru-RU"))
                self.assertEqual(cache["prepared"], _inventory(source, source.parent / "ru-RU")[0])
                self.assertIn("a.ftl", cache["verified"])
                self.assertNotIn("b.ftl", cache["verified"])
                with patch(
                    "ss14_localization.strings.prepare_target_files",
                    side_effect=AssertionError("подготовка должна использовать кэш"),
                ):
                    again = create_app(repo)
                    again.error_log_path = repo / "translation-errors.log"
                    async with again.run_test() as pilot:
                        await pilot.press("enter")
                        for _ in range(50):
                            await pilot.pause(0.1)
                            if again.model_names:
                                break
                        await pilot.press("down", "enter")
                        for _ in range(100):
                            await pilot.pause(0.1)
                            if again.phase in {"review", "failed"}:
                                break
                        await pilot.pause()
                        self.assertEqual(again.phase, "review")
                        await pilot.press("ctrl+enter")
                        self.assertEqual(again.phase, "summary")
                        self.assertEqual(again.skipped, 2)
                        self.assertEqual(again.total, 1)
                        self.assertEqual(len(again.failures), 1)
                self.assertEqual(len(requests), 9)
                (source.parent / "ru-RU" / "a.ftl").write_text("a = Hello\n", encoding="utf-8")
                edited = create_app(repo)
                edited.error_log_path = repo / "translation-errors.log"
                async with edited.run_test() as pilot:
                    await pilot.press("enter")
                    for _ in range(50):
                        await pilot.pause(0.1)
                        if edited.model_names:
                            break
                    await pilot.press("down", "enter")
                    for _ in range(100):
                        await pilot.pause(0.1)
                        if edited.phase in {"review", "failed"}:
                            break
                    await pilot.pause()
                    self.assertEqual(edited.phase, "review")
                    await pilot.press("ctrl+enter")
                    self.assertEqual(edited.phase, "summary")
                    self.assertEqual(edited.success, 1)
                    self.assertEqual(edited.skipped, 1)
                    self.assertEqual(len(requests), 14)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
