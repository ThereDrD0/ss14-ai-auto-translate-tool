from __future__ import annotations

import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import tempfile
from threading import Thread
import unittest
from unittest.mock import patch

from textual.widgets import Checkbox, OptionList, RichLog
from textual.color import Color

from ss14_localization.tui import TokenEta, _cache_path, _inventory, _load_cache, create_app, summary_counts


class TuiTests(unittest.IsolatedAsyncioTestCase):
    def test_eta_uses_finished_file_time_and_remaining_token_sizes(self):
        small, large, larger = (Path(name) for name in ("small.ftl", "large.ftl", "larger.ftl"))
        eta = TokenEta({small: 10, large: 100, larger: 100}, concurrency=2)
        self.assertIsNone(eta.remaining(0))
        eta.start(small, 0)
        eta.start(large, 0)
        eta.finish(small, 4)
        eta.start(larger, 4)
        self.assertAlmostEqual(eta.remaining(6), 36)
        self.assertGreater(eta.remaining(100), 0)
        eta.finish(large, 8)
        eta.finish(larger, 9)
        self.assertEqual(eta.remaining(9), 0)

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

    async def test_model_screen_token_setting_defaults_off(self):
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
                self.assertFalse(checkbox.value)
                self.assertIn("▐ ▌", checkbox.render().plain)
                self.assertEqual(checkbox.styles.background, Color.parse("#111821"))
                await pilot.press("f3")
                self.assertTrue(checkbox.value)
                self.assertIn("▐✓▌", checkbox.render().plain)
                await pilot.press("f3")
                self.assertFalse(checkbox.value)
                self.assertIn("▐ ▌", checkbox.render().plain)

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
                repeated = len(body["messages"]) > 2
                content = ("a = Привет" if repeated else "a = {") if "a =" in source else "b = {"
                self.reply({"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": 13, "completion_tokens": 5}})

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

            def log_message(self, *_):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {
                "TRANSLATE_AI_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
                "TRANSLATE_AI_MODEL": "ignored",
                "TRANSLATE_AI_RESPONSE_MAX_ATTEMPTS": "2",
                "TRANSLATE_AI_RESPONSE_COOLDOWN_SECONDS": "0",
                "TRANSLATE_CONCURRENCY": "2",
            }):
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
                    await pilot.press("down")
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
                        if app.phase in {"summary", "failed"}:
                            break
                    self.assertEqual(app.phase, "summary")
                    self.assertEqual(app.success, 1)
                    self.assertEqual(app.skipped, 1)
                    self.assertEqual(app.total, 2)
                    self.assertEqual(len(app.failures), 1)
                    self.assertEqual(app.failures[0].path.name, "b.ftl")
                    self.assertEqual(app.prompt_tokens + app.completion_tokens, 72)
                    self.assertEqual(app.retry_tokens, 36)
                    self.assertIn("ПОВТОР", "\n".join(line.text for line in app.query_one("#log", RichLog).lines))
                    self.assertEqual(summary_counts(app.success, app.failures, app.skipped,
                                                    app.prompt_tokens, app.completion_tokens,
                                                    app.retry_tokens)["success_percent"], 50.0)
                    self.assertEqual((source.parent / "ru-RU" / "a.ftl").read_text(encoding="utf-8"),
                                     "a = Привет\n")
                    self.assertEqual((source.parent / "ru-RU" / "b.ftl").read_text(encoding="utf-8"),
                                     "b = Hello\n")
                    self.assertFalse((source.parent / "ru-RU" / "empty.ftl").exists())
                    self.assertTrue(all(body["model"] == "test-model" for body in requests))
                    self.assertFalse(any("c =" in item["content"] for body in requests
                                         for item in body["messages"]))
                    error_log = app.error_log_path.read_text(encoding="utf-8")
                    self.assertIn("[ПОВТОР]", error_log)
                    self.assertIn("[ОШИБКА]", error_log)
                    self.assertIn("b.ftl", error_log)
                    self.assertIn("Исходный текст:", error_log)
                    await pilot.press("q")
                    self.assertEqual(app.phase, "summary")
                cache = _load_cache(_cache_path(repo, "en-US", "ru-RU"))
                self.assertEqual(cache["prepared"], _inventory(source, source.parent / "ru-RU")[0])
                self.assertIn("a.ftl", cache["verified"])
                self.assertNotIn("b.ftl", cache["verified"])
                with patch("ss14_localization.strings.prepare_target_files",
                           side_effect=AssertionError("подготовка должна использовать кэш")):
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
                            if again.phase in {"summary", "failed"}:
                                break
                        self.assertEqual(again.phase, "summary")
                        self.assertEqual(again.skipped, 2)
                        self.assertEqual(again.total, 1)
                        self.assertEqual(len(again.failures), 1)
                        self.assertEqual(len(requests), 6)
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
                        if edited.phase in {"summary", "failed"}:
                            break
                    self.assertEqual(edited.phase, "summary")
                    self.assertEqual(edited.success, 1)
                    self.assertEqual(edited.skipped, 1)
                    self.assertEqual(len(requests), 10)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
