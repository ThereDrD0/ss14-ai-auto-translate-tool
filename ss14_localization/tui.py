"""Интерактивный запуск перевода в полноэкранном терминале."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from threading import Thread
from time import monotonic
from types import SimpleNamespace
import os

from .ai import AiConfig, AiEndpoint
from .constants import DEFAULT_LOCALE_ROOT, DEFAULT_SOURCE_CULTURE, DEFAULT_TARGET_CULTURE
from .dependencies import import_or_install
from .paths import find_repo_root


def available_locales(root: Path) -> tuple[list[str], list[str]]:
    """Исходные папки с FTL и целевые локали, доступные определителю языка."""
    existing = {path.name for path in root.iterdir() if path.is_dir()} if root.is_dir() else set()
    sources = sorted(name for name in existing if any((root / name).rglob("*.ftl")))
    lingua = import_or_install("lingua", "lingua-language-detector>=2.0,<3")
    langcodes = import_or_install("langcodes", "langcodes>=3.4,<4")
    supported = {
        f"{code}-{langcodes.Language.get(code).maximize().territory}"
        for language in lingua.Language.all()
        if (code := language.iso_code_639_1.name.lower())
    }
    return sources, sorted(existing | supported | {DEFAULT_TARGET_CULTURE})


def fetch_models(config: AiConfig) -> dict[str, tuple[AiEndpoint, ...]]:
    """Собирает модели из /v1/models всех настроенных серверов."""
    httpx = import_or_install("httpx", "httpx>=0.27,<1")
    found: dict[str, list[AiEndpoint]] = defaultdict(list)
    errors = []
    for endpoint in config.endpoints:
        try:
            with httpx.Client(timeout=config.timeout_seconds, proxy=endpoint.proxy, trust_env=False) as client:
                response = client.get(f"{endpoint.base_url}/models",
                                      headers={"Authorization": f"Bearer {endpoint.api_key}"})
                response.raise_for_status()
                for item in response.json()["data"]:
                    if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"]:
                        if endpoint not in found[item["id"]]:
                            found[item["id"]].append(endpoint)
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as error:
            errors.append(f"{endpoint.base_url}: {error}")
    if not found:
        raise ValueError("Не удалось получить модели из /v1/models. " + "; ".join(errors))
    return {name: tuple(endpoints) for name, endpoints in sorted(found.items())}


def summary_counts(success: int, failures: list, skipped: int, prompt_tokens: int,
                   completion_tokens: int, retry_tokens: int) -> dict:
    full = sum(item.translated_messages == 0 for item in failures)
    partial = len(failures) - full
    total = success + len(failures)
    return {"success": success, "full": full, "partial": partial, "skipped": skipped,
            "success_percent": 100 * success / total if total else 0.0,
            "tokens": prompt_tokens + completion_tokens,
            "retry_tokens": retry_tokens,
            "retry_percent": 100 * retry_tokens / (prompt_tokens + completion_tokens)
            if prompt_tokens + completion_tokens else 0.0}


def create_app(repo: Path):
    import_or_install("textual", "textual>=7.5,<8")
    from rich.table import Table
    from rich.text import Text
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical, VerticalScroll
    from textual.widgets import Header, OptionList, ProgressBar, RichLog, Static

    from .cli import _translation_settings
    from .strings import prepare_target_files
    from .translate import run_translate_files

    sources, targets = available_locales(repo / DEFAULT_LOCALE_ROOT)
    if not sources:
        raise ValueError(f"В {repo / DEFAULT_LOCALE_ROOT} нет исходных файлов .ftl")

    class TranslationApp(App):
        TITLE = "Перевод локализации SS14"
        BINDINGS = [Binding("ctrl+c", "quit", "Выход"),
                    Binding("left", "previous_column", "Левая колонка", show=False),
                    Binding("right", "next_column", "Правая колонка", show=False)]
        CSS = """
        Screen { background: #111821; color: #d4dde7; }
        Header { background: #1c2b39; color: #e6edf4; }
        #choose, #models, #work, #summary { width: 100%; height: 1fr; padding: 1 2; }
        #models, #work, #summary { display: none; }
        .title { height: 2; color: #a9c7d9; text-style: bold; }
        #columns { height: 1fr; }
        .column { width: 1fr; height: 1fr; margin-right: 2; }
        OptionList { height: 1fr; border: round #506474; background: #18232e; }
        OptionList:focus { border: round #83b4c7; }
        OptionList > .option-list--option-highlighted { background: #355467; color: #ffffff; }
        #model-list { width: 60%; }
        #stage, #eta, #active { height: 1; }
        #stage { color: #a9c7d9; text-style: bold; }
        #bar { height: 3; margin: 1 0; }
        #log { height: 1fr; border: round #506474; background: #18232e; }
        #hint { height: 1; background: #1c2b39; color: #c3d1da; padding: 0 2; }
        #summary-scroll { height: 1fr; }
        """

        def __init__(self):
            super().__init__()
            self.sources, self.targets = sources, targets
            self.source = DEFAULT_SOURCE_CULTURE if DEFAULT_SOURCE_CULTURE in sources else sources[0]
            self.target = DEFAULT_TARGET_CULTURE
            self.target_options = []
            self.models = {}
            self.model_names = []
            self.config = None
            self.phase = "choose"
            self.active = set()
            self.done = self.total = 0
            self.stage_started = monotonic()
            self.success = self.skipped = 0
            self.failures = []
            self.prompt_tokens = self.completion_tokens = self.retry_tokens = 0

        def compose(self) -> ComposeResult:
            yield Header(show_clock=False)
            with Vertical(id="choose"):
                yield Static("Выберите направление перевода", classes="title")
                with Horizontal(id="columns"):
                    with Vertical(classes="column"):
                        yield Static("Исходный язык")
                        yield OptionList(*self.sources, id="source-list")
                    with Vertical(classes="column"):
                        yield Static("Целевой язык")
                        yield OptionList(id="target-list")
            with Vertical(id="models"):
                yield Static("Выберите модель из /v1/models", classes="title")
                yield Static("Загрузка моделей...", id="model-status")
                yield OptionList(id="model-list")
            with Vertical(id="work"):
                yield Static("Подготовка", id="stage")
                yield ProgressBar(total=100, show_eta=False, id="bar")
                yield Static("Прошло: 0 с  ·  Осталось: —", id="eta")
                yield Static("Файлы: —", id="active")
                yield RichLog(wrap=False, auto_scroll=True, id="log")
            with Vertical(id="summary"):
                yield Static("Итоги перевода", classes="title")
                with VerticalScroll(id="summary-scroll"):
                    yield Static(id="summary-content")
            yield Static("↑/↓ выбор   Tab/←/→ колонка   Enter подтвердить   Ctrl+C выход", id="hint")

        def on_mount(self) -> None:
            source_list = self.query_one("#source-list", OptionList)
            source_list.highlighted = self.sources.index(self.source)
            self._refresh_targets()
            source_list.focus()
            self.set_interval(0.5, self._tick)

        def _refresh_targets(self) -> None:
            previous = self.target
            self.target_options = [name for name in self.targets
                                   if name.split("-")[0].lower() != self.source.split("-")[0].lower()]
            options = self.query_one("#target-list", OptionList)
            options.set_options(self.target_options)
            self.target = previous if previous in self.target_options else self.target_options[0]
            options.highlighted = self.target_options.index(self.target)

        def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
            if self.phase == "choose" and event.option_index is not None:
                if event.option_list.id == "source-list":
                    selected = self.sources[event.option_index]
                    if selected != self.source:
                        self.source = selected
                        self._refresh_targets()
                elif event.option_list.id == "target-list" and event.option_index < len(self.target_options):
                    self.target = self.target_options[event.option_index]

        def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
            if self.phase == "choose":
                self.phase = "models"
                self.query_one("#choose").display = False
                self.query_one("#models").display = True
                self.query_one("#hint", Static).update("↑/↓ выбор модели   Enter подтвердить   Ctrl+C выход")
                self._load_models()
            elif self.phase == "models" and event.option_index is not None and self.model_names:
                self._start_translation(self.model_names[event.option_index])

        def action_previous_column(self) -> None:
            if self.phase == "choose":
                self.query_one("#source-list", OptionList).focus()

        def action_next_column(self) -> None:
            if self.phase == "choose":
                self.query_one("#target-list", OptionList).focus()

        def _load_models(self) -> None:
            self.query_one("#model-status", Static).update("Загрузка моделей...")

            def worker():
                try:
                    config = AiConfig.from_env()
                    models = fetch_models(config)
                    self.call_from_thread(self._models_loaded, config, models, None)
                except Exception as error:
                    self.call_from_thread(self._models_loaded, None, {}, str(error))

            Thread(target=worker, daemon=True).start()

        def _models_loaded(self, config, models, error):
            if error:
                self.query_one("#model-status", Static).update(f"Ошибка: {error}. Enter — повторить запрос")
                self.model_names = []
                return
            self.config = config
            self.models = models
            self.model_names = list(models)
            options = self.query_one("#model-list", OptionList)
            options.set_options(self.model_names)
            options.highlighted = 0
            options.focus()
            self.query_one("#model-status", Static).update(f"Доступно моделей: {len(models)}")

        def on_key(self, event) -> None:
            if self.phase == "models" and not self.model_names and event.key == "enter":
                self._load_models()

        def _start_translation(self, model: str) -> None:
            self.phase = "work"
            self.query_one("#models").display = False
            self.query_one("#work").display = True
            self.query_one("#hint", Static).update("↑/↓ прокрутка журнала   Ctrl+C выход")
            self._new_stage("Подготовка", 1)
            config = replace(self.config, endpoints=tuple(replace(endpoint, model=model)
                                                           for endpoint in self.models[model]))
            Thread(target=self._translate_worker, args=(config,), daemon=True).start()

        def _new_stage(self, name: str, total: int) -> None:
            self.done = 0
            self.total = max(total, 1)
            self.active.clear()
            self.stage_started = monotonic()
            self.query_one("#stage", Static).update(name)
            self.query_one("#bar", ProgressBar).update(total=self.total, progress=0)
            self._tick()

        def _tick(self) -> None:
            if self.phase != "work":
                return
            elapsed = monotonic() - self.stage_started
            remaining = elapsed / self.done * (self.total - self.done) if self.done else None
            eta = f"{remaining:.0f} с" if remaining is not None else "ожидание первого файла"
            self.query_one("#eta", Static).update(
                f"Готово: {self.done}/{self.total}  ·  Прошло: {elapsed:.0f} с  ·  Осталось: ~{eta}")
            names = [str(path.relative_to(repo)) if repo in path.parents else path.name
                     for path in sorted(self.active)]
            shown = ", ".join(names[:5]) or "—"
            if len(names) > 5:
                shown += f", ... (+{len(names) - 5})"
            self.query_one("#active", Static).update("Файлы: " + shown)

        def _log(self, status: str, path: Path | None = None, detail: str = "") -> None:
            colors = {"ГОТОВО": "#a8cfb1", "ПРОВЕРЕН": "#a9b9c6", "ПРОПУСК": "#a9b9c6", "ОШИБКА": "#e2a2a5",
                      "НАЧАТО": "#aac6d5", "СОЗДАНО": "#a8cfb1", "ОБНОВЛЕНО": "#a8cfb1",
                      "УДАЛЕНО": "#cbbba5"}
            line = Text()
            line.append(f"[{status}] ", style=colors.get(status, "#c2d0dc"))
            if path:
                line.append(str(path.relative_to(repo)) if repo in path.parents else str(path))
            if detail:
                line.append("\n" + detail)
            self.query_one("#log", RichLog).write(line)

        def _prepare_event(self, kind, path, done, total):
            if kind == "started":
                self.active.add(path)
                self.total = max(total, 1)
                self.query_one("#bar", ProgressBar).update(total=self.total)
            elif kind in {"completed", "finished"}:
                self.active.discard(path)
                self.done = done
                self.query_one("#bar", ProgressBar).update(progress=done)
                if kind == "completed":
                    self._log("ПРОВЕРЕН", path)
            else:
                self._log({"создать": "СОЗДАНО", "обновить": "ОБНОВЛЕНО",
                           "удалить": "УДАЛЕНО"}[kind], path)
            self._tick()

        def _translation_event(self, kind, path, payload):
            if kind == "started":
                self.active.add(path)
                self._log("НАЧАТО", path)
            else:
                self.active.discard(path)
                self.done += 1
                self.query_one("#bar", ProgressBar).update(progress=self.done)
                if kind == "skipped":
                    self.skipped += 1
                    self._log("ПРОПУСК", path, "Перевод уже есть или файл исключён.")
                elif kind == "completed":
                    self.success += 1
                    self._log("ГОТОВО", path, payload["text"])
                else:
                    self._log("ОШИБКА", path,
                              f"Причина: {payload['error']}\nИсходный текст:\n{payload['source']}\n"
                              f"Итоговый ответ:\n{payload['response'] if payload['response'] is not None else 'Ответ не получен'}")
            self._tick()

        def _usage(self, prompt_tokens, completion_tokens, retry):
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            if retry:
                self.retry_tokens += prompt_tokens + completion_tokens

        def _translate_worker(self, config):
            try:
                args = SimpleNamespace(
                    repo_root=repo, source_culture=self.source, target_culture=self.target,
                    locale_root=DEFAULT_LOCALE_ROOT, pass_list=Path(os.environ["TRANSLATE_PASS_LIST"])
                    if os.environ.get("TRANSLATE_PASS_LIST") else None,
                    language_profile=Path(os.environ["TRANSLATE_LANGUAGE_PROFILE"])
                    if os.environ.get("TRANSLATE_LANGUAGE_PROFILE") else None,
                    language_ratio=float(os.environ.get("TRANSLATE_LANGUAGE_RATIO", "0.8")),
                    prompt=Path(os.environ["TRANSLATE_PROMPT"]) if os.environ.get("TRANSLATE_PROMPT") else None,
                    glossary=Path(os.environ["TRANSLATE_GLOSSARY"]) if os.environ.get("TRANSLATE_GLOSSARY") else None,
                    chunk_size=int(os.environ.get("TRANSLATE_CHUNK_SIZE", "4000")),
                    concurrency=int(os.environ.get("TRANSLATE_CONCURRENCY", "2")),
                    batch_size=int(os.environ.get("TRANSLATE_BATCH_SIZE", "100")), dry_run=False)
                checker, budget, prompt = _translation_settings(args)
                source_root = repo / DEFAULT_LOCALE_ROOT / self.source
                target_root = repo / DEFAULT_LOCALE_ROOT / self.target
                prepared = prepare_target_files(source_root, target_root, [Path(".")],
                                                on_event=lambda *items: self.call_from_thread(self._prepare_event, *items))
                files = list(prepared.target_files)
                self.call_from_thread(self._new_stage, "Перевод", len(files))
                # ponytail: одна группа файлов сохраняет общую статистику; ограничение параллельности задаёт semaphore.
                result = run_translate_files(
                    files, prompt, args.chunk_size, target_culture=self.target,
                    concurrency=args.concurrency, checker=checker, budget=budget, ai_config=config,
                    on_event=lambda *items: self.call_from_thread(self._translation_event, *items),
                    on_usage=lambda *items: self.call_from_thread(self._usage, *items))
                self.call_from_thread(self._finish, result)
            except Exception as error:
                self.call_from_thread(self._fatal, str(error))

        def _finish(self, result):
            self.failures = list(result.failed_details)
            counts = summary_counts(self.success, self.failures, self.skipped,
                                    self.prompt_tokens, self.completion_tokens, self.retry_tokens)
            table = Table(title="Итоги", show_header=False, border_style="#506474")
            table.add_column("Показатель", style="#a9c7d9")
            table.add_column("Значение")
            table.add_row("Успешно переведено файлов", str(counts["success"]))
            table.add_row("Полных ошибок", str(counts["full"]))
            table.add_row("Частичных ошибок", str(counts["partial"]))
            table.add_row("Уже переведено / пропущено", str(counts["skipped"]))
            table.add_row("Успех среди обработанных", f"{counts['success_percent']:.1f}%")
            table.add_row("Токенов всего", str(counts["tokens"]))
            table.add_row("Из них на повторы", f"{counts['retry_tokens']} ({counts['retry_percent']:.1f}%)")
            table.add_row("Токены", "Нет данных от API" if not counts["tokens"] else
                          f"Вход: {self.prompt_tokens}; выход: {self.completion_tokens}")
            errors = Table(title="Ошибки по частоте", border_style="#506474")
            errors.add_column("Ошибка")
            errors.add_column("Раз", justify="right")
            errors.add_column("Файлы")
            by_error = defaultdict(list)
            for failure in self.failures:
                by_error[failure.error].append(str(failure.path.relative_to(repo)))
            for error, paths in sorted(by_error.items(), key=lambda item: (-len(item[1]), item[0])):
                errors.add_row(error, str(len(paths)), "\n".join(paths))
            if not by_error:
                errors.add_row("Нет", "0", "—")
            content = self.query_one("#summary-content", Static)
            content.update(table)
            self.query_one("#summary-scroll", VerticalScroll).mount(Static(errors))
            self.query_one("#work").display = False
            self.query_one("#summary").display = True
            self.query_one("#hint", Static).update("Ctrl+C выход")
            self.phase = "summary"

        def _fatal(self, error):
            self._log("ОШИБКА", detail=error)
            self.query_one("#stage", Static).update("Работа остановлена из-за ошибки")
            self.query_one("#hint", Static).update("Ctrl+C выход")
            self.phase = "failed"

    return TranslationApp()


def run() -> int:
    try:
        repo = find_repo_root()
    except RuntimeError as error:
        raise ValueError(str(error)) from error
    create_app(repo).run()
    return 0
