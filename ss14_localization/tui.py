"""Интерактивный запуск перевода в полноэкранном терминале."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from hashlib import blake2b
import json
from pathlib import Path
from threading import Thread
from time import monotonic
from types import SimpleNamespace
import os

from .ai import AiConfig, AiEndpoint
from .constants import DEFAULT_LOCALE_ROOT, DEFAULT_SOURCE_CULTURE, DEFAULT_TARGET_CULTURE
from .dependencies import import_or_install
from .filesystem import iter_files, read_text, write_text_if_changed
from .paths import TOOL_ROOT, find_repo_root


def _file_hash(path: Path) -> str:
    return blake2b(path.read_bytes(), digest_size=16).hexdigest()


def _inventory(source_root: Path, target_root: Path):
    """Снимок содержимого без зависимости от времени изменения файлов."""
    source_files = iter_files(source_root, ".ftl")
    target_files = iter_files(target_root, ".ftl")
    fingerprints = {}
    digests = []
    for label, root, files in (("s", source_root, source_files), ("t", target_root, target_files)):
        for path in files:
            relative = path.relative_to(root).as_posix()
            digest = _file_hash(path)
            digests.append(f"{label}:{relative}:{digest}")
            if label == "t":
                fingerprints[relative] = digest
    source_digest = blake2b("\n".join(digests[:len(source_files)]).encode(), digest_size=16).hexdigest()
    all_digest = blake2b("\n".join(digests).encode(), digest_size=16).hexdigest()
    return all_digest, source_digest, source_files, fingerprints


def _cache_path(repo: Path, source: str, target: str) -> Path:
    key = blake2b(f"{repo.resolve()}:{source}:{target}".encode(), digest_size=12).hexdigest()
    return TOOL_ROOT / ".deps" / f"tui-{key}.json"


def _load_cache(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if data.get("version") == 3 else {}
    except (OSError, ValueError, AttributeError):
        return {}


def _save_cache(path: Path, data: dict) -> None:
    try:
        write_text_if_changed(path, json.dumps(data, ensure_ascii=False))
    except OSError:
        pass  # ponytail: кэш ускоряет повторный запуск, но не должен останавливать перевод.


def _checker_key(checker) -> str:
    profile = checker.profile.read_bytes() if checker.profile else b""
    settings = repr((checker.source_culture, checker.target_culture, checker.minimum_ratio,
                     checker.pass_list.terms, checker.pass_list.ignored_files,
                     os.environ.get("TRANSLATE_DETECT_SOURCE"), os.environ.get("TRANSLATE_DETECT_TARGET"))).encode()
    return blake2b(settings + profile, digest_size=16).hexdigest()


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


class TokenEta:
    def __init__(self, weights: dict[Path, int], concurrency: int):
        self.weights = weights.copy()
        self.concurrency = max(1, concurrency)
        self.started = {}
        self.sample_tokens = 0
        self.sample_seconds = 0.0

    def start(self, path: Path, now: float) -> None:
        self.started[path] = now

    def finish(self, path: Path, now: float) -> None:
        weight = self.weights.pop(path, 0)
        started = self.started.pop(path, None)
        if weight and started is not None:
            self.sample_tokens += weight
            self.sample_seconds += max(0, now - started)

    def remaining(self, now: float) -> float | None:
        if not self.weights:
            return 0.0
        if not self.sample_tokens:
            return None
        seconds_per_token = self.sample_seconds / self.sample_tokens
        work = sum(self.weights.values()) * seconds_per_token
        # ponytail: без прогресса внутри файла вычитаем не больше 80%; поток токенов даст точный остаток.
        work -= sum(min(now - started, self.weights[path] * seconds_per_token * 0.8)
                    for path, started in self.started.items() if path in self.weights)
        return max(0.0, work / min(self.concurrency, len(self.weights)))


def create_app(repo: Path):
    import_or_install("textual", "textual>=7.5,<8")
    from rich.table import Table
    from rich.text import Text
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical, VerticalScroll
    from textual.widgets import Checkbox, Header, OptionList, ProgressBar, RichLog, Static

    from .cli import _translation_settings
    from .strings import prepare_target_files
    from .translate import _split_messages, run_translate_files

    sources, targets = available_locales(repo / DEFAULT_LOCALE_ROOT)
    if not sources:
        raise ValueError(f"В {repo / DEFAULT_LOCALE_ROOT} нет исходных файлов .ftl")

    class SaveTokensCheckbox(Checkbox):
        @property
        def BUTTON_INNER(self) -> str:
            return "✓" if self.value else " "

    class TranslationApp(App):
        TITLE = "Перевод локализации SS14"
        BINDINGS = [Binding("ctrl+c", "quit", "Выход"),
                    Binding("ctrl+q", "noop", "", show=False, priority=True),
                    Binding("f2", "toggle_auto_scroll", "Автопрокрутка", priority=True),
                    Binding("f3", "toggle_save_tokens", "Экономия токенов", priority=True),
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
        #save-tokens {
            height: 1;
            width: auto;
            border: none;
            padding: 0;
            background: #111821;
        }
        #save-tokens:focus { border: none; background: #111821; background-tint: #111821 0%; }
        #save-tokens > .toggle--button { background: #355467; color: #a8cfb1; }
        #save-tokens:focus > .toggle--label { background: #355467; color: #ffffff; }
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
            self.token_eta = None
            self.error_log_path = TOOL_ROOT / "translation-errors.log"
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
                yield SaveTokensCheckbox("Экономить токены: без примеров готового перевода",
                                         value=False, id="save-tokens")
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
                self._model_hint()
                self._load_models()
            elif self.phase == "models" and event.option_index is not None and self.model_names:
                self._start_translation(self.model_names[event.option_index])

        def action_previous_column(self) -> None:
            if self.phase == "choose":
                self.query_one("#source-list", OptionList).focus()

        def action_noop(self) -> None:
            pass

        def action_next_column(self) -> None:
            if self.phase == "choose":
                self.query_one("#target-list", OptionList).focus()

        def action_toggle_auto_scroll(self) -> None:
            if self.phase not in {"work", "failed"}:
                return
            log = self.query_one("#log", RichLog)
            log.auto_scroll = not log.auto_scroll
            if log.auto_scroll:
                log.scroll_end(animate=False)
            self._work_hint()

        def action_toggle_save_tokens(self) -> None:
            if self.phase == "models":
                checkbox = self.query_one("#save-tokens", Checkbox)
                checkbox.value = not checkbox.value

        def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
            if self.phase == "models" and event.checkbox.id == "save-tokens":
                self._model_hint()

        def _model_hint(self) -> None:
            state = "вкл" if self.query_one("#save-tokens", Checkbox).value else "выкл"
            self.query_one("#hint", Static).update(
                f"↑/↓ модель  Tab/Space настройка  F3: {state}  Enter модель  Ctrl+C выход")

        def _work_hint(self) -> None:
            state = "включена" if self.query_one("#log", RichLog).auto_scroll else "выключена"
            self.query_one("#hint", Static).update(
                f"↑/↓, PgUp/PgDn журнал   F2 автопрокрутка: {state}   Ctrl+C выход")

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
            save_tokens = self.query_one("#save-tokens", Checkbox).value
            self.phase = "work"
            self.query_one("#models").display = False
            self.query_one("#work").display = True
            self.query_one("#log", RichLog).focus()
            self._work_hint()
            self._new_stage("Подготовка", 1)
            self._log("ЖУРНАЛ", detail=f"Ошибки и повторы: {self.error_log_path}")
            config = replace(self.config, endpoints=tuple(replace(endpoint, model=model)
                                                           for endpoint in self.models[model]))
            Thread(target=self._translate_worker, args=(config, save_tokens), daemon=True).start()

        def _new_stage(self, name: str, total: int, weights=None, concurrency=1) -> None:
            self.done = 0
            self.total = max(total, 1)
            self.active.clear()
            self.stage_started = monotonic()
            self.token_eta = TokenEta(weights, concurrency) if weights is not None else None
            self.query_one("#stage", Static).update(name)
            self.query_one("#bar", ProgressBar).update(total=self.total, progress=0)
            self._tick()

        def _tick(self) -> None:
            if self.phase != "work" or not list(self.query("#eta")):
                return
            elapsed = monotonic() - self.stage_started
            remaining = self.token_eta.remaining(monotonic()) if self.token_eta else None
            eta = f"~{remaining:.0f} с" if remaining is not None else (
                "после первого файла" if self.token_eta else "—")
            self.query_one("#eta", Static).update(
                f"Готово: {self.done}/{self.total}  ·  Прошло: {elapsed:.0f} с  ·  Осталось: {eta}")
            names = [str(path.relative_to(repo)) if repo in path.parents else path.name
                     for path in sorted(self.active)]
            shown = ", ".join(names[:5]) or "—"
            if len(names) > 5:
                shown += f", ... (+{len(names) - 5})"
            self.query_one("#active", Static).update("Файлы: " + shown)

        def _log(self, status: str, path: Path | None = None, detail: str = "") -> None:
            colors = {"ГОТОВО": "#a8cfb1", "ПРОВЕРЕН": "#a9b9c6", "ПРОПУСК": "#a9b9c6", "ОШИБКА": "#e2a2a5",
                      "ПОВТОР": "#d5bd94",
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
                if self.token_eta:
                    self.token_eta.start(path, monotonic())
                self._log("НАЧАТО", path)
            else:
                self.active.discard(path)
                if self.token_eta:
                    self.token_eta.finish(path, monotonic())
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

        def _retry_event(self, path, kind, attempt, maximum, error, cooldown, will_retry):
            if not will_retry:
                return
            reason = "Повтор запроса к ИИ" if kind == "request" else "Повтор после проверки ответа"
            wait = f"; ожидание сервера: {cooldown:g} с" if cooldown else ""
            self._log("ПОВТОР", path, f"{reason}: попытка {attempt + 1}/{maximum or '∞'}{wait}; причина: {error}")

        def _prepared_skipped(self, count: int) -> None:
            self.skipped += count
            if count:
                self._log("ПРОПУСК", detail=f"После подготовки перевод не нужен: {count} файлов")
            self._tick()

        def _plan_progress(self, done: int) -> None:
            self.done = done
            self.query_one("#bar", ProgressBar).update(progress=done)
            self._tick()

        def _translate_worker(self, config, save_tokens=False):
            try:
                args = SimpleNamespace(
                    repo_root=repo, source_culture=self.source, target_culture=self.target,
                    locale_root=DEFAULT_LOCALE_ROOT, pass_list=Path(os.environ["TRANSLATE_PASS_LIST"])
                    if os.environ.get("TRANSLATE_PASS_LIST") else None,
                    language_profile=Path(os.environ["TRANSLATE_LANGUAGE_PROFILE"])
                    if os.environ.get("TRANSLATE_LANGUAGE_PROFILE") else None,
                    language_ratio=float(os.environ["TRANSLATE_LANGUAGE_RATIO"])
                    if os.environ.get("TRANSLATE_LANGUAGE_RATIO") else None,
                    prompt=Path(os.environ["TRANSLATE_PROMPT"]) if os.environ.get("TRANSLATE_PROMPT") else None,
                    glossary=Path(os.environ["TRANSLATE_GLOSSARY"]) if os.environ.get("TRANSLATE_GLOSSARY") else None,
                    chunk_size=int(os.environ.get("TRANSLATE_CHUNK_SIZE", "4000")),
                    concurrency=int(os.environ.get("TRANSLATE_CONCURRENCY", "2")),
                    batch_size=int(os.environ.get("TRANSLATE_BATCH_SIZE", "100")), dry_run=False)
                checker, budget, prompt = _translation_settings(args)
                source_root = repo / DEFAULT_LOCALE_ROOT / self.source
                target_root = repo / DEFAULT_LOCALE_ROOT / self.target
                cache_path = _cache_path(repo, self.source, self.target)
                cache = _load_cache(cache_path)
                inventory, source_digest, source_files, target_hashes = _inventory(source_root, target_root)
                initial_source, initial_targets = source_digest, target_hashes
                checker_key = _checker_key(checker)
                if cache.get("source") != source_digest or cache.get("checker") != checker_key:
                    cache["verified"] = {}
                verified = cache.get("verified")
                if not isinstance(verified, dict):
                    verified = {}
                verified = {name: digest for name, digest in verified.items()
                            if name in target_hashes and target_hashes[name] == digest}
                prep_safe = True
                if cache.get("prepared") == inventory:
                    files = [target_root / path.relative_to(source_root) for path in source_files
                             if path.relative_to(source_root).as_posix() in target_hashes]
                    self.call_from_thread(self._prepare_event, "finished", target_root, 1, 1)
                    self.call_from_thread(self._log, "ПРОПУСК", None, "Подготовка не требуется: файлы не менялись")
                else:
                    prepared = prepare_target_files(
                        source_root, target_root, [Path(".")],
                        on_event=lambda *items: self.call_from_thread(self._prepare_event, *items))
                    files = list(prepared.target_files)
                    inventory, source_digest, _, target_hashes = _inventory(source_root, target_root)
                    changed_names = {path.relative_to(target_root).as_posix()
                                     for path in prepared.changed_paths}
                    prep_safe = source_digest == initial_source and all(
                        initial_targets.get(name) == target_hashes.get(name)
                        for name in initial_targets.keys() | target_hashes.keys()
                        if name not in changed_names)
                    if source_digest != initial_source:
                        verified = {}
                    verified = {name: digest for name, digest in verified.items()
                                if name in target_hashes and target_hashes[name] == digest}
                cache = {"version": 3, "source": source_digest, "checker": checker_key,
                         "prepared": inventory if prep_safe else None, "verified": verified}
                _save_cache(cache_path, cache)
                self.call_from_thread(self._new_stage, "Проверка перевода", len(files))
                candidates, plans, weights = [], {}, {}
                checked = {}
                skipped = 0
                for number, path in enumerate(files, 1):
                    name = path.relative_to(target_root).as_posix()
                    if name in verified:
                        skipped += 1
                    elif path.name in checker.pass_list.ignored_files:
                        checked[name] = _file_hash(path)
                        skipped += 1
                    else:
                        text = ""
                        try:
                            text = read_text(path)
                            messages, context = _split_messages(text, None, self.target, checker)
                            if messages:
                                plans[path] = (text, messages, context)
                                candidates.append(path)
                                weights[path] = max(1, budget.tokens("\n\n".join(item.text for item in messages)))
                            else:
                                checked[name] = _file_hash(path)
                                skipped += 1
                        except Exception:
                            candidates.append(path)  # Ошибка чтения или разбора будет показана при переводе.
                            weights[path] = max(1, budget.tokens(text))
                    if number % 25 == 0 or number == len(files):
                        self.call_from_thread(self._plan_progress, number)
                self.call_from_thread(self._prepared_skipped, skipped)
                self.call_from_thread(self._new_stage, "Перевод", len(candidates), weights, args.concurrency)

                def on_translation_event(kind, path, payload):
                    if kind in {"completed", "skipped"} and path.is_file():
                        checked[path.relative_to(target_root).as_posix()] = _file_hash(path)
                    self.call_from_thread(self._translation_event, kind, path, payload)

                # ponytail: одна группа файлов сохраняет общую статистику; ограничение параллельности задаёт semaphore.
                result = run_translate_files(
                    candidates, prompt, args.chunk_size, target_culture=self.target,
                    concurrency=args.concurrency, checker=checker, budget=budget, ai_config=config,
                    save_tokens=save_tokens, plans=plans,
                    on_event=on_translation_event,
                    on_retry=lambda *items: self.call_from_thread(self._retry_event, *items),
                    on_usage=lambda *items: self.call_from_thread(self._usage, *items))
                try:
                    if candidates:
                        final_inventory, final_source, _, final_hashes = _inventory(source_root, target_root)
                    else:
                        final_inventory, final_source, final_hashes = inventory, source_digest, target_hashes
                    expected_hashes = {**target_hashes, **checked}
                    if prep_safe and final_source == source_digest and final_hashes == expected_hashes:
                        cache["prepared"] = final_inventory
                    cache["verified"] = {**verified, **{name: digest for name, digest in checked.items()
                                                        if final_hashes.get(name) == digest}}
                    _save_cache(cache_path, cache)
                except OSError:
                    pass  # Не превращаем завершённый перевод в ошибку из-за кэша.
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
            self.phase = "failed"
            self._work_hint()

    return TranslationApp()


def run() -> int:
    try:
        repo = find_repo_root()
    except RuntimeError as error:
        raise ValueError(str(error)) from error
    create_app(repo).run()
    return 0
