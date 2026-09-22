from __future__ import annotations

import asyncio
from contextvars import ContextVar
from dataclasses import dataclass
import os
from pathlib import Path
import re
import sys

from .ai import AiConfig, OpenAICompatibleClient, ResponseTruncatedError
from .budget import OutputBudget
from .filesystem import read_text, write_text_if_changed
from .fluent import (FluentMessage, assert_structure, entries, message_map, parse_resource,
                     rich_tags, serialize_entry, serialize_resource, syntax, visible_parts)
from .language import LanguageChecker, PassList, load_pass_list


_current_file = ContextVar("translation_file", default=None)


@dataclass(frozen=True)
class TranslationFailure:
    path: Path
    translated_messages: int
    changed: bool
    error: str


@dataclass(frozen=True)
class TranslationRunResult:
    translated_messages: int
    changed_files: int
    failed_files: tuple[Path, ...] = ()
    failed_details: tuple[TranslationFailure, ...] = ()


class TranslationValidationError(ValueError):
    def __init__(self, message, ai_response=None):
        super().__init__(message)
        self.ai_response = ai_response


class TranslationFileError(RuntimeError):
    def __init__(self, path, translated_messages, changed, error):
        super().__init__(f"{path}: {error}")
        self.path, self.translated_messages, self.changed, self.error = path, translated_messages, changed, error


def build_translation_prompt(prompt_path, glossary_path=None, source_culture="en-US", target_culture="ru-RU",
                             pass_list=None):
    prompt = read_text(prompt_path)
    prompt = prompt.replace("{{SOURCE}}", source_culture).replace("{{TARGET}}", target_culture)
    if glossary_path is not None:
        if not glossary_path.is_file():
            raise FileNotFoundError(glossary_path)
        prompt += "\n\nСловарь терминов (используйте нужное направление перевода, не подменяйте целевой язык):\n"
        prompt += read_text(glossary_path)
    if pass_list and pass_list.terms:
        prompt += "\n\nНазвания из pass-листа сохраняйте дословно и с исходным регистром:\n"
        prompt += "\n".join(pass_list.terms)
    prompt += (f"\n\nИсходный язык: {source_culture}. Целевой язык: {target_culture}. "
               "Верните только обычный текст FTL, без JSON, пояснений и ограждений Markdown. "
               "Сохраните ключи, атрибуты, комментарии, ссылки, переменные, функции, варианты выбора, "
               "параметры разметки и названия из pass-листа. Не переписывайте уже переведённые части.")
    return prompt


def _split_messages(text, source_text, target_culture, checker=None):
    checker = checker or LanguageChecker("ru-RU" if target_culture.startswith("en") else "en-US",
                                         target_culture, load_pass_list())
    resource = parse_resource(text)
    nodes = entries(resource)
    pending, completed = [], []
    for key, message in message_map(text, resource).items():
        (pending if checker.needs_translation(nodes[key]) else completed).append(message)
    return pending, completed


def _messages_to_translate(text, source_text, target_culture, checker=None):
    return _split_messages(text, source_text, target_culture, checker)[0]


def _chunks(messages, chunk_size, budget=None, prompt=""):
    budget = budget or OutputBudget.from_env()
    result, current = [], []
    for message in messages:
        joined = "\n\n".join(item.text for item in [*current, message])
        if current and (len(joined) > chunk_size or not budget.fits(joined, prompt)):
            result.append(current)
            current = []
        current.append(message)
    if current:
        result.append(current)
    return result


def _strip_fence(response):
    # ponytail: убираем только вводные блоки рассуждений, не теги внутри FTL-значений.
    def strip_thoughts(text):
        while match := re.match(r"(?is)\A<think(?:\s[^>]*)?>.*?</think\s*>\s*", text):
            text = text[match.end():]
        return text

    text = strip_thoughts(response.strip())
    if text.startswith("```") and text.endswith("```"):
        text = "\n".join(text.splitlines()[1:-1]).strip()
    return strip_thoughts(text)


def _validate_translated_message(source, translated, checker=None, pass_list=None):
    source_resource = parse_resource(source.text if isinstance(source, FluentMessage) else source)
    target_resource = parse_resource(translated)
    assert_structure(source_resource, target_resource)
    source_nodes, target_nodes = entries(source_resource), entries(target_resource)
    pass_list = pass_list or (checker.pass_list if checker else PassList())
    visible_text = []
    for key in source_nodes:
        source_node, target_node = source_nodes[key], target_nodes[key]
        source_patterns = ([source_node.value] if source_node.value else []) + [a.value for a in source_node.attributes]
        target_patterns = ([target_node.value] if target_node.value else []) + [a.value for a in target_node.attributes]
        if checker:
            from .fluent import pattern_text
            for original, translated in zip(source_patterns, target_patterns):
                if checker.ratio(pattern_text(original)) >= checker.minimum_ratio and not original.equals(translated, ignored_fields=["span"]):
                    raise TranslationValidationError(f"ИИ переписал уже переведённое или разрешённое поле для {key}")
        source_text = "\n".join(visible_parts(source_nodes[key]))
        target_text = "\n".join(visible_parts(target_nodes[key]))
        visible_text.append(target_text)
        from .fluent import RICH_TAG_RE, RICH_TAG_NAMES
        source_tags = [m.group(0) for m in RICH_TAG_RE.finditer(source_text) if m.group(2).lower() in RICH_TAG_NAMES]
        target_tags = [m.group(0) for m in RICH_TAG_RE.finditer(target_text) if m.group(2).lower() in RICH_TAG_NAMES]
        if source_tags != target_tags:
            raise TranslationValidationError(f"ИИ изменил rich-text-разметку для {key}")
        angle_tags = r"</?[^>]+>"
        if re.findall(angle_tags, source_text) != re.findall(angle_tags, target_text):
            raise TranslationValidationError(f"ИИ изменил XML-разметку для {key}")
        if re.findall(r"https?://\S+", source_text) != re.findall(r"https?://\S+", target_text):
            raise TranslationValidationError(f"ИИ изменил адрес ссылки для {key}")
        pass_list.assert_preserved(source_text, target_text)
    if checker:
        checker.validate_text("\n".join(visible_text))


def _parse_translation_response(response, expected, target_culture=None, checker=None):
    text = _strip_fence(response)
    parsed = parse_resource(text)
    received = entries(parsed)
    if set(received) != set(expected):
        raise TranslationValidationError(
            f"Ключи ответа не совпадают: отсутствуют {sorted(set(expected) - set(received))}; "
            f"лишние {sorted(set(received) - set(expected))}")
    source_text = "\n\n".join(message.text for message in expected.values())
    _validate_translated_message(source_text, text, checker)
    return {key: serialize_entry(received[key]).rstrip("\n") for key in expected}


async def _translate_chunk(client, prompt, chunk, target_culture, checker=None, budget=None, context=()):
    budget = budget or OutputBudget.from_env()
    payload = "\n\n".join(message.text for message in chunk)
    if not budget.fits(payload, prompt):
        raise ResponseTruncatedError("Блок не помещается в заданный бюджет ответа/контекста")
    expected = {message.id: message for message in chunk}
    attempts = int(os.environ.get("TRANSLATE_AI_RESPONSE_MAX_ATTEMPTS", "3"))
    cooldown = float(os.environ.get("TRANSLATE_AI_RESPONSE_COOLDOWN_SECONDS", "0"))
    if attempts < 0 or cooldown < 0:
        raise ValueError("Число попыток и ожидание не могут быть отрицательными")
    last_error = None
    last_response = None
    index = 0
    while attempts == 0 or index < attempts:
        index += 1
        messages = [{"role": "system", "content": prompt}, {"role": "user", "content": payload}]
        if last_error is not None:
            feedback = f"Предыдущая попытка не прошла проверку: {last_error}. Исправьте ошибку и верните полный FTL-блок."
            messages.append({"role": "user", "content": feedback})
        if context:
            heading = "Примеры уже переведённых ключей. Используйте как контекст и не возвращайте в ответ:\n"
            examples = ""
            remaining = (budget.max_input_tokens - budget.reserve -
                         sum(budget.tokens(item["content"]) for item in messages)) if budget.max_input_tokens else None
            for item in context:
                candidate = f"{examples}\n\n{item.text}" if examples else item.text
                if remaining is None or budget.tokens(heading + candidate) <= remaining:
                    examples = candidate
            if examples:
                messages.insert(1, {"role": "user", "content": heading + examples})
        if budget.max_input_tokens and sum(budget.tokens(item["content"]) for item in messages) + budget.reserve > budget.max_input_tokens:
            raise ResponseTruncatedError("Источник, подсказка и обратная связь не помещаются в контекст")
        if getattr(client, "_supports_retry", False):
            response = await client.chat(messages, retry=index > 1)
        else:
            response = await client.chat(messages)
        response = _strip_fence(response)
        if budget.tokens(response) > budget.max_tokens:
            raise ResponseTruncatedError("Ответ превышает указанное окно; блок будет уменьшен")
        try:
            return _parse_translation_response(response, expected, target_culture, checker)
        except ValueError as error:
            last_error, last_response = error, response
            if getattr(client, "_on_retry", None):
                client._on_retry("validation", index, attempts, error,
                                 cooldown if attempts == 0 or index < attempts else 0,
                                 attempts == 0 or index < attempts)
            if not getattr(client, "_quiet", False):
                print(f"Повтор проверки {index}/{attempts or '∞'}: {error}", file=sys.stderr, flush=True)
            if attempts == 0 or index < attempts:
                await asyncio.sleep(cooldown)
    raise TranslationValidationError(str(last_error), last_response) from last_error


def _text_slots(node, technical=False):
    ast = syntax().ast
    if isinstance(node, ast.TextElement) or (isinstance(node, ast.StringLiteral) and not technical and node.value):
        yield node
    elif isinstance(node, ast.BaseNode):
        for key, value in vars(node).items():
            if key != "span":
                yield from _text_slots(value, technical or key == "arguments" or key == "comment")
    elif isinstance(node, list):
        for value in node:
            yield from _text_slots(value, technical)


def _fragment_node(text):
    ast = syntax().ast
    # Escaping leading rich-text lines is owned by the program, not by the model.
    from .fluent import escape_leading_multiline_markup
    elements = []
    for part in re.split(r"([{}])", escape_leading_multiline_markup(text)):
        if part in {"{", "}"}:
            elements.append(ast.Placeable(ast.StringLiteral(part)))
        elif part:
            elements.append(ast.TextElement(part))
    return ast.Message(ast.Identifier("translation-part"), ast.Pattern(elements))


def _slot_text(slot):
    return slot.parse()["value"] if isinstance(slot, syntax().ast.StringLiteral) else slot.value


def _literal_value(text):
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", r"\u000A").replace("\r", r"\u000D")


async def _translate_large_message(message, client, prompt, checker, budget, depth=0, context=()):
    if depth > 8:
        raise TranslationValidationError("Не удалось уменьшить фрагмент до допустимого размера")
    source_node = next(iter(entries(parse_resource(message.text)).values()))
    result_node = source_node.clone()
    for original, translated in zip(_text_slots(source_node), _text_slots(result_node)):
        original_text = _slot_text(original)
        if checker.ratio(original_text) >= checker.minimum_ratio:
            continue
        render = lambda value: serialize_entry(_fragment_node(value))
        pieces = budget.split_text(original_text, render, prompt, checker.pass_list.pattern)
        translated_pieces = []
        for piece in pieces:
            leading = piece[:len(piece) - len(piece.lstrip())]
            trailing = piece[len(piece.rstrip()):]
            raw = render(piece.strip())
            part = next(iter(message_map(raw).values()))
            try:
                response = await _translate_chunk(client, prompt, [part], checker.target_culture, checker, budget, context)
                node = next(iter(entries(parse_resource(response[part.id])).values()))
                from .fluent import pattern_text
                value = pattern_text(node.value)
                translated_pieces.append(leading + value.strip() + trailing)
            except ResponseTruncatedError:
                translated_pieces.append(await _translate_fragment_again(piece, client, prompt, checker,
                                                                         budget.smaller(), depth + 1, context))
        value = "".join(translated_pieces)
        translated.value = _literal_value(value) if isinstance(translated, syntax().ast.StringLiteral) else value
    result = serialize_entry(result_node).rstrip("\n")
    _validate_translated_message(message, result, checker)
    return {message.id: result}


async def _translate_fragment_again(piece, client, prompt, checker, budget, depth, context=()):
    raw = serialize_entry(_fragment_node(piece))
    message = next(iter(message_map(raw).values()))
    response = await _translate_large_message(message, client, prompt, checker, budget, depth, context)
    from .fluent import pattern_text
    value = pattern_text(next(iter(entries(parse_resource(response[message.id])).values())).value)
    leading = piece[:len(piece) - len(piece.lstrip())]
    trailing = piece[len(piece.rstrip()):]
    return leading + value.strip() + trailing


async def _safe_chunk(client, prompt, chunk, checker, budget, context=()):
    payload = "\n\n".join(message.text for message in chunk)
    if len(chunk) == 1 and not budget.fits(payload, prompt):
        return await _translate_large_message(chunk[0], client, prompt, checker, budget, context=context)
    try:
        return await _translate_chunk(client, prompt, chunk, checker.target_culture, checker, budget, context)
    except ResponseTruncatedError:
        if len(chunk) == 1:
            return await _translate_large_message(chunk[0], client, prompt, checker, budget.smaller(), context=context)
        middle = len(chunk) // 2
        first = await _safe_chunk(client, prompt, chunk[:middle], checker, budget, context)
        first.update(await _safe_chunk(client, prompt, chunk[middle:], checker, budget, context))
        return first


def _replace_messages(text, replacements):
    resource = parse_resource(text)
    ast = syntax().ast
    for index, node in enumerate(resource.body):
        if isinstance(node, (ast.Message, ast.Term)):
            from .fluent import entry_id
            key = entry_id(node)
            if key in replacements:
                resource.body[index] = entries(parse_resource(replacements[key]))[key]
    result = serialize_resource(resource)
    parse_resource(result)
    return result


async def translate_file(path, client, prompt, chunk_size, source_text=None, target_culture=None,
                         *, allow_partial=False, dry_run=False, checker=None, budget=None, text=None,
                         messages=None, context=None, save_tokens=False):
    text = read_text(path) if text is None else text
    checker = checker or LanguageChecker("en-US", target_culture, load_pass_list())
    budget = budget or OutputBudget.from_env()
    if messages is None or context is None:
        found, examples = _split_messages(text, source_text, target_culture, checker)
        messages = found if messages is None else messages
        context = examples if context is None else context
    replacements = {}
    changed = False
    for chunk in _chunks(messages, chunk_size, budget, prompt):
        try:
            replacements.update(await _safe_chunk(client, prompt, chunk, checker, budget,
                                                  () if save_tokens else context))
            if allow_partial:
                changed = write_text_if_changed(path, _replace_messages(text, replacements), dry_run) or changed
        except Exception as error:
            raise TranslationFileError(path, len(replacements), changed, error) from error
    if replacements and not allow_partial:
        changed = write_text_if_changed(path, _replace_messages(text, replacements), dry_run)
    return len(replacements), changed


async def translate_files(files, prompt, chunk_size, source_texts=None, target_culture=None, concurrency=2,
                          *, allow_partial=False, dry_run=False, checker=None, budget=None, texts=None,
                          on_event=None, on_usage=None, on_retry=None, ai_config=None, save_tokens=False,
                          plans=None):
    checker = checker or LanguageChecker("en-US", target_culture, load_pass_list())
    budget = budget or OutputBudget.from_env()
    pending, failures = [], []
    texts = texts or {}
    plans = plans or {}
    for path in dict.fromkeys(files):
        if path.name in checker.pass_list.ignored_files:
            if on_event:
                on_event("skipped", path, {})
            else:
                print(f"Исключён файл: {path}")
            continue
        text = ""
        try:
            if path in plans:
                text, messages, context = plans[path]
            else:
                text = texts[path] if path in texts else read_text(path)
                messages, context = _split_messages(text, (source_texts or {}).get(path), target_culture, checker)
            if not messages:
                if on_event:
                    on_event("skipped", path, {})
                continue
            pending.append((path, text, messages, context))
            if dry_run:
                chunks = _chunks(messages, chunk_size, budget, prompt)
                estimates = []
                for chunk in chunks:
                    raw = "\n\n".join(item.text for item in chunk)
                    if budget.fits(raw, prompt):
                        estimates.append(budget.estimated_output(raw))
                    else:
                        node = next(iter(entries(parse_resource(chunk[0].text)).values()))
                        render = lambda value: serialize_entry(_fragment_node(value))
                        for slot in _text_slots(node):
                            value = _slot_text(slot)
                            if checker.ratio(value) < checker.minimum_ratio:
                                pieces = budget.split_text(value, render, prompt, checker.pass_list.pattern)
                                estimates.extend(budget.estimated_output(render(piece)) for piece in pieces)
                print(f"План перевода: {path}; ключи={','.join(message.id for message in messages)}; "
                      f"блоки={len(estimates)}; оценки ответа={estimates}; безопасное окно={budget.capacity}")
        except Exception as error:
            failures.append(TranslationFailure(path, 0, False, str(error)))
            if on_event:
                on_event("failed", path, {"error": str(error), "source": text, "response": None})
    if dry_run or not pending:
        return TranslationRunResult(0, 0, tuple(item.path for item in failures), tuple(failures))
    retry_callback = (lambda *items: on_retry(_current_file.get(), *items)) if on_retry else None
    client = OpenAICompatibleClient(ai_config or AiConfig.from_env(), on_usage,
                                    quiet=bool(on_event), on_retry=retry_callback)
    semaphore = asyncio.Semaphore(concurrency)

    async def one(path, text, messages, context):
        async with semaphore:
            _current_file.set(path)
            if on_event:
                on_event("started", path, {})
            if not on_event:
                print(f"Перевод: {path}", file=sys.stderr, flush=True)
            try:
                result = await translate_file(path, client, prompt, chunk_size, target_culture=target_culture,
                                              allow_partial=allow_partial, checker=checker, budget=budget,
                                              text=text, messages=messages, context=context, save_tokens=save_tokens)
                if on_event:
                    on_event("completed", path, {"text": read_text(path), "messages": result[0]})
                return result
            except TranslationFileError as error:
                if on_event:
                    on_event("failed", path, {"error": str(error.error), "source": text,
                                              "response": getattr(error.error, "ai_response", None)})
                return TranslationFailure(path, error.translated_messages, error.changed, str(error.error))
            except Exception as error:
                if on_event:
                    on_event("failed", path, {"error": str(error), "source": text, "response": None})
                return TranslationFailure(path, 0, False, str(error))

    results = await asyncio.gather(*(one(path, text, messages, context)
                                     for path, text, messages, context in pending))
    translated = changed = 0
    for result in results:
        if isinstance(result, TranslationFailure):
            failures.append(result)
            translated += result.translated_messages
            changed += result.changed
        else:
            translated += result[0]
            changed += result[1]
    return TranslationRunResult(translated, changed, tuple(item.path for item in failures), tuple(failures))


def run_translate_files(*args, **kwargs):
    return asyncio.run(translate_files(*args, **kwargs))
