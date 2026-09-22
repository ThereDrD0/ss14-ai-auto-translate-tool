from __future__ import annotations

import asyncio
from dataclasses import dataclass
import os
import sys
import time
from typing import Any, Callable
from urllib.parse import urlsplit

from .dependencies import import_or_install


DEFAULT_LOCAL_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_LOCAL_MODEL = "gpt-5.6-luna"
DEFAULT_LOCAL_API_KEY = "local"


@dataclass
class AiEndpoint:
    base_url: str
    model: str
    api_key: str
    proxy: str | None = None
    cooldown_until: float = 0

    @property
    def available(self) -> bool:
        return time.monotonic() >= self.cooldown_until

    def cool_down(self, seconds: int) -> None:
        self.cooldown_until = time.monotonic() + seconds


@dataclass(frozen=True)
class AiConfig:
    endpoints: tuple[AiEndpoint, ...]
    timeout_seconds: int = 300
    cooldown_seconds: int = 60
    max_attempts: int = 0
    max_output_tokens: int = 128000

    @classmethod
    def from_env(cls) -> "AiConfig":
        base_urls = [
            _normalize_base_url(base_url)
            for base_url in _split_secret_list(os.environ.get("TRANSLATE_AI_BASE_URL", DEFAULT_LOCAL_BASE_URL))
        ]
        models = _split_secret_list(os.environ.get("TRANSLATE_AI_MODEL", DEFAULT_LOCAL_MODEL))
        keys = _split_secret_list(os.environ.get("TRANSLATE_AI_KEYS", ""))
        proxies = _split_secret_list(os.environ.get("TRANSLATE_AI_PROXIES", ""))

        if not base_urls:
            raise ValueError("TRANSLATE_AI_BASE_URL is required for AI translation.")
        if not models:
            raise ValueError("TRANSLATE_AI_MODEL is required for AI translation.")
        if not keys:
            if all(_is_local_base_url(base_url) for base_url in base_urls):
                keys = [DEFAULT_LOCAL_API_KEY]
            else:
                raise ValueError("TRANSLATE_AI_KEYS must contain at least one key.")

        endpoint_count = max(len(base_urls), len(models), len(keys), len(proxies) if proxies else 0)
        endpoints = tuple(
            AiEndpoint(
                base_url=base_urls[index % len(base_urls)],
                model=models[index % len(models)],
                api_key=keys[index % len(keys)],
                proxy=proxies[index % len(proxies)] if proxies else None,
            )
            for index in range(endpoint_count)
        )

        config = cls(
            endpoints=endpoints,
            timeout_seconds=int(os.environ.get("TRANSLATE_AI_TIMEOUT_SECONDS", "300")),
            cooldown_seconds=int(os.environ.get("TRANSLATE_AI_COOLDOWN_SECONDS", "60")),
            max_attempts=int(os.environ.get("TRANSLATE_AI_MAX_ATTEMPTS", "0")),
            max_output_tokens=int(os.environ.get("TRANSLATE_AI_MAX_OUTPUT_TOKENS", "128000")),
        )
        if config.timeout_seconds <= 0 or config.cooldown_seconds < 0 or config.max_attempts < 0 or config.max_output_tokens < 64:
            raise ValueError("Некорректные таймаут, ожидание, число попыток или окно ответа")
        return config


class OpenAICompatibleClient:
    def __init__(self, config: AiConfig, on_usage: Callable[[int, int, bool], None] | None = None,
                 quiet: bool = False, on_retry: Callable | None = None):
        self._config = config
        self._on_usage = on_usage
        self._on_retry = on_retry
        self._quiet = quiet
        self._supports_retry = True
        self._endpoint_index = 0
        self._httpx = import_or_install("httpx", "httpx>=0.27,<1")

    async def chat(self, messages: list[dict[str, str]], temperature: float = 0.1, retry: bool = False) -> str:
        attempts = 0
        last_error: Exception | None = None

        while self._config.max_attempts <= 0 or attempts < self._config.max_attempts:
            attempts += 1
            endpoint = await self._next_endpoint()

            try:
                return await self._send(endpoint, messages, temperature, retry or attempts > 1)
            except RateLimitedError as error:
                self._handle_retry(endpoint, attempts, error)
                last_error = error
            except TransientAiError as error:
                self._handle_retry(endpoint, attempts, error)
                last_error = error

        raise RuntimeError(f"Перевод ИИ не удался после {attempts} попыток. Последняя ошибка: {last_error}") from last_error

    def _handle_retry(self, endpoint: AiEndpoint, attempts: int, error: Exception) -> None:
        max_attempts = self._config.max_attempts
        will_retry = max_attempts <= 0 or attempts < max_attempts
        max_attempts_text = "∞" if max_attempts <= 0 else str(max_attempts)
        if self._on_retry:
            self._on_retry("request", attempts, max_attempts, error,
                           self._config.cooldown_seconds if will_retry else 0, will_retry)

        if not self._quiet:
            print(
                "Повтор запроса ИИ: "
                f"адрес={endpoint.base_url} модель={endpoint.model} "
                f"попытка={attempts}/{max_attempts_text} "
                f"повтор={'да' if will_retry else 'нет'} "
                f"ожидание={self._config.cooldown_seconds if will_retry else 0} с "
                f"причина={error}",
                file=sys.stderr,
                flush=True,
            )

        if will_retry:
            endpoint.cool_down(self._config.cooldown_seconds)

    async def _next_endpoint(self) -> AiEndpoint:
        while True:
            for _ in range(len(self._config.endpoints)):
                endpoint = self._config.endpoints[self._endpoint_index]
                self._endpoint_index = (self._endpoint_index + 1) % len(self._config.endpoints)
                if endpoint.available:
                    return endpoint

            await asyncio.sleep(1)

    async def _send(self, endpoint: AiEndpoint, messages: list[dict[str, str]], temperature: float, retry: bool = False) -> str:
        headers = {
            "Authorization": f"Bearer {endpoint.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {"model": endpoint.model, "messages": messages}
        if endpoint.model == "gpt-5.6-luna":
            payload["reasoning_effort"] = "none"
        else:
            payload["temperature"] = temperature
        if self._config.max_output_tokens > 0:
            field = os.environ.get("TRANSLATE_AI_TOKEN_LIMIT_FIELD",
                                   "max_completion_tokens" if endpoint.model == "gpt-5.6-luna" else "max_tokens")
            if field not in {"max_tokens", "max_completion_tokens"}:
                raise ValueError("TRANSLATE_AI_TOKEN_LIMIT_FIELD: max_tokens или max_completion_tokens")
            payload[field] = self._config.max_output_tokens

        url = f"{endpoint.base_url}/chat/completions"

        try:
            async with self._httpx.AsyncClient(
                timeout=self._config.timeout_seconds,
                proxy=endpoint.proxy,
                trust_env=False,
            ) as client:
                response = await client.post(url, headers=headers, json=payload)
        except self._httpx.HTTPError as error:
            raise TransientAiError(f"Сбой запроса к серверу ИИ: {error.__class__.__name__}") from None

        if response.status_code == 429:
            raise RateLimitedError("Сервер ИИ ограничил частоту запросов (429).")

        if response.status_code >= 500:
            raise TransientAiError(f"Сервер ИИ вернул {response.status_code}. {response.text}")

        if response.status_code >= 400:
            raise RuntimeError(f"Сервер ИИ вернул {response.status_code}. {response.text}")

        try:
            data = response.json()
            usage = data.get("usage") or {}
            if self._on_usage and isinstance(usage, dict):
                prompt_tokens, completion_tokens = usage.get("prompt_tokens"), usage.get("completion_tokens")
                if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
                    self._on_usage(prompt_tokens, completion_tokens, retry)
            if data["choices"][0].get("finish_reason") == "length":
                raise ResponseTruncatedError("ИИ обрезал ответ по пределу выходного окна; блок будет уменьшен")
            content = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError):
            raise TransientAiError(
                f"Сервер ИИ вернул некорректный ответ. {response.text}"
            ) from None

        if not isinstance(content, str):
            raise TransientAiError("Сервер ИИ вернул ответ без текста.")

        return content


class RateLimitedError(RuntimeError):
    pass


class TransientAiError(RuntimeError):
    pass


class ResponseTruncatedError(RuntimeError):
    pass


def _split_secret_list(value: str) -> list[str]:
    normalized = value.replace("\r", "\n").replace(",", "\n")
    return [item.strip() for item in normalized.split("\n") if item.strip()]


def _normalize_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if "://" not in normalized:
        normalized = f"http://{normalized}"
    return normalized


def _is_local_base_url(base_url: str) -> bool:
    host = urlsplit(base_url).hostname
    return host in {"127.0.0.1", "localhost", "::1"}
