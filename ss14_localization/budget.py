from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, replace
from functools import lru_cache

from .dependencies import import_or_install


@lru_cache(maxsize=8)
def _encoding(name):
    module = import_or_install("tiktoken", "tiktoken>=0.7,<1")
    return module.get_encoding(name)


@dataclass(frozen=True)
class OutputBudget:
    max_tokens: int = 16384
    safety: float = 0.65
    expansion: float = 3.0
    reserve: int = 128
    tokenizer: str = "bytes"
    max_input_tokens: int = 922000

    def __post_init__(self):
        if (
            self.max_tokens < 64
            or not 0 < self.safety < 1
            or self.expansion < 1
            or not math.isfinite(self.expansion)
            or self.reserve < 0
            or self.max_input_tokens < 0
        ):
            raise ValueError(
                "Некорректный бюджет ответа: tokens >= 64, 0 < safety < 1, "
                "expansion >= 1, reserve >= 0"
            )
        if self.capacity <= self.reserve + 8:
            raise ValueError("Окно ответа слишком мало относительно заданного запаса")

    @classmethod
    def from_env(cls):
        return cls(
            int(os.environ.get("TRANSLATE_AI_MAX_OUTPUT_TOKENS", "16384")),
            float(os.environ.get("TRANSLATE_AI_OUTPUT_SAFETY", "0.65")),
            float(os.environ.get("TRANSLATE_AI_OUTPUT_EXPANSION", "3")),
            int(os.environ.get("TRANSLATE_AI_OUTPUT_RESERVE", "128")),
            os.environ.get("TRANSLATE_AI_TOKENIZER", "bytes"),
            int(os.environ.get("TRANSLATE_AI_MAX_INPUT_TOKENS", "922000")),
        )

    @property
    def capacity(self):
        return math.floor(self.max_tokens * self.safety)

    def tokens(self, text):
        # Conservative byte-BPE upper estimate. Never infer an encoding from a model name.
        if self.tokenizer == "bytes":
            return len(text.encode("utf-8"))
        return len(_encoding(self.tokenizer).encode(text, disallowed_special=()))

    def estimated_output(self, text):
        return math.ceil(self.tokens(text) * self.expansion) + self.reserve

    def fits(self, text, prompt=""):
        return self.estimated_output(text) <= self.capacity and (
            not self.max_input_tokens
            or self.tokens(prompt + text) + self.reserve <= self.max_input_tokens
        )

    def smaller(self):
        return replace(self, expansion=self.expansion * 2)

    def split_text(self, text, render, prompt="", protected=None):
        """Split at whitespace when possible; never split a preserved phrase or markup tag."""
        forbidden = []
        if protected:
            forbidden.extend((match.start(), match.end()) for match in protected.finditer(text))
        forbidden.extend(
            (match.start(), match.end()) for match in re.finditer(r"\[[^\]]*\]|<[^>]*>", text)
        )
        pieces = []
        start = 0
        while start < len(text):
            low, high = start + 1, len(text)
            best = start
            while low <= high:
                middle = (low + high) // 2
                if self.fits(render(text[start:middle]), prompt):
                    best, low = middle, middle + 1
                else:
                    high = middle - 1
            if best == start:
                raise ValueError("Даже один символ с FTL-обёрткой не помещается в окно ответа")
            if best < len(text):
                for left, right in forbidden:
                    if left < best < right:
                        best = left
                        break
                if best <= start:
                    raise ValueError(
                        "Неделимое название из pass-листа или тег не помещается в окно ответа"
                    )
                whitespace = [match.end() for match in re.finditer(r"\s+", text[start:best])]
                if whitespace:
                    boundary = start + whitespace[-1]
                    if not any(left < boundary < right for left, right in forbidden):
                        best = boundary
            piece = text[start:best]
            if not self.fits(render(piece), prompt):
                raise ValueError("Не удалось подобрать безопасный размер фрагмента")
            pieces.append(piece)
            start = best
        return pieces
