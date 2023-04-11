from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from itertools import chain
from typing import Any
from .base import BaseTransform
import re


@dataclass
class RegexReplaceTransform(BaseTransform):
    regex: str | re.Pattern[str]
    special_token: str

    def __post_init_(self):
        self._regex = re.compile(self.regex)

    def fit(self, sequences: list[list[str]]) -> RegexReplaceTransform:
        return self

    def transform(self, sequences: list[list[str]], *_) -> list[list[str]]:
        return [
            [
                (self.special_token if self._regex.match(word) else word)
                for word in sequence
            ] 
            for sequence in sequences
        ]


@dataclass
class RegexCountReplaceTransform(BaseTransform):
    threshold: int
    regex: str | re.Pattern[str]
    special_token: str

    def __post_init_(self):
        self._regex = re.compile(self.regex)

    def fit(self, sequences: list[list[str]]) -> RegexCountReplaceTransform:
        frequency = Counter(chain(*sequences)) # Get the word counts
        self.words = { word: count for word, count in frequency.items() if count >= self.threshold }
        return self

    def transform(self, sequences: list[list[str]], *_) -> list[list[str]]:
        return [
            [
                (self.special_token if word not in self.words and self._regex.match(word) else word)
                for word in sequence
            ] 
            for sequence in sequences
        ]