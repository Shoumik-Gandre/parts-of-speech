from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from itertools import chain


@dataclass
class CountReplaceTransform:
    threshold: int
    special_token: str

    def fit(self, sequences: list[list[str]], *_) -> CountReplaceTransform:
        frequency = Counter(chain(*sequences)) # Get the word counts
        self.words = { word: count for word, count in frequency.items() if count >= self.threshold }
        return self

    def transform(self, sequences: list[list[str]], *_) -> list[list[str]]:
        
        return [
            [
                (word if word in self.words else self.special_token)
                for word in sequence
            ] 
            for sequence in sequences
        ]