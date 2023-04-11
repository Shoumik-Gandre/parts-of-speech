from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from itertools import chain
from .pseudowords import match
from ..base import BaseTransform
    

@dataclass
class PseudoWordReplaceTransform(BaseTransform):
    threshold: int
    unk_token: str

    def fit(self, sequences: list[list[str]], *_) -> PseudoWordReplaceTransform:
        frequency = Counter(chain(*sequences)) # Get the word counts
        self.words = { word: count for word, count in frequency.items() if count >= self.threshold }
        return self

    def transform(self, sequences: list[list[str]], *_) -> list[list[str]]:
        
        return [
            [
                (match(word, self.unk_token) if word not in self.words else word)
                for word in sequence
            ] 
            for sequence in sequences
        ]