from __future__ import annotations
from dataclasses import dataclass


@dataclass
class VocabEncoderTransform:
    vocab: dict[str, int]
    special_token: str

    def fit(self, sequences: list[list[str]], *_) -> VocabEncoderTransform:
        return self

    def transform(self, sequences: list[list[str]], *_) -> list[list[int]]:
        
        return [
            [
                self.vocab.get(word, self.vocab[self.special_token]) # Label Encode the words
                for word in sequence # Inner Loop
            ] 
            for sequence in sequences # Outer Loop
        ]