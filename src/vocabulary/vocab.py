from __future__ import annotations
from collections import UserDict
import json
from pathlib import Path



class Vocab(UserDict[str, int]):

    def inverse(self) -> dict[int, str]:
        return { index: word for word, index in self.items() }

    @classmethod
    def from_sequences(cls, sequences: list[list[str]], start_index: int=0) -> Vocab:
        vocab = cls()
        index = start_index

        for sent in sequences:
            for item in sent:
                if item not in vocab:
                    vocab[item] = index
                    index += 1

        return vocab

    @classmethod
    def from_file(cls, path: Path | str) -> Vocab:
        if Path(path).exists():
            with open(path) as f:
                vocab = cls(json.load(f))
            return vocab
        else:
            raise FileNotFoundError()

    def to_file(self, path: Path | str) -> None:
        with open(path, 'w') as f:
            json.dump(self.data, f)
