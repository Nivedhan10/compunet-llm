"""A transparent character-level tokenizer with an unknown-character fallback."""
from __future__ import annotations

import json
from pathlib import Path


class CharTokenizer:
    def __init__(self, characters: list[str]):
        self.itos = ["<unk>", *characters]
        self.stoi = {token: index for index, token in enumerate(self.itos)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        unknown = self.stoi["<unk>"]
        return [self.stoi.get(character, unknown) for character in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[index] for index in ids if self.itos[index] != "<unk>")

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.itos, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        tokens = json.loads(Path(path).read_text(encoding="utf-8"))
        if not tokens or tokens[0] != "<unk>":
            raise ValueError("Invalid vocabulary file")
        return cls(tokens[1:])
