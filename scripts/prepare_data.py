"""Validate, normalize, split, and tokenize the original dialogue corpus."""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tinytalk.tokenizer import CharTokenizer


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    source = ROOT / "data" / "dialogues.jsonl"
    records: list[dict[str, str]] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        user, assistant = clean_text(row["user"]), clean_text(row["assistant"])
        if user and assistant:
            records.append({"user": user, "assistant": assistant})
        else:
            raise ValueError(f"Empty user or assistant text on line {line_number}")
    if len(records) < 2:
        raise ValueError("At least two valid dialogues are required")

    random.Random(42).shuffle(records)
    val_count = max(1, round(len(records) * 0.1))
    test_count = max(1, round(len(records) * 0.1))
    validation = records[:val_count]
    testing = records[val_count:val_count + test_count]
    training = records[val_count + test_count:]

    def serialize(items: list[dict[str, str]]) -> str:
        return "\n\n".join(f"User: {item['user']}\nAssistant: {item['assistant']}" for item in items)

    train_text, val_text, test_text = serialize(training), serialize(validation), serialize(testing)
    tokenizer = CharTokenizer.from_text(train_text)
    output = ROOT / "artifacts"
    output.mkdir(exist_ok=True)
    (output / "train.txt").write_text(train_text, encoding="utf-8")
    (output / "validation.txt").write_text(val_text, encoding="utf-8")
    (output / "test.txt").write_text(test_text, encoding="utf-8")
    tokenizer.save(output / "vocab.json")
    print(f"Prepared {len(training)} training, {len(validation)} validation, and {len(testing)} test dialogues.")
    print(f"Training characters: {len(train_text):,}; vocabulary size: {tokenizer.vocab_size}.")
    print(f"Wrote data to {output}")


if __name__ == "__main__":
    main()
