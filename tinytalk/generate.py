"""Generate a continuation from a trained TinyTalk checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from tinytalk.model import TinyTransformer
from tinytalk.tokenizer import CharTokenizer

ROOT = Path(__file__).resolve().parents[1]


def load_model(device: torch.device) -> tuple[TinyTransformer, CharTokenizer]:
    checkpoint_path = ROOT / "artifacts" / "tinytalk.pt"
    vocab_path = ROOT / "artifacts" / "vocab.json"
    if not checkpoint_path.exists() or not vocab_path.exists():
        raise FileNotFoundError("Train the model first with python -m tinytalk.train")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    tokenizer = CharTokenizer.load(vocab_path)
    model = TinyTransformer(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="User: What is a Transformer?\nAssistant:")
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(device)
    ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    print(tokenizer.decode(model.generate(ids, args.max_new_tokens, args.temperature, args.top_k)[0].tolist()))


if __name__ == "__main__":
    main()
