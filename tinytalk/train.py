"""Train TinyTransformer on next-character prediction."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from tinytalk.model import TinyTransformer
from tinytalk.tokenizer import CharTokenizer

ROOT = Path(__file__).resolve().parents[1]


def estimate_loss(model: TinyTransformer, datasets: dict[str, torch.Tensor],
                  block_size: int, batch_size: int, device: torch.device, batches: int = 10) -> dict[str, float]:
    model.eval()
    result = {}
    with torch.no_grad():
        for name, data in datasets.items():
            losses = []
            for _ in range(batches):
                x, y = get_batch(data, block_size, batch_size, device)
                _, loss = model(x, y)
                losses.append(loss.item())
            result[name] = sum(losses) / len(losses)
    model.train()
    return result


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[start:start + block_size] for start in starts])
    y = torch.stack([data[start + 1:start + block_size + 1] for start in starts])
    return x.to(device), y.to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    torch.manual_seed(1337)
    # Small educational batches often run faster with fewer CPU worker threads.
    torch.set_num_threads(min(4, torch.get_num_threads()))
    random.seed(1337)

    train_path = ROOT / "artifacts" / "train.txt"
    val_path = ROOT / "artifacts" / "validation.txt"
    test_path = ROOT / "artifacts" / "test.txt"
    vocab_path = ROOT / "artifacts" / "vocab.json"
    if not train_path.exists() or not val_path.exists() or not vocab_path.exists():
        raise SystemExit("Prepared data is missing. Run: python scripts/prepare_data.py")
    tokenizer = CharTokenizer.load(vocab_path)
    train_ids = torch.tensor(tokenizer.encode(train_path.read_text(encoding="utf-8")), dtype=torch.long)
    val_ids = torch.tensor(tokenizer.encode(val_path.read_text(encoding="utf-8")), dtype=torch.long)
    test_ids = torch.tensor(tokenizer.encode(test_path.read_text(encoding="utf-8")), dtype=torch.long) if test_path.exists() else None

    # Verify all splits meet minimum length requirement before starting training
    all_datasets = {"train": train_ids, "validation": val_ids}
    if test_ids is not None:
        all_datasets["test"] = test_ids
    for name, tensor in all_datasets.items():
        if len(tensor) < args.block_size + 1:
            raise SystemExit(f"Dataset split '{name}' ({len(tensor)} tokens) is too short for block size {args.block_size}; reduce --block-size")

    # Periodic evaluation uses only train and validation to maintain test set integrity
    eval_datasets = {"train": train_ids, "validation": val_ids}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyTransformer(tokenizer.vocab_size, args.d_model, args.heads, args.layers, args.block_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    print(f"Training on {device}; {sum(p.numel() for p in model.parameters()):,} parameters; vocab {tokenizer.vocab_size}.")
    
    for step in range(1, args.steps + 1):
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            losses = estimate_loss(model, eval_datasets, args.block_size, args.batch_size, device)
            loss_str = " | ".join(f"{name} loss {val:.4f}" for name, val in losses.items())
            print(f"step {step:5d} | {loss_str}")
        x, y = get_batch(train_ids, args.block_size, args.batch_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    checkpoint = {"model_state": model.state_dict(), "config": model.config, "seed": 1337}
    torch.save(checkpoint, artifacts / "tinytalk.pt")

    # Evaluate final model on train, validation, and untouched held-out test data
    final_eval_sets = {"train": train_ids, "validation": val_ids}
    if test_ids is not None:
        final_eval_sets["test"] = test_ids
    final_losses = estimate_loss(model, final_eval_sets, args.block_size, args.batch_size, device, batches=20)
    (artifacts / "training_config.json").write_text(
        json.dumps({**model.config, "steps": args.steps, "final_losses": final_losses}, indent=2), encoding="utf-8"
    )
    print(f"Final evaluation on untouched test set: {final_losses}")
    print(f"Saved checkpoint to {artifacts / 'tinytalk.pt'}")


if __name__ == "__main__":
    main()
