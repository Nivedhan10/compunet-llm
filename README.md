# TinyTalk: a language model built from scratch

TinyTalk is an educational, character-level causal Transformer implemented in PyTorch. It is trained from random initialization on the small, original dialogue corpus in `data/`. A FastAPI service loads the checkpoint, and a React interface sends messages and keeps conversation history in the browser.

> This is a learning demo, not a capable general-purpose assistant. Its vocabulary, corpus, and model are intentionally small.

## Architecture

```text
Original dialogue corpus -> deterministic train/validation split -> character vocabulary
                                                   |
                                                   v
React UI -> FastAPI -> prompt formatter -> TinyTalk Transformer -> generated assistant text
```

The model predicts the next character from all preceding characters. It has token and learned position embeddings, masked multi-head self-attention, feed-forward layers, residual connections, and layer normalization. No pretrained weights or external AI services are used.

## Setup

Python 3.10+ is recommended. Install PyTorch 2.0 or newer (required for `scaled_dot_product_attention`) from the [official PyTorch install selector](https://pytorch.org/get-started/locally/), then install the remaining dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/prepare_data.py
python -m tinytalk.train --steps 3500 --block-size 128
uvicorn tinytalk.api:app --reload
```

In another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the Vite URL (usually http://localhost:5173). The backend listens on http://localhost:8000. If no checkpoint exists, the API returns HTTP 503 with instructions to train it.

## Training and evaluation

`python -m tinytalk.train --steps 3500 --block-size 128` saves `artifacts/tinytalk.pt`, `artifacts/vocab.json`, and records final evaluation loss across train, validation, and test splits. The default model is 64-dimensional with two Transformer blocks and a 128-character context. Periodic evaluation monitors training and validation loss, while the held-out test split is evaluated once at the conclusion of training. Reported evaluation losses are sampled window estimates (averaging 20 random windows per split) providing diagnostic generalization metrics. Reduce `--steps` for a quick run, or increase it for a more converged toy model. CPU training is fast and CUDA is used automatically when available.

To sample a continuation directly:

```powershell
python -m tinytalk.generate --prompt "User: What is attention?\nAssistant:" --max-new-tokens 180
```

## Running Tests

An automated test suite validates the tokenizer, causal self-attention properties, weight tying, Transformer output shapes, and FastAPI endpoints:

```powershell
python tests/test_tinytalk.py
```

## Project structure

```text
data/dialogues.jsonl       Original self-authored dialogue examples (120 records)
scripts/prepare_data.py    Cleaning and deterministic 3-way split (train/val/test)
tinytalk/tokenizer.py      Character vocabulary and encode/decode
tinytalk/model.py          Decoder-only Transformer with Pre-LN and weight tying
tinytalk/train.py          Batching, loss, validation/test metrics, checkpointing
tinytalk/generate.py       Sampling from a trained checkpoint with top-k/temp
tinytalk/api.py            FastAPI chat endpoint with dynamic context budget
tests/test_tinytalk.py     Automated test suite (tokenizer, model, causality, API)
frontend/                  React + Vite chatbot with suggestion chips
docs/TECHNICAL.md          Component-by-component architectural explanation
artifacts/                 Generated splits (train/val/test), vocab, and checkpoint
```

## GitHub and limitations

This folder is initialized as a local Git repository. Create an empty GitHub repository, add its URL as `origin`, then push. You will need to provide the repository URL or repository access before I can publish it. The compact trained checkpoint is included in the source tree.

Character-level generation may be slow and can produce malformed or repetitive answers. The corpus is intentionally small; the model demonstrates the mechanics, not production chatbot quality.
