# TinyTalk technical documentation

## Goal and scope

TinyTalk demonstrates the path from raw examples to a trained text generator and a small chat UI. It follows the assessment constraints: Python and PyTorch, a custom tokenizer, a Transformer implemented in this repository, and random initialization. It does not call a hosted AI service or load pretrained model weights.

The scale is deliberately educational. The corpus contains original short explanations about language models and the implementation itself. It supports a reproducible demonstration, not broad factual question answering.

## Architecture and folder structure

```text
data/dialogues.jsonl       Original user/assistant examples
scripts/prepare_data.py    Cleans, splits, and serializes the corpus
tinytalk/tokenizer.py      Character to integer vocabulary
tinytalk/model.py          Attention, Transformer block, decoder model
tinytalk/train.py          Batches, optimization, loss reporting, checkpoint
tinytalk/generate.py       Checkpoint loading and sampling
tinytalk/api.py            FastAPI request validation and inference endpoint
frontend/src/              React chat interface and styling
artifacts/                 Generated dataset splits, vocabulary, and weights
```

At runtime React sends the latest message and recent turns to FastAPI. The API formats them as `User:` and `Assistant:` lines, encodes the prompt, and asks the local PyTorch model to continue it. It returns the generated assistant text. The browser owns the history in local storage; the API is stateless.

## Dataset preparation and tokenizer

Each JSON Lines record has a `user` string and an `assistant` string. `prepare_data.py` collapses repeated whitespace, drops empty records, shuffles with a fixed seed (42), and reserves held-out sets for validation and testing (10% validation, 10% test, 80% training). The training, validation, and test dialogues are serialized with role labels. The vocabulary is built only from training text.

`CharTokenizer` maps each distinct training character to an integer. A reserved `<unk>` ID handles characters that appear only in validation, test, or an inference prompt. Decoding joins the corresponding characters. Character tokenization is easy to inspect and has no word segmentation problem, though sequences are longer than with subword tokenization.

## Transformer and self-attention

The input IDs index a learned token embedding. A learned position embedding is added at each sequence location, since attention needs order information. The resulting vectors pass through two decoder blocks by default.

Within a block, a linear layer projects each vector into query, key, and value vectors. Attention scores are scaled query-key dot products:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$
A causal mask ($M$) hides later positions; softmax turns remaining scores into weights that combine the value vectors. Multiple heads perform this calculation in separate feature subspaces. PyTorch's `scaled_dot_product_attention` performs the operation with the causal mask enabled.

Each block uses Pre-LayerNorm: it normalizes before attention and feed-forward, adding outputs directly to the residual stream:
$$x = x + \text{Attention}(\text{LayerNorm}(x))$$
$$x = x + \text{FeedForward}(\text{LayerNorm}(x))$$
The feed-forward network is a two-layer GELU network. After the final normalization, a linear output head produces one score per vocabulary character. The input and output embedding weights are tied. All weights are initialized randomly by this code.

## Training and generation

Training samples fixed-length windows from the training character stream. The input is the window and the target is shifted by one character, teaching next-character prediction. Cross entropy compares each position's vocabulary scores with its correct next character. AdamW updates parameters using gradients from backpropagation; gradient clipping limits unusually large updates.

At intervals, the script estimates average loss on random training and held-out validation windows. Periodic evaluation is strictly restricted to training and validation splits to prevent data leakage. At the end of training, the model is evaluated once on the untouched held-out test split. Note that reported evaluation losses are sampled window estimates (averaging 20 batches of 16 windows, totaling 320 sampled windows per split) rather than an exhaustive single-pass metric. The checkpoint stores the model state and architecture configuration; `vocab.json` stores the matching tokenizer. Both are needed to use the checkpoint. Generation repeatedly predicts the next character, samples from top-k candidates after temperature scaling, and appends the result. The API stops the visible answer when a new `User:` turn begins.

## API and frontend

`GET /health` reports whether both the checkpoint and vocabulary files exist and can be loaded successfully for inference. `POST /chat` accepts a message plus conversation history. To maintain optimal generation quality within the 128-token context window, the API dynamically budgets history to prioritize the active question. If an individual prompt exceeds the context window, the message body is trimmed while preserving the structural `User: ` and `\nAssistant:` role delimiters. The request schema limits text sizes and validates roles. CORS permits the local Vite development origin. If training has not produced a checkpoint, `/chat` returns HTTP 503 with the training command.

React renders the transcript, loading state, error banner, and quick suggested question chips. It submits on Enter and inserts a newline on Shift+Enter. It keeps recent conversation state in `localStorage`, so refreshing the page preserves the browser's transcript. No messages are sent anywhere except the local FastAPI service.

## Automated Verification

An automated test suite under `tests/test_tinytalk.py` contains 12 unit and integration tests verifying:
1. Tokenizer encode/decode roundtrip invertibility.
2. Tokenizer `<unk>` fallback for out-of-vocabulary characters.
3. Tokenizer vocabulary JSON serialization and loading.
4. Transformer forward output tensor shapes and cross-entropy loss computation.
5. Parameter tying between language head and token embeddings.
6. Strict causality property (future tokens cannot alter past token logits).
7. Autoregressive generation output sequence dimensions.
8. Generation loop execution with optional `eos_id`.
9. FastAPI `/health` endpoint verifying model and vocabulary loadability.
10. FastAPI `/chat` input schema validation and error rejection.
11. End-to-end `/chat` inference and response generation.
12. Safe prompt truncation for oversized inputs while preserving `User: ` and `\nAssistant:` role delimiters.

## Install and run

Use Python 3.10 or newer. Install PyTorch 2.0 or newer (required for `F.scaled_dot_product_attention`) using the official PyTorch selector, followed by `python -m pip install -r requirements.txt`. Then run:

```powershell
python scripts/prepare_data.py
python -m tinytalk.train --steps 3500 --block-size 128
uvicorn tinytalk.api:app --reload
```

In another terminal, run `cd frontend`, `npm install`, and `npm run dev`. Open the local Vite address. CUDA is selected when available; otherwise CPU is used (CPU training takes approximately 3-4 minutes).

To see a text-only sample, use `python -m tinytalk.generate --prompt "User: What is attention?\nAssistant:"`. Run training with fewer steps for a quick pipeline run. Increase steps or enrich the original dataset for more stable demonstrations. Since training data is tiny, a lower validation loss only indicates better character prediction on this particular held-out sample; it does not establish general language ability.

## Discussion points

- **Why causal attention?** At generation time, the model has only the prefix, so it must not use future characters during training either.
- **Why report validation loss?** It gives a held-out comparison that can reveal memorization, though this corpus is small.
- **Why character-level?** It keeps vocabulary construction transparent and avoids unknown words, trading that simplicity for longer sequences and slower generation.
- **Why role labels?** They give the next-character objective examples of user turns followed by assistant turns, which the prompt can imitate.
- **What would improve it?** More varied original dialogues, a subword tokenizer, larger training budget, stronger evaluation, and carefully tuned decoding. Those changes still would not make a small from-scratch model equivalent to a pretrained assistant.
