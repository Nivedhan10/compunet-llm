# TinyTalk technical guide

## 1. Project goal and design plan

The assessment asks for a language model built from scratch, a training workflow, and a chat application that uses the trained model. TinyTalk implements that as a small decoder-only character Transformer plus a separate retrieval layer and a local web application.

The design separates two jobs:

1. **Learn text patterns:** train the local Transformer from random initialization on the project dialogue corpus. It learns next-character prediction.
2. **Supply reference information:** use BM25 to find saved knowledge, and optionally search a limited set of reference sources for information that is not in the small corpus.

Retrieval does not retrain the Transformer. This is an intentional choice: the model weights remain reproducible and no pretrained LLM or hosted AI API is involved. Updating a retrieval index is safer and more direct than fine-tuning a tiny model on each question or unverified web result.

## 2. System architecture

```text
                         OFFLINE PREPARATION / TRAINING
data/dialogues.jsonl -> prepare_data.py -> train/validation/test + vocab
                                               |
                                               v
                                train.py -> artifacts/tinytalk.pt
                                           artifacts/vocab.json

                              ONLINE CHAT REQUEST
React UI + localStorage -> FastAPI validation/context handling
                                  |
                                  +-> user-taught facts or developer templates
                                  +-> live reference lookup (allowlisted URLs)
                                  |       +-> persist successful result with source
                                  +-> local BM25 knowledge retrieval
                                  `-> trained Transformer generation -> fallback
```

The API keeps no server-side conversation session. React sends recent turns on each request and keeps the transcript in browser `localStorage`. The API validates roles and message sizes, constructs a role-labelled prompt, and fits the prompt to the configured context size.

## 3. Repository structure

| Path | Responsibility |
| --- | --- |
| `data/dialogues.jsonl` | Original training dialogue records (`user`, `assistant`) |
| `data/knowledge.jsonl` | Curated/imported/user-taught and saved online Q&A records |
| `scripts/prepare_data.py` | Whitespace cleanup, deterministic split, text serialization, tokenizer creation |
| `scripts/import_knowledge.py` | Optional import from URL, JSONL, text, or Markdown |
| `tinytalk/tokenizer.py` | Character vocabulary, encode/decode, JSON persistence |
| `tinytalk/model.py` | Causal attention, Transformer blocks, model, autoregressive sampler |
| `tinytalk/train.py` | Random window batches, optimizer, loss estimates, checkpoint and config output |
| `tinytalk/generate.py` | Load trained model/tokenizer and generate a continuation |
| `tinytalk/rag.py` | BM25 index, fuzzy token matching, knowledge persistence |
| `tinytalk/online_search.py` | Entity/context heuristics, reference search, URL allowlist, source excerpts |
| `tinytalk/dev_assistant.py` | Fixed programming templates linked to documentation |
| `tinytalk/api.py` | FastAPI health, chat, and RAG endpoints; response routing |
| `frontend/src/` | React interface, local history, loading and status states |
| `tests/test_tinytalk.py` | Unit and API/inference behavior tests |
| `artifacts/` | Prepared splits, vocabulary, model checkpoint, and training metrics |

## 4. Dataset preparation and tokenizer

Each non-empty line in `data/dialogues.jsonl` is a JSON object with a `user` and an `assistant` string. `scripts/prepare_data.py` normalizes repeated whitespace, rejects empty sides, and shuffles records with seed 42. At 120 records, it assigns 12 to validation, 12 to test, and the remaining 96 to training.

Each dialogue is serialized with role prefixes:

```text
User: What is attention?
Assistant: Attention lets a model weigh information from earlier positions.
```

The tokenizer is built from **training text only**. `CharTokenizer` maps each distinct character to an integer and reserves ID zero for `<unk>`. An unseen character encodes to `<unk>`; decoding skips that marker. The current vocabulary has 60 IDs. A character vocabulary is transparent and easy to implement, but consumes more sequence positions than a subword tokenizer.

## 5. Transformer implementation

The current default has 64 model dimensions, four attention heads, two decoder blocks, dropout 0.1, a 128-character context, and about 112K parameters. Its weights are initialized by TinyTalk's own initialization routine; the checkpoint is produced by the project training script.

### Embeddings and positions

For token ID (t_i) at position (i), the model adds a learned token embedding and learned position embedding:

\[
x_i = E_{token}(t_i) + E_{position}(i)
\]

Position embeddings let attention distinguish ordering. The model rejects inputs longer than its context window; the API trims the active question body while preserving `User: ` and `\nAssistant:` markers.

### Causal multi-head self-attention

Each block projects hidden states into query, key, and value tensors and splits the feature dimension into heads. For a head, the mathematical operation is:

\[
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}+M\right)V
\]

Here, (d_k) is the per-head key size. The causal mask (M) prevents a position from attending to future characters. The implementation calls PyTorch's `scaled_dot_product_attention(..., is_causal=True)`; the model architecture and training remain project code, while PyTorch supplies the optimized attention kernel.

### Transformer block and output head

Each block uses Pre-LayerNorm residual sublayers:

\[
x' = x + \operatorname{Attention}(\operatorname{LayerNorm}(x))
\]
\[
x_{next} = x' + \operatorname{FFN}(\operatorname{LayerNorm}(x'))
\]

The feed-forward network expands to four times the hidden width, applies GELU, then projects back. A final normalization and linear head produce one logit per vocabulary character. The output-head weight is the same parameter object as the input token embedding weight (weight tying), reducing parameters and connecting input/output character representations.

## 6. Training, evaluation, and generation

The data is encoded as one character stream per split. `get_batch` samples random starting positions and forms input/target windows offset by one character. For each batch, the model predicts the next character at every position and uses cross-entropy loss. AdamW updates parameters; gradients are clipped to norm 1.0. The configured random seed is 1337.

Periodic loss estimates sample only training and validation windows. The test split is checked for sufficient length before training and evaluated only once at the end. The current final estimates in `artifacts/training_config.json` are:

| Split | Sampled loss |
| --- | ---: |
| Train | 1.1713 |
| Validation | 1.8336 |
| Test | 1.6998 |

Each estimate averages 20 batches of 16 windows (320 windows). This is a sampled estimate, not a complete pass over all possible windows. With only 12 dialogues in each held-out split, it is a useful training diagnostic but weak evidence of generalization.

Generation starts from a prompt and repeatedly samples the next character. Temperature adjusts sampling sharpness; top-k restricts candidates; repetition penalty discourages repeated characters/tokens. Generation can stop at a configured EOS token or a conversation delimiter. In the API, the default fallback limit is 64 new characters, configurable per request up to 160. The model may still produce ungrammatical or irrelevant text because it is small and trained on little data.

## 7. Retrieval and online knowledge

### Local BM25 RAG

`TinyRAG` loads `data/dialogues.jsonl` and `data/knowledge.jsonl`. It tokenizes words with regular expressions and builds a BM25-style lexical index. User-question tokens receive extra weight. Stopwords and generic terms are reduced or excluded, and longer unknown words can use a high-threshold `difflib` fuzzy match. Retrieval also applies subject-overlap and prompt-coverage checks to reduce unrelated matches.

BM25 ranks lexical relevance; it does not perform semantic reasoning or fact verification. A high retrieval score means that words overlap, not that the answer is true. The API returns local matches with a `rag_grounded` source label.

### Dynamic reference lookup

For queries not handled by user-taught facts or developer templates, the API tries online retrieval before saved online knowledge and the local RAG/model fallback. This ordering lets a successful live lookup refresh information that may change. Search routes include Wikipedia summaries/search, Stack Exchange for programming questions, Hugging Face for model queries, and DuckDuckGo as a discovery service. DuckDuckGo results are only accepted when their destination is HTTPS on an allowlisted reference domain defined in `online_search.py`.

The entity resolver is a lightweight rule-based aid, not a general named-entity recognition model. For follow-up pronouns it considers prior user turns, not assistant-generated prose, to reduce accidental subject switching. Search responses present retrieved text and a source URL; they do not make categorical life/retirement claims from keyword matching.

Successful online results are saved in `data/knowledge.jsonl` with a source marker and an explicit resolved subject in the record key. The cache can be reused when appropriate, but time-sensitive external matches are not served as current when live lookup fails. The file can grow over time and should be inspected/curated.

### Manual knowledge import

`scripts/import_knowledge.py --url ...` extracts heading/paragraph Q&A records; `--file ...` accepts JSONL, Markdown, or text. By default it appends to `data/knowledge.jsonl`. The importer currently does **not** enforce the online search allowlist, validate truth, or refresh stale content. Only import trusted sources, and review the generated records before using them. Use `--output` to target another JSONL file, but note that the default RAG index only reads its configured data files.

### Why retrieval instead of automatic fine-tuning?

An answer fetched from a website is not automatically correct, and a user message is not necessarily trustworthy. Re-training on either without review risks contaminating the corpus and changing model behavior unpredictably. TinyTalk therefore “learns” new reference records for retrieval, but does not update model weights during chat. A fully automated continual-training system would need source review, data provenance, deduplication, evaluation gates, rollback, and privacy controls; that is outside this educational assessment.

## 8. API and frontend integration

FastAPI defines:

- `GET /health`: checks checkpoint and vocabulary files and attempts to load the model bundle.
- `GET /rag/stats`: reports indexed document count and available data files.
- `POST /rag/reload`: reloads JSONL records into the in-memory BM25 index.
- `POST /chat`: validates a message, optional recent history, and generation limit; routes to user-taught facts/developer templates, online retrieval, local RAG, then model generation.

Pydantic limits messages to 2,000 characters, history to 20 entries, and generation to 1–160 characters (default 64). The prompt formatter preserves the active user/assistant role markers and allocates remaining context to recent history. The API allows the local Vite origins in CORS settings.

React sends the message and recent transcript, displays source/status labels, and keeps chat history in `localStorage`. It does not make any external AI calls; only the backend performs optional HTTP reference retrieval.

## 9. Tests and verification

Run tests from the repository root:

```powershell
python tests/test_tinytalk.py -v
```

The suite contains 25 tests covering tokenizer roundtrip/unknown IDs/serialization; model forward shapes, loss, weight tying, causality, generation, and EOS; BM25 relevance and rejection; query cleaning, source allowlisting, user-history entity resolution, source routing; developer templates; and API schema, health, prompt truncation, default model fallback, and chat inference. External search route tests mock source results, so they do not prove that third-party sites are reachable or that live content is correct.

## 10. Setup and operation

1. Install Python 3.10+ and a platform-appropriate PyTorch 2.0+ build.
2. Install API dependencies with `python -m pip install -r requirements.txt`.
3. Run `python scripts/prepare_data.py` if prepared artifacts are absent or the source corpus changed.
4. Run `python -m tinytalk.train --steps 3500 --block-size 128` if the checkpoint is absent or retraining is desired.
5. Start the API with `uvicorn tinytalk.api:app --reload`.
6. In another terminal, run `cd frontend`, `npm install`, `npm run dev` and open the Vite URL.
7. Verify the API at `http://localhost:8000/health` and run the test command above.

For a direct model sample, run `python -m tinytalk.generate --prompt "User: What is attention?\nAssistant:" --max-new-tokens 100`.

## 11. Assessment boundaries and improvement opportunities

TinyTalk implements a Transformer language model from scratch in the educational sense: the architecture, initialization, objective, optimization loop, and sampling path are repository code using PyTorch primitives. The PyTorch scaled attention kernel is an optimized library operation. The project does not implement a foundation-scale LLM, and retrieval is an application layer around the trained model.

High-value future improvements include a larger and more diverse licensed corpus, repeated-seed evaluation, a larger held-out test set, an explicit benchmark for retrieval relevance/factuality, better general entity extraction without adding person-name maps, cache expiry and provenance review, query-level source diagnostics, and deployment/security hardening. None of these should be described as complete until implemented and tested.
