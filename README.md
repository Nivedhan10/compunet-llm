# TinyTalk

TinyTalk is an educational, character-level language model and chatbot built to demonstrate the main steps in a small NLP system: preparing a dataset, building a tokenizer, training a decoder-only Transformer from random initialization, and serving it through an API and web interface.

It uses **PyTorch**, **FastAPI**, and **React + Vite**. It does not load pretrained model weights or call a hosted LLM or AI API. Optional web retrieval uses ordinary HTTP requests to reference/search services; it is not another language model.

> TinyTalk is a learning project, not a general-purpose or production-grade assistant. Its Transformer has about 112K parameters, a 60-character vocabulary, and a small training corpus. Open-world answers depend on retrieval and can be incomplete or incorrect.

## How a chat request works

```text
React chat and localStorage history
                 |
                 v
FastAPI validates request and keeps recent context
                 |
                 +--> user-taught facts / developer response templates
                 |
                 +--> live reference search (allowlisted result domains)
                 |       `--> save sourced result into the local knowledge file
                 |
                 +--> local BM25 retrieval (curated and saved records)
                 |
                 `--> TinyTalk Transformer generation (default 64 characters)
                         `--> safe fallback if generation cannot respond
```

The live-search and local-retrieval paths provide information to the chatbot; they do not update model weights. The Transformer itself is trained separately with next-character prediction. This keeps the project within the assessment constraint against pretrained models and external AI APIs.

## Technology stack

| Area | Technology | Role |
| --- | --- | --- |
| Model and training | Python 3.10+, PyTorch 2.0+ | Custom decoder-only Transformer, optimization, checkpointing, generation |
| Tokenizer | Python standard library | Character vocabulary, encoding, decoding, `<unk>` fallback |
| API | FastAPI, Pydantic, Uvicorn | Request validation, health/status, chat and RAG endpoints |
| Retrieval | Python standard library BM25; HTTP search clients | Local knowledge matching and optional sourced web retrieval |
| Web UI | React 18, Vite | Chat experience, suggestions, status and browser history |
| Tests | `unittest` | Model, tokenizer, retrieval, and API behavior |

## Repository map

```text
data/dialogues.jsonl       Original user/assistant training examples
data/knowledge.jsonl       Curated, imported, user-taught, and saved online records
scripts/prepare_data.py    Normalize, deterministically split, and tokenize data
scripts/import_knowledge.py Optional website/file knowledge importer
tinytalk/tokenizer.py      Character tokenizer and vocabulary serialization
tinytalk/model.py          Attention, Transformer block, and language model
tinytalk/train.py          Batching, optimization, evaluation, checkpoint creation
tinytalk/generate.py       Load a checkpoint and sample text from it
tinytalk/rag.py            BM25 retrieval and JSONL knowledge persistence
tinytalk/online_search.py  Reference search, source allowlist, and query handling
tinytalk/api.py            FastAPI request flow and model/RAG integration
frontend/src/              React application and styles
frontend/dist/             Built frontend assets
tests/test_tinytalk.py     Automated test suite
artifacts/                 Prepared splits, vocabulary, checkpoint, metrics
docs/TECHNICAL.md          Detailed design and operational explanation
```

## Installation and run

Use Python 3.10 or newer and Node.js 18 or newer. Install a platform-appropriate PyTorch 2.0+ build from the [official PyTorch installation selector](https://pytorch.org/get-started/locally/); this project uses `torch.nn.functional.scaled_dot_product_attention`. Then install the API dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Prepare the artifacts and train the model (skip training if using the included compatible checkpoint and vocabulary):

```powershell
python scripts/prepare_data.py
python -m tinytalk.train --steps 3500 --block-size 128
```

Start the API in one terminal:

```powershell
uvicorn tinytalk.api:app --reload
```

Start the React development server in a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the Vite URL printed by the command, usually `http://localhost:5173`. The API is at `http://localhost:8000`; interactive web retrieval requires internet connectivity. `GET /health` reports `ready` only when the checkpoint and vocabulary can be loaded.

### Production frontend build

```powershell
cd frontend
npm install
npm run build
```

The generated static files are written to `frontend/dist/`. Hosting those files is separate from starting the FastAPI API.

## Data preparation and training

`data/dialogues.jsonl` contains 120 original technical user/assistant examples. `scripts/prepare_data.py` cleans whitespace, shuffles with seed 42, makes a deterministic 80/10/10 split, and builds the character vocabulary from the training text only. The current split contains 96 training dialogues, 12 validation dialogues, and 12 test dialogues; the vocabulary contains 60 IDs including `<unk>`.

Training predicts the next character in fixed-size windows. It uses AdamW, gradient clipping, and periodic train/validation loss estimates. The test split is not used for training or periodic monitoring; it is evaluated once at the end. The saved checkpoint is `artifacts/tinytalk.pt`, the matching tokenizer is `artifacts/vocab.json`, and metrics/configuration are recorded in `artifacts/training_config.json`.

The checked-in configuration records 3,500 steps and sampled final losses of approximately 1.1713 (train), 1.8336 (validation), and 1.6998 (test). Each split estimate averages 20 batches of 16 random windows (320 windows). These are small-sample diagnostic estimates, not exhaustive evaluation or proof of language ability.

To generate from the model directly:

```powershell
python -m tinytalk.generate --prompt "User: What is attention?`nAssistant:" --max-new-tokens 100
```

## Tests

Run the suite from the repository root:

```powershell
python tests/test_tinytalk.py -v
```

Search-routing tests use mocked reference responses so the suite does not depend on third-party sites being online. The suite checks behavior such as tokenizer roundtrips, model shapes, causality, weight tying, source allowlisting, entity resolution from user history, retrieval, API validation, and generation.

## Knowledge retrieval and its limits

`tinytalk/rag.py` indexes Q&A records from the corpus and `data/knowledge.jsonl` with BM25. It applies lexical overlap checks and a conservative fuzzy match for longer words. `tinytalk/api.py` tries a live reference lookup before reusing saved online knowledge, so a successful search can refresh a matching saved record. Results are cited and cached locally with an explicit subject in the question key; time-sensitive external cache entries are not presented as current when live search fails.

The online search code checks result URLs against an allowlist in `tinytalk/online_search.py` (Wikipedia, Stack Overflow/Stack Exchange, Hugging Face, and selected official documentation domains). It can use DuckDuckGo as a discovery service, but only accepts result URLs on the allowlist. Other provider APIs have fixed endpoints on their respective domains. The allowlist is a code-level policy, not a complete web-safety system.

The optional `scripts/import_knowledge.py` command can import a URL, JSONL, text, or Markdown file into the knowledge file. That manual importer currently accepts arbitrary URLs and should only be used with sources you trust. Retrieved or imported text is not automatically fact-checked. BM25 selects relevant text; it does not understand whether a claim is true or current. User-taught facts and online results persist in `data/knowledge.jsonl`, which can grow and should be reviewed periodically.

TinyTalk does **not** retrain itself after a conversation. RAG adds/retrieves records without changing model weights. Automatic fine-tuning on arbitrary questions or search output would risk storing misinformation, contaminating the training set, and making experiments less reproducible.

## Known limitations

- A 112K-parameter character model trained on 120 dialogues has limited language and reasoning ability; its generated text can be malformed or repetitive.
- Character-level generation is slower and less efficient than subword-token generation.
- Search, entity extraction, and follow-up resolution use lightweight rules. They can fail on ambiguous names, unusual wording, or questions whose answer is absent from the selected source.
- Search requires network access. If it fails, TinyTalk may use local knowledge or generate a low-quality answer; a citation does not guarantee correctness.
- BM25 is lexical retrieval, not semantic search, and cached/imported knowledge needs maintenance.
- The browser stores chat history in localStorage. The API does not provide user accounts or server-side conversation storage.

For the implementation details and design trade-offs, see [docs/TECHNICAL.md](docs/TECHNICAL.md).
