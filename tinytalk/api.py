"""FastAPI wrapper for the locally trained TinyTalk model."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from tinytalk.generate import load_model

app = FastAPI(title="TinyTalk API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173",
                   "http://localhost:5174", "http://127.0.0.1:5174"],
                   allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["*"])


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1, max_length=2000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=20)


class ChatResponse(BaseModel):
    reply: str


def format_prompt(message: str, history: list[ChatMessage], block_size: int = 128) -> str:
    """Format dialogue prompt preserving role markers and fitting within block_size."""
    user_prefix = "User: "
    assistant_suffix = "\nAssistant:"
    overhead = len(user_prefix) + len(assistant_suffix)
    max_question_chars = max(10, block_size - overhead)

    user_message = message.strip()
    if len(user_message) > max_question_chars:
        user_message = user_message[-max_question_chars:]
    current_turn = f"{user_prefix}{user_message}{assistant_suffix}"

    history_budget = max(0, block_size - len(current_turn) - 10)
    prompt_turns = []
    accumulated_len = 0
    for turn in reversed(history):
        role_label = "User" if turn.role == "user" else "Assistant"
        line = f"{role_label}: {turn.content.strip()}"
        if accumulated_len + len(line) + 1 <= history_budget:
            prompt_turns.insert(0, line)
            accumulated_len += len(line) + 1
        else:
            break

    prompt_turns.append(current_turn)
    return "\n".join(prompt_turns)


@lru_cache(maxsize=1)
def model_bundle():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return load_model(device)


@app.get("/health")
def health() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    has_model = (root / "artifacts" / "tinytalk.pt").exists()
    has_vocab = (root / "artifacts" / "vocab.json").exists()
    if not (has_model and has_vocab):
        return {"status": "needs_training"}
    try:
        model_bundle()
        return {"status": "ready"}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model failed to load: {exc}")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        model, tokenizer = model_bundle()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model service unavailable: {exc}")

    block_size = model.config.get("block_size", 128)
    prompt = format_prompt(request.message, request.history, block_size)

    device = next(model.parameters()).device
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    with torch.inference_mode():
        output = model.generate(ids, max_new_tokens=240, temperature=0.6, top_k=6)[0].tolist()
    reply = tokenizer.decode(output[len(ids[0]):]).split("\nUser:", 1)[0].split("\n\nUser:", 1)[0].strip()
    if not reply:
        reply = "I am a small Transformer learning from scratch. Try asking about attention, embeddings, or training!"
    return ChatResponse(reply=reply)

    device = next(model.parameters()).device
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    with torch.inference_mode():
        output = model.generate(ids, max_new_tokens=240, temperature=0.6, top_k=6)[0].tolist()
    reply = tokenizer.decode(output[len(ids[0]):]).split("\nUser:", 1)[0].split("\n\nUser:", 1)[0].strip()
    if not reply:
        reply = "I am a small Transformer learning from scratch. Try asking about attention, embeddings, or training!"
    return ChatResponse(reply=reply)
