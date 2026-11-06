"""FastAPI wrapper for the locally trained TinyTalk model with RAG knowledge grounding and dynamic online auto-learning."""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from tinytalk.dev_assistant import find_developer_response
from tinytalk.generate import load_model
from tinytalk.online_search import fetch_online_knowledge, detect_intent_and_entity
from tinytalk.rag import TinyRAG

# Limit CPU threads to avoid OpenMP contention in multi-core and container environments
if not torch.cuda.is_available():
    torch.set_num_threads(min(4, torch.get_num_threads()))

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
    max_new_tokens: int | None = Field(default=64, ge=1, le=160)


class ChatResponse(BaseModel):
    reply: str
    source: str = "generator"
    confidence: float | None = None


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


@lru_cache(maxsize=1)
def rag_instance() -> TinyRAG:
    return TinyRAG()


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


@app.get("/rag/stats")
def rag_stats() -> dict:
    rag = rag_instance()
    return {
        "indexed_documents": len(rag.documents),
        "data_files": [f.name for f in rag.data_files if f.exists()]
    }


@app.post("/rag/reload")
def rag_reload() -> dict:
    rag = rag_instance()
    count = rag.reload()
    return {"status": "reloaded", "indexed_documents": count}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    rag = rag_instance()
    msg_clean = request.message.strip()

    # 1. Interactive User Teaching (e.g. "My name is Nivedhan", "I am Nivedhan")
    name_match = re.match(r"^(?:my name is|i am|call me)\s+([a-zA-Z0-9_\- ]+)$", msg_clean, re.I)
    if name_match:
        name = name_match.group(1).strip()
        rag.add_and_save_knowledge("what is my name?", f"Your name is {name}.", source="user_taught")
        rag.add_and_save_knowledge("who am i?", f"You are {name}.", source="user_taught")
        return ChatResponse(
            reply=f"Nice to meet you, {name}! I have stored your name in my knowledge base and will remember it.",
            source="user_taught",
            confidence=1.0
        )

    # Teaching a custom fact: "remember that [fact]" or "teach: [fact]"
    remember_match = re.match(r"^(?:remember that|teach:)\s+(.+)$", msg_clean, re.I)
    if remember_match:
        fact = remember_match.group(1).strip()
        rag.add_and_save_knowledge(fact, fact, source="user_taught")
        return ChatResponse(
            reply=f"I have learned that and saved it to my knowledge base: \"{fact}\"",
            source="user_taught",
            confidence=1.0
        )

    # 2. Check personal name inquiry
    if re.search(r"^(what is my name|who am i)\??$", msg_clean, re.I):
        matches = rag.retrieve("what is my name?", top_k=1, threshold=2.0)
        if matches and "name is" in matches[0].answer:
            return ChatResponse(reply=matches[0].answer, source="user_taught", confidence=1.0)
        return ChatResponse(
            reply="I don't know your name yet! You can tell me by saying 'My name is [your name]', and I will remember it.",
            source="user_taught",
            confidence=None
        )

    # 3. Developer & Codebase Assistance Check (e.g. CRUD in Python, FastAPI, PyTorch, React, Java)
    dev_res = find_developer_response(msg_clean)
    if dev_res:
        reply_code = f"### {dev_res['title']}\n\n{dev_res['content']}\n\n[Official Documentation: {dev_res['doc_url']}]"
        rag.add_and_save_knowledge(msg_clean, reply_code, source=dev_res["doc_url"])
        return ChatResponse(
            reply=reply_code,
            source="developer_code",
            confidence=1.0
        )

    # 4. Context & Intent Analysis
    target_entity, intent, meta = detect_intent_and_entity(msg_clean, request.history)
    is_clarification = meta.get("is_clarification", False)

    # 5. Search current reference sources before using saved online material.
    # This prevents a previously cached fact (e.g. current team/retirement status)
    # from silently winning over newer evidence.
    title, answer_text, source_url = fetch_online_knowledge(msg_clean, request.history)
    if answer_text:
        if source_url and source_url not in answer_text:
            answer_text = f"{answer_text.rstrip()}\n\n[Source: {source_url}]"
        # Cache only with the resolved subject in its key: never store a pronoun-only
        # question like "did he retire?" where it could match another conversation.
        cache_question = f"{target_entity}: {msg_clean}" if target_entity else msg_clean
        rag.add_and_save_knowledge(cache_question, answer_text, source=f"online:{source_url or 'reference'}")
        return ChatResponse(
            reply=answer_text,
            source="online_learned",
            confidence=0.75
        )

    # 6. Local RAG grounding (curated corpus and previously learned knowledge).
    # Clarifications bypass fuzzy local retrieval because it may match the prior
    # subject instead of the correction supplied by the user.
    if not is_clarification:
        matches = rag.retrieve(msg_clean, top_k=1, threshold=2.2)
        if matches:
            top = matches[0]
            is_time_sensitive = intent in {"retirement_status", "life_status"} or bool(
                re.search(r"\b(current|currently|latest|today|now|playing|team|club|roster|score|status)\b", msg_clean, re.I)
            )
            # Never present a previously cached web fact as current when live
            # sources could not be reached. External facts need a fresh citation.
            if is_time_sensitive and (top.source.startswith("online:") or top.source.startswith("http")):
                top = None
            if top is None:
                matches = []
        if matches:
            top = matches[0]
            if top.source.startswith("online:"):
                # Online facts can go stale and are only a fallback when live search
                # is unavailable. Their confidence is deliberately lower.
                return ChatResponse(reply=top.answer, source="online_cached", confidence=0.45)
            return ChatResponse(
                reply=top.answer,
                source="rag_grounded",
                confidence=round(top.confidence, 2)
            )

    # 7. Run the trained from-scratch model for novel questions. The UI does not
    # need to set max_new_tokens; 64 is the default, with a request-level override.
    try:
        model, tokenizer = model_bundle()
        block_size = model.config.get("block_size", 128)
        prompt = format_prompt(request.message, request.history, block_size)
        device = next(model.parameters()).device
        ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        stop_seqs = [tokenizer.encode("\nUser:"), tokenizer.encode("\n\nUser:")]
        with torch.inference_mode():
            output = model.generate(
                ids,
                max_new_tokens=request.max_new_tokens or 64,
                temperature=0.25,
                top_k=2,
                repetition_penalty=1.15,
                stop_sequences=stop_seqs
            )[0].tolist()
        reply = tokenizer.decode(output[len(ids[0]):]).split("\nUser:", 1)[0].split("\n\nUser:", 1)[0].strip()
        if reply:
            return ChatResponse(reply=reply, source="generator", confidence=None)
    except Exception:
        # Keep the API usable if the checkpoint is unavailable or inference fails;
        # /health communicates model readiness separately.
        pass

    # For end users, provide a polite learning prompt instead of garbled character hallucinations
    fallback = (
        "I don't have verified data on that in my knowledge base yet, and online search did not find an encyclopedic match. "
        "You can teach me by saying: 'Remember that [fact]' or 'The answer is [explanation]', and I will permanently save it to my knowledge base!"
    )
    return ChatResponse(reply=fallback, source="needs_teaching", confidence=None)
