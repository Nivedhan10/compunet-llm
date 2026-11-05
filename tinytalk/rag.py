"""Zero-dependency Retrieval-Augmented Generation (RAG) engine for TinyTalk.

Provides BM25 content-word indexing with typo-tolerant fuzzy matching over curated
dialogues and imported external knowledge documents.
"""
from __future__ import annotations

import difflib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from collections import Counter


STOPWORDS = {
    "a", "about", "above", "after", "again", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between",
    "both", "but", "by", "can", "cannot", "could", "did", "do", "does", "doing",
    "down", "during", "each", "few", "for", "from", "further", "had", "has", "have",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how",
    "i", "if", "in", "into", "is", "it", "its", "itself", "me", "more", "most",
    "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs",
    "them", "themselves", "then", "there", "these", "they", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "we", "were",
    "what", "when", "where", "which", "while", "who", "whom", "why", "with", "would",
    "you", "your", "yours", "yourself", "yourselves", "tell"
}


@dataclass
class RAGMatch:
    query: str
    matched_prompt: str
    answer: str
    score: float
    confidence: float
    source: str


def clean_words(text: str) -> list[str]:
    """Extract lowercased alphanumeric words, normalizing hyphens."""
    return re.findall(r"\b[a-z0-9]+\b", text.lower().replace("-", " "))


class TinyRAG:
    """Fast, zero-dependency BM25 retriever with typo tolerance and dynamic learning."""

    def __init__(self, data_files: list[Path | str] | None = None, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: list[dict[str, str]] = []
        self.doc_tokens: list[list[str]] = []
        self.doc_lens: list[int] = []
        self.vocab: set[str] = set()
        self.df: Counter[str] = Counter()
        self.idf: dict[str, float] = {}
        self.avg_dl: float = 0.0

        if data_files is None:
            root = Path(__file__).resolve().parents[1]
            data_files = [
                root / "data" / "dialogues.jsonl",
                root / "data" / "knowledge.jsonl",
            ]
        self.data_files = [Path(f) for f in data_files]
        self.reload()

    def reload(self) -> int:
        """Reload and index all existing knowledge files."""
        self.documents = []
        for path in self.data_files:
            if not path.exists():
                continue
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        user = record.get("user", "").strip()
                        assistant = record.get("assistant", "").strip()
                        source = record.get("source", path.name)
                        if user and assistant:
                            self.documents.append({"user": user, "assistant": assistant, "source": source})
                    except json.JSONDecodeError:
                        continue

        self.doc_tokens = []
        self.doc_lens = []
        self.vocab = set()
        self.df = Counter()

        n = len(self.documents)
        if n == 0:
            return 0

        for doc in self.documents:
            # Index user question intent heavily with assistant concepts
            user_words = [t for t in clean_words(doc["user"]) if len(t) > 1]
            assistant_words = [t for t in clean_words(doc["assistant"]) if t not in STOPWORDS and len(t) > 1]
            content = user_words * 3 + assistant_words
            self.doc_tokens.append(content)
            self.doc_lens.append(len(content))
            self.vocab.update(content)
            for t in set(content):
                self.df[t] += 1

        self.avg_dl = sum(self.doc_lens) / max(1, n)
        self.idf = {
            t: math.log((n - freq + 0.5) / (freq + 0.5) + 1.0) for t, freq in self.df.items()
        }
        return n

    def add_and_save_knowledge(self, user_q: str, assistant_ans: str, source: str = "auto_learned") -> None:
        """Append a new knowledge record to data/knowledge.jsonl and reload indices."""
        target = self.data_files[1]  # data/knowledge.jsonl
        target.parent.mkdir(exist_ok=True)
        record = {
            "user": user_q.strip(),
            "assistant": assistant_ans.strip(),
            "source": source
        }
        with open(target, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.reload()

    def _resolve_word(self, word: str) -> str:
        """Resolve a query word against the vocabulary using exact or high-precision fuzzy match."""
        if word in self.vocab:
            return word
        # Only fuzzy match words with at least 6 letters and >=0.90 similarity to avoid false positive collisions
        if len(word) >= 6:
            matches = difflib.get_close_matches(word, self.vocab, n=1, cutoff=0.90)
            if matches:
                return matches[0]
        return word

    def retrieve(self, query: str, top_k: int = 1, threshold: float = 2.2) -> list[RAGMatch]:
        """Retrieve the best matching knowledge records.
        
        Args:
            query: The user prompt.
            top_k: Number of candidate matches to return.
            threshold: Minimum BM25 score required for high-confidence grounding.
        """
        if not self.documents:
            return []

        raw_words = clean_words(query)
        q_tokens = [self._resolve_word(w) for w in raw_words if w not in STOPWORDS and len(w) > 1]
        # If all words were stopwords (e.g. 'who are you', 'what can you do'), fall back to raw words
        if not q_tokens:
            q_tokens = [self._resolve_word(w) for w in raw_words if len(w) > 1]
        if not q_tokens:
            return []

        q_counts = Counter(q_tokens)
        scored = []

        for i, doc in enumerate(self.documents):
            tokens = self.doc_tokens[i]
            dl = self.doc_lens[i]
            counts = Counter(tokens)
            score = 0.0

            for t in q_counts:
                if t in counts:
                    tf = counts[t]
                    idf = self.idf.get(t, 0.5)
                    denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / max(1.0, self.avg_dl)))
                    score += idf * (tf * (self.k1 + 1.0)) / denom

            if score >= threshold:
                confidence = min(1.0, max(0.5, score / 8.0))
                scored.append((score, confidence, doc))

        scored.sort(key=lambda x: x[0], reverse=True)

        results: list[RAGMatch] = []
        for score, conf, doc in scored[:top_k]:
            results.append(
                RAGMatch(
                    query=query,
                    matched_prompt=doc["user"],
                    answer=doc["assistant"],
                    score=score,
                    confidence=conf,
                    source=doc.get("source", "knowledge_base"),
                )
            )

        return results
