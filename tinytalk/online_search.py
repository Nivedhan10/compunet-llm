"""Dynamic multi-source internet knowledge retriever for TinyTalk.

Connects to open web resources and official developer platforms including:
- Wikipedia (encyclopedic definitions, concepts, animals, geography, science, history)
- Stack Overflow (programming code, errors, exceptions, terminal commands)
- Hugging Face (AI models, neural architectures, datasets, model cards)
- Open Web Search / DuckDuckGo Lite (practical how-tos, recipes, and broad inquiries)
"""
from __future__ import annotations

import gzip
import json
import re
import urllib.parse
import urllib.request

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

TECH_ENTITY_MAP = {
    "java": "Java (programming language)",
    "java language": "Java (programming language)",
    "c": "C (programming language)",
    "c++": "C++",
    "cpp": "C++",
    "c#": "C Sharp (programming language)",
    "csharp": "C Sharp (programming language)",
    "golang": "Go (programming language)",
    "go language": "Go (programming language)",
    "django": "Django (web framework)",
    "django framework": "Django (web framework)",
    "python": "Python (programming language)",
    "python language": "Python (programming language)",
    "react": "React (software)",
    "react js": "React (software)",
    "reactjs": "React (software)",
    "fastapi": "FastAPI",
    "pytorch": "PyTorch",
    "docker": "Docker (software)",
    "kubernetes": "Kubernetes",
    "rust": "Rust (programming language)",
    "ruby": "Ruby (programming language)",
    "php": "PHP",
    "swift": "Swift (programming language)",
    "kotlin": "Kotlin (programming language)",
    "git": "Git",
    "linux": "Linux",
    "sqlite": "SQLite",
    "postgresql": "PostgreSQL",
    "postgres": "PostgreSQL",
    "mongodb": "MongoDB",
    "redis": "Redis",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "vue": "Vue.js",
    "angular": "Angular (web framework)",
    "flask": "Flask (web framework)",
    "sql": "SQL",
    "spring": "Spring Framework",
    "spring boot": "Spring Framework"
}


def extract_primary_clause(query: str) -> str:
    """Extract the primary informative question clause when multiple sentences or questions are provided."""
    clauses = [c.strip() for c in re.split(r"[?!;]+", query) if c.strip()]
    if not clauses:
        return query
    for c in clauses:
        words = re.findall(r"\b[a-zA-Z0-9]+\b", c)
        if len(words) >= 2:
            return c
    return clauses[0]


def resolve_tech_entity(topic: str) -> str:
    """Resolve topic to an exact unambiguous technical title if it contains known programming entities."""
    topic_key = topic.lower().strip()
    if topic_key in TECH_ENTITY_MAP:
        return TECH_ENTITY_MAP[topic_key]
    words = re.findall(r"\b[a-zA-Z0-9+#.]+\b", topic_key)
    for w in words:
        if w in TECH_ENTITY_MAP:
            return TECH_ENTITY_MAP[w]
    return topic


def clean_topic_query(query: str) -> str:
    """Strip conversational prefixes, multi-clause noise, and trailing fluff to isolate the core entity."""
    primary = extract_primary_clause(query)
    cleaned = primary.strip("?.! \t\r\n")

    patterns = [
        r"^(?:can|could|would)\s+you\s+(?:please\s+)?(?:tell me|explain|describe|give me|show me)\s+(?:about\s+|what is\s+|who is\s+|how\s+)?",
        r"^(?:please\s+)?(?:tell me|explain|describe|give me|show me)\s+(?:about\s+|what is\s+|who is\s+|how\s+)?",
        r"^(?:do you know\s+)(?:what is\s+|who is\s+|about\s+)?",
        r"^(?:the\s+)?(?:start\s+|run\s+|build\s+)?(?:command|commands|cli|script)\s+(?:for|to|of)\s+",
        r"^(?:what is the\s+|what is an\s+|what is a\s+|what is\s+|what are\s+|what was\s+|what were\s+)",
        r"^(?:who is\s+|who was\s+|who were\s+|who are\s+)",
        r"^(?:where is\s+|where are\s+|where was\s+)",
        r"^(?:how does\s+|how do\s+|how to\s+|how\s+)",
        r"^(?:make|cook|prepare|bake|brew)\s+",
        r"^(?:the\s+|a\s+|an\s+)",
    ]

    changed = True
    while changed:
        changed = False
        for pat in patterns:
            m = re.match(pat, cleaned, re.IGNORECASE)
            if m:
                cleaned = cleaned[m.end():].strip("?.! \t\r\n")
                changed = True

    trailing_fluff = [
        r"\s+(?:generally\s+)?in\s+terminal\??$",
        r"\s+generally\??$",
        r"\s+please\??$",
        r"\s+in\s+detail\??$",
        r"\s+briefly\??$",
        r"\s+works\??$",
        r"\s+steps\??$",
        r"\s+proeprly\??$",
        r"\s+properly\??$"
    ]
    for pat in trailing_fluff:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE).strip("?.! ")

    return cleaned


def search_wikipedia(query: str, topic: str) -> tuple[str | None, str | None, str | None]:
    """Search Wikipedia REST summary and search APIs for encyclopedic articles."""
    lookup_title = resolve_tech_entity(topic)
    topic_normalized = re.sub(r"[^\w\s-]", "", lookup_title).strip().replace(" ", "_")

    # Step 1: Direct summary lookup for clean unambiguous topics
    if topic_normalized and len(topic_normalized) > 1:
        try:
            safe_topic = urllib.parse.quote(lookup_title.replace(" ", "_"), safe="()_")
            summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{safe_topic}"
            req = urllib.request.Request(summary_url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=3.5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                extract = data.get("extract", "").strip()
                if extract and data.get("type") != "disambiguation" and "may refer to" not in extract.lower():
                    title = data.get("title", topic)
                    page_url = data.get("content_urls", {}).get("desktop", {}).get("page")
                    if len(extract) > 25:
                        return title, extract, page_url or f"https://en.wikipedia.org/wiki/{safe_topic}"
        except Exception:
            pass

    # Step 2: Wikipedia Search API in strict relevance order
    try:
        search_query = topic if topic and len(topic) > 1 else query
        url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode({
            "action": "query",
            "list": "search",
            "srsearch": search_query,
            "format": "json",
            "srlimit": 4
        })
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=3.5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            results = data.get("query", {}).get("search", [])
            for res in results:
                best_title = res.get("title")
                if not best_title:
                    continue
                safe_title = urllib.parse.quote(best_title.replace(" ", "_"), safe="()_")
                surl = f"https://en.wikipedia.org/api/rest_v1/page/summary/{safe_title}"
                req2 = urllib.request.Request(surl, headers=HEADERS)
                try:
                    with urllib.request.urlopen(req2, timeout=3.0) as resp2:
                        sdata = json.loads(resp2.read().decode("utf-8"))
                        ext = sdata.get("extract", "").strip()
                        if sdata.get("type") != "disambiguation" and "may refer to" not in ext.lower() and len(ext) > 30:
                            page_url = sdata.get("content_urls", {}).get("desktop", {}).get("page")
                            return best_title, ext, page_url or f"https://en.wikipedia.org/wiki/{safe_title}"
                except Exception:
                    continue
    except Exception:
        pass

    return None, None, None


def search_stack_overflow(query: str) -> tuple[str | None, str | None, str | None]:
    """Search Stack Overflow for developer questions, error fixes, and code solutions."""
    clean_q = re.sub(r"[^\w\s-]", " ", query).strip()
    words = clean_q.split()
    search_q = " ".join(words[:6]) if len(words) > 6 else clean_q

    url = "https://api.stackexchange.com/2.3/search/advanced?" + urllib.parse.urlencode({
        "order": "desc",
        "sort": "relevance",
        "q": search_q,
        "site": "stackoverflow",
        "filter": "withbody"
    })
    req = urllib.request.Request(url, headers={**HEADERS, "Accept-Encoding": "gzip"})
    try:
        with urllib.request.urlopen(req, timeout=4.0) as resp:
            raw = resp.read()
            if resp.info().get("Content-Encoding") == "gzip":
                raw = gzip.decompress(raw)
            data = json.loads(raw.decode("utf-8"))
            items = data.get("items", [])
            for it in items[:2]:
                title = it.get("title", "")
                link = it.get("link", "")
                body = it.get("body", "")
                clean_body = re.sub(r"<pre><code>(.*?)</code></pre>", r"\n```\n\1\n```\n", body, flags=re.DOTALL)
                clean_body = re.sub(r"<[^>]+>", " ", clean_body).strip()
                clean_body = re.sub(r"\s+", " ", clean_body)
                if len(clean_body) > 40:
                    summary = f"**{title}**\n\n{clean_body[:400]}..."
                    return title, summary, link
    except Exception:
        pass
    return None, None, None


def search_hugging_face(query: str) -> tuple[str | None, str | None, str | None]:
    """Search Hugging Face Hub for open-source AI models and checkpoints."""
    clean_q = re.sub(r"[^\w\s-]", "", query).strip()
    words = [w for w in clean_q.split() if w.lower() not in {"model", "models", "huggingface", "hf", "hub", "on"}]
    search_term = " ".join(words) if words else clean_q

    url = f"https://huggingface.co/api/models?search={urllib.parse.quote(search_term)}&limit=3"
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=3.5) as resp:
            models = json.loads(resp.read().decode("utf-8"))
            if models and isinstance(models, list):
                top_m = models[0]
                model_id = top_m.get("id")
                pipeline_tag = top_m.get("pipeline_tag", "AI Model")
                likes = top_m.get("likes", 0)
                downloads = top_m.get("downloads", 0)
                summary = (
                    f"**{model_id}** is an open-source model hosted on Hugging Face Hub.\n"
                    f"- Pipeline / Task: `{pipeline_tag}`\n"
                    f"- Community Likes: {likes:,} | Downloads: {downloads:,}\n"
                    f"- Other matching models: {', '.join(m.get('id', '') for m in models[1:3])}"
                )
                return model_id, summary, f"https://huggingface.co/{model_id}"
    except Exception:
        pass
    return None, None, None


def search_web_ddg(query: str) -> tuple[str | None, str | None, str | None]:
    """Search DuckDuckGo Lite for practical how-tos, recipes, and broad web answers."""
    data = urllib.parse.urlencode({"q": query}).encode("utf-8")
    req = urllib.request.Request("https://lite.duckduckgo.com/lite/", data=data, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=4.5) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
            links = re.findall(r"<a[^>]*rel=[\"']nofollow[\"'][^>]*href=[\"']([^\"']*)[\"'][^>]*>(.*?)</a>", html)
            snippets = re.findall(r"class=[\"']result-snippet[\"']>(.*?)</td>", html, re.DOTALL)
            if snippets:
                valid_snippets = []
                for s in snippets[:3]:
                    c = re.sub(r"<[^>]+>", "", s).strip()
                    c = re.sub(r"\s+", " ", c)
                    if len(c) > 35:
                        valid_snippets.append(c)
                if valid_snippets:
                    top_url = links[0][0] if links else "https://duckduckgo.com"
                    top_title = re.sub(r"<[^>]+>", "", links[0][1]).strip() if links else query
                    combined = "\n\n".join(valid_snippets)
                    return top_title, combined, top_url
    except Exception:
        pass
    return None, None, None


def fetch_online_knowledge(query: str) -> tuple[str | None, str | None, str | None]:
    """Search live internet sources (Wikipedia, Stack Overflow, Hugging Face, Open Web) dynamically.
    
    Returns:
        tuple of (title, summary_extract, source_url) or (None, None, None).
    """
    topic = clean_topic_query(query)
    q_lower = query.lower()

    # Strategy 1: AI Model / Transformers / Hugging Face Queries
    if any(k in q_lower for k in ["hugging face", "huggingface", "hf model", "pretrained model", "bert model", "gpt model", "llama model"]):
        title, extract, url = search_hugging_face(topic or query)
        if extract:
            return title, extract, url

    # Strategy 2: Code error, exception, or runtime bug -> Stack Overflow first
    if any(k in q_lower for k in ["error", "exception", "traceback", "syntaxerror", "indexerror", "keyerror", "how to catch", "failed to", "bug", "stackoverflow"]):
        title, extract, url = search_stack_overflow(query)
        if extract:
            return title, extract, url

    # Strategy 3: How-to guides, recipes, and practical step-by-step questions -> Web search + Wikipedia
    if any(k in q_lower for k in ["how to make", "how to cook", "how to prepare", "how to bake", "how to brew", "recipe", "steps properly", "step by step"]):
        # Try open web search first for rich procedural how-to guides (e.g. wikiHow, cooking blogs)
        title, extract, url = search_web_ddg(query)
        if extract and len(extract) > 50:
            return title, extract, url
        # Fall back to Wikipedia
        title, extract, url = search_wikipedia(query, topic)
        if extract:
            return title, extract, url

    # Strategy 4: General encyclopedic concepts, animals, geography, science, definitions -> Wikipedia
    title, extract, url = search_wikipedia(query, topic)
    if extract:
        return title, extract, url

    # Strategy 5: Open Web Search (DuckDuckGo Lite) as universal fallback for any question
    title, extract, url = search_web_ddg(query)
    if extract:
        return title, extract, url

    # Strategy 6: Stack Overflow fallback for general programming terms
    if any(k in q_lower for k in ["python", "javascript", "java", "code", "function", "class", "import", "package", "api", "database", "sql"]):
        title, extract, url = search_stack_overflow(query)
        if extract:
            return title, extract, url

    return None, None, None
