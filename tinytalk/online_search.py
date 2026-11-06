"""Dynamic multi-source internet knowledge retriever for TinyTalk.

Connects to open web resources and official developer platforms including:
- Wikipedia (encyclopedic definitions, concepts, animals, geography, science, history)
- Stack Overflow (programming code, errors, exceptions, terminal commands)
- Hugging Face (AI models, neural architectures, datasets, model cards)
- Open Web Search / DuckDuckGo Lite (practical how-tos, recipes, and broad inquiries)
"""
from __future__ import annotations

import gzip
import html
import json
import re
import urllib.parse
import urllib.request

# Search may use DuckDuckGo to discover results, but only these reference sites
# are accepted as answer sources. Add domains by configuration/policy, not by
# adding person-specific facts or names to application code.
ALLOWED_REFERENCE_DOMAINS = (
    "wikipedia.org", "stackoverflow.com", "stackexchange.com", "huggingface.co",
    "python.org", "pytorch.org", "fastapi.tiangolo.com", "react.dev", "developer.mozilla.org",
)


def is_allowed_reference_url(url: str) -> bool:
    """Reject non-HTTPS and off-allowlist result URLs before citing them."""
    try:
        parsed = urllib.parse.urlparse(url)
        host = (parsed.hostname or "").lower().rstrip(".")
        return parsed.scheme == "https" and any(
            host == domain or host.endswith("." + domain)
            for domain in ALLOWED_REFERENCE_DOMAINS
        )
    except ValueError:
        return False

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
        r"^(?:the\s+|a\s+|an\s+)(?![A-Z])",
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
            page_html = resp.read().decode("utf-8", errors="ignore")
            links = re.findall(r"<a[^>]*rel=[\"']nofollow[\"'][^>]*href=[\"']([^\"']*)[\"'][^>]*>(.*?)</a>", page_html)
            snippets = re.findall(r"class=[\"']result-snippet[\"']>(.*?)</td>", page_html, re.DOTALL)
            if snippets and links:
                for index, (href, raw_title) in enumerate(links[:len(snippets)]):
                    parsed_link = urllib.parse.urlparse(html.unescape(href))
                    target_url = urllib.parse.parse_qs(parsed_link.query).get("uddg", [href])[0]
                    target_url = urllib.parse.unquote(target_url)
                    if not is_allowed_reference_url(target_url):
                        continue
                    s = snippets[index]
                    c = re.sub(r"<[^>]+>", "", s).strip()
                    c = re.sub(r"\s+", " ", c)
                    if len(c) > 35:
                        top_title = html.unescape(re.sub(r"<[^>]+>", "", raw_title).strip())
                        return top_title, html.unescape(c), target_url
    except Exception:
        pass
    return None, None, None


STOP_STARTERS = {
    "who", "what", "where", "when", "why", "how", "did", "is", "was", "are",
    "were", "has", "have", "can", "could", "will", "would", "does", "do",
    "tell", "me", "about", "the", "a", "an", "i", "mean", "meant", "please"
}

PREDICATE_NOISE = {
    "retired", "retire", "retirement", "from", "alive", "dead", "die", "died",
    "today", "now", "still", "playing", "active", "born", "live", "lives"
}
CAPITALIZED_QUESTION_STARTERS = {
    "who", "what", "where", "when", "why", "how", "did", "is", "was", "are",
    "were", "has", "have", "can", "could", "will", "would", "does", "do",
}


def extract_general_subject(query: str) -> str:
    """Grammatically extract the subject entity from a query using general NLP with ZERO hardcoded names."""
    # 1. Check for clarification / correction: "I mean [the] X", "I meant X", "talking about X"
    clarif = re.search(r"\b(?:i mean|i meant|talking about|referring to)\s+(?:the\s+)?(.+?)(?:,|\?|$)", query, re.I)
    if clarif:
        return clarif.group(1).strip("?.! \t\r\n")

    # 2. Extract alphanumeric tokens
    words = re.findall(r"\b[a-zA-Z0-9'-]+\b", query)
    if not words:
        return query.strip("?.! ")

    # 3. Look for capitalized multi-word sequences (proper nouns)
    caps_matches = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", query)
    valid_caps = []
    for match in caps_matches:
        parts = match.split()
        # Keep capitalized articles such as "The Rock" and "The Undertaker";
        # they can be part of an entity's actual name.
        while parts and parts[0].lower() in CAPITALIZED_QUESTION_STARTERS:
            parts.pop(0)
        if parts:
            valid_caps.append(" ".join(parts))
    if valid_caps:
        return max(valid_caps, key=len)

    # 4. Strip leading question starters and predicate noise to isolate the subject
    content_words = [w for w in words if w.lower() not in STOP_STARTERS and w.lower() not in PREDICATE_NOISE]
    if content_words:
        return " ".join(content_words)

    return query.strip("?.! ")


def detect_intent_and_entity(query: str, history: list[dict] | None = None) -> tuple[str, str, dict]:
    """Resolve active entity and question intent using general conversation coreference with ZERO hardcoded names."""
    q_clean = query.strip()
    history = history or []

    # 1. Check for clarification / correction
    is_clarification = bool(re.search(r"\b(?:i mean|i meant|talking about|referring to)\b", q_clean, re.I))
    current_subject = extract_general_subject(q_clean)

    # 2. Extract most recent substantive subject from history
    prev_subject = None
    for turn in reversed(history):
        # Assistant output is generated/retrieved text and may mention many entities.
        # Only user turns are evidence for what a follow-up pronoun refers to.
        role = turn.get("role", "user") if isinstance(turn, dict) else getattr(turn, "role", "user")
        if role != "user":
            continue
        content = turn.get("content", "") if isinstance(turn, dict) else getattr(turn, "content", "")
        cand = extract_general_subject(content)
        if cand and len(cand) > 2 and cand.lower() not in STOP_STARTERS:
            prev_subject = cand
            break

    # 3. Coreference resolution: pronouns (he, she, they, it) and sub-tokens
    target_entity = current_subject
    q_words = set(re.findall(r"\b[a-z0-9]+\b", q_clean.lower()))

    # If user used a pronoun ("did he retire?"), inherit the previous entity
    if q_words & {"he", "she", "they", "it", "him", "her"} and prev_subject:
        target_entity = prev_subject
    # If the user used a single partial word of the previous entity ("did ronaldo retire?", "did kohli retire?"), inherit the full entity
    elif prev_subject and current_subject:
        cs_tokens = set(current_subject.lower().split())
        ps_tokens = set(prev_subject.lower().split())
        if cs_tokens.issubset(ps_tokens) and len(cs_tokens) < len(ps_tokens):
            target_entity = prev_subject

    # 4. Detect Question Intent
    q_lower = q_clean.lower()
    intent = "general"
    if re.search(r"\b(?:retired?|retire|still playing|still active)\b", q_lower):
        intent = "retirement_status"
    elif re.search(r"\b(?:alive|dead|die|died|passed away)\b", q_lower):
        intent = "life_status"
    elif re.search(r"\b(?:free|open source|paid|proprietary)\b", q_lower):
        intent = "open_source_status"
    elif re.search(r"^(?:did|is|was|are|were|has|have|can|does|do)\b", q_lower):
        intent = "verification_yes_no"

    return target_entity or clean_topic_query(query), intent, {
        "is_clarification": is_clarification,
        "raw_query": q_clean,
        "resolved_entity": target_entity
    }


def synthesize_wise_answer(entity: str, raw_extract: str, intent: str, meta: dict, source_url: str) -> str:
    """Present retrieved evidence without inferring facts the source did not state.

    TinyTalk has no second language model, so keyword-based "yes/no" conclusions
    are intentionally avoided. The linked source is the authority for interpretation.
    """
    is_clarification = meta.get("is_clarification", False)
    prefix = f"For **{entity}**: " if is_clarification else ""
    return f"{prefix}{raw_extract}\n\n[Source: {source_url}]"


def fetch_online_knowledge(query: str, history: list[dict] | None = None) -> tuple[str | None, str | None, str | None]:
    """Search live internet sources (Wikipedia, Stack Overflow, Hugging Face, Open Web) dynamically and synthesize intelligent answers.
    
    Returns:
        tuple of (title, synthesized_answer, source_url) or (None, None, None).
    """
    entity, intent, meta = detect_intent_and_entity(query, history)
    topic = clean_topic_query(entity) if entity else clean_topic_query(query)
    # A pronoun-only follow-up must include its resolved subject in the search.
    search_query = query
    if entity and entity.lower() not in query.lower():
        search_query = f"{entity} {query}"
    q_lower = search_query.lower()
    # A biography summary is insufficient for questions about changing facts.
    # Search the full question so the source lookup sees the requested predicate.
    needs_fresh_result = intent in {"retirement_status", "life_status", "verification_yes_no"} or bool(
        re.search(r"\b(current|currently|latest|today|now|playing|team|club|roster|score|status)\b", q_lower)
    )
    search_topic = search_query if needs_fresh_result else topic

    # Strategy 1: AI Model / Transformers / Hugging Face Queries
    if any(k in q_lower for k in ["hugging face", "huggingface", "hf model", "pretrained model", "bert model", "gpt model", "llama model"]):
        title, extract, url = search_hugging_face(topic or query)
        if extract:
            return title, extract, url

    # Strategy 2: Code error, exception, or runtime bug -> Stack Overflow first
    if any(k in q_lower for k in ["error", "exception", "traceback", "syntaxerror", "indexerror", "keyerror", "how to catch", "failed to", "bug", "stackoverflow"]):
        title, extract, url = search_stack_overflow(search_query)
        if extract:
            return title, extract, url

    # Strategy 3: How-to guides, recipes, and practical step-by-step questions -> Web search + Wikipedia
    if any(k in q_lower for k in ["how to make", "how to cook", "how to prepare", "how to bake", "how to brew", "recipe", "steps properly", "step by step"]):
        title, extract, url = search_web_ddg(search_query)
        if extract and len(extract) > 50:
            return title, extract, url
        title, extract, url = search_wikipedia(search_query, search_topic)
        if extract:
            return title, extract, url

    # Prefer a result for the complete question when asking about changing facts;
    # the accepted result URL must pass the reference-domain allowlist.
    if needs_fresh_result:
        title, extract, url = search_web_ddg(search_query)
        if extract:
            return title, synthesize_wise_answer(topic or title, extract, intent, meta, url), url

    # Strategy 4: General encyclopedic concepts, persons, animals, geography, science -> Wikipedia
    title, extract, url = search_wikipedia(search_query, search_topic)
    if extract:
        wise_ans = synthesize_wise_answer(topic or title, extract, intent, meta, url)
        return title, wise_ans, url

    # Strategy 5: Open Web Search (DuckDuckGo Lite) as universal fallback
    title, extract, url = search_web_ddg(search_query)
    if extract:
        wise_ans = synthesize_wise_answer(topic or title, extract, intent, meta, url)
        return title, wise_ans, url

    # Strategy 6: Stack Overflow fallback for general programming terms
    if any(k in q_lower for k in ["python", "javascript", "java", "code", "function", "class", "import", "package", "api", "database", "sql"]):
        title, extract, url = search_stack_overflow(search_query)
        if extract:
            return title, extract, url

    return None, None, None
