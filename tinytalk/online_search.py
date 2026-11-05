"""Online knowledge retriever for TinyTalk.

Fetches factual encyclopedia summaries with source references via Wikipedia APIs
when a query is not present in the local dataset.
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
}


def clean_topic_query(query: str) -> str:
    """Strip conversational prefixes to isolate the core entity or question topic."""
    cleaned = query.strip("?.! ").strip()
    prefixes = [
        "what is the ", "what is a ", "what is an ", "what is ", "what are ",
        "who is ", "who was ", "who were ", "who are ",
        "tell me about ", "explain ", "where is ", "how does ", "can you explain "
    ]
    for prefix in prefixes:
        if cleaned.lower().startswith(prefix):
            return cleaned[len(prefix):].strip()
    return cleaned


def fetch_online_knowledge(query: str) -> tuple[str | None, str | None, str | None]:
    """Search for factual knowledge online and return (title, extract, source_url).
    
    Returns:
        tuple of (title, summary_extract, source_url) or (None, None, None).
    """
    topic = clean_topic_query(query)
    topic_normalized = re.sub(r"[^\w\s-]", "", topic).strip().replace(" ", "_")

    # Step 1: Direct summary lookup for clean topics (fastest, high quality)
    if topic_normalized and len(topic_normalized) > 2:
        try:
            summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(topic_normalized)}"
            req = urllib.request.Request(summary_url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=3.5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if data.get("extract") and data.get("type") != "disambiguation":
                    title = data.get("title", topic)
                    extract = data.get("extract", "").strip()
                    page_url = data.get("content_urls", {}).get("desktop", {}).get("page")
                    if extract and len(extract) > 20:
                        return title, extract, page_url or f"https://en.wikipedia.org/wiki/{topic_normalized}"
        except Exception:
            pass

    # Step 2: Wikipedia Search API generator fallback
    try:
        search_query = topic if topic and len(topic) > 2 else query
        url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode({
            "action": "query",
            "generator": "search",
            "gsrsearch": search_query,
            "gsrlimit": "1",
            "prop": "extracts",
            "exintro": "1",
            "explaintext": "1",
            "exsentences": "3",
            "format": "json"
        })
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=4.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            pages = data.get("query", {}).get("pages", {})
            for pid, p in pages.items():
                title = p.get("title")
                extract = p.get("extract", "").strip()
                if extract and len(extract) > 20:
                    wiki_url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
                    return title, extract, wiki_url
    except Exception:
        pass

    return None, None, None

