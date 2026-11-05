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
            safe_topic = urllib.parse.quote(topic_normalized, safe="()_")
            summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{safe_topic}"
            req = urllib.request.Request(summary_url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=3.5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                extract = data.get("extract", "").strip()
                if extract and data.get("type") != "disambiguation" and "may refer to" not in extract.lower():
                    title = data.get("title", topic)
                    page_url = data.get("content_urls", {}).get("desktop", {}).get("page")
                    if len(extract) > 20:
                        return title, extract, page_url or f"https://en.wikipedia.org/wiki/{safe_topic}"
        except Exception:
            pass

    # Step 2: Wikipedia Search API in strict relevance order (Index 0 first)
    try:
        search_query = topic if topic and len(topic) > 2 else query
        url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode({
            "action": "query",
            "list": "search",
            "srsearch": search_query,
            "format": "json",
            "srlimit": 3
        })
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=4.0) as resp:
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
                    with urllib.request.urlopen(req2, timeout=3.5) as resp2:
                        sdata = json.loads(resp2.read().decode("utf-8"))
                        ext = sdata.get("extract", "").strip()
                        if sdata.get("type") != "disambiguation" and "may refer to" not in ext.lower() and len(ext) > 25:
                            page_url = sdata.get("content_urls", {}).get("desktop", {}).get("page")
                            return best_title, ext, page_url or f"https://en.wikipedia.org/wiki/{safe_title}"
                except Exception:
                    continue
    except Exception:
        pass

    return None, None, None
