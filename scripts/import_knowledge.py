"""Ingest knowledge from external websites, text files, or markdown into TinyTalk RAG."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def extract_from_url(url: str) -> list[dict[str, str]]:
    """Fetch a web page and parse structured Q&A / knowledge blocks."""
    try:
        import requests
        from bs4 import BeautifulSoup
        resp = requests.get(url, timeout=10, headers={"User-Agent": "TinyTalk-Knowledge-Importer/1.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        page_title = soup.title.string.strip() if soup.title and soup.title.string else "External Resource"
        page_title = page_title.split("-")[0].split("|")[0].strip()

        records = []
        headings = soup.find_all(["h1", "h2", "h3"])
        for h in headings:
            heading_text = h.get_text(strip=True)
            if not heading_text or len(heading_text) < 3 or len(heading_text) > 80:
                continue
            # Collect following paragraphs until next heading
            paragraphs = []
            curr = h.next_sibling
            while curr and getattr(curr, "name", None) not in ["h1", "h2", "h3"]:
                if getattr(curr, "name", None) == "p":
                    p_text = curr.get_text(strip=True)
                    if len(p_text) > 30:
                        paragraphs.append(p_text)
                curr = curr.next_sibling

            if paragraphs:
                body = " ".join(paragraphs)[:350].strip()
                q = f"What is {heading_text}?" if not heading_text.endswith("?") else heading_text
                records.append({
                    "user": q,
                    "assistant": body,
                    "source": url
                })

        # If no heading blocks found, chunk paragraphs
        if not records:
            paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if len(p.get_text(strip=True)) > 50]
            for i, p in enumerate(paragraphs[:10], 1):
                records.append({
                    "user": f"Tell me about {page_title} part {i}",
                    "assistant": p[:300].strip(),
                    "source": url
                })
        return records

    except ImportError:
        # Fallback to standard library urllib
        import urllib.request
        from html.parser import HTMLParser

        class TextParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.text = []
                self.ignore = False

            def handle_starttag(self, tag, attrs):
                if tag in ("script", "style", "nav", "footer"):
                    self.ignore = True

            def handle_endtag(self, tag):
                if tag in ("script", "style", "nav", "footer"):
                    self.ignore = False

            def handle_data(self, data):
                if not self.ignore and data.strip():
                    self.text.append(data.strip())

        req = urllib.request.Request(url, headers={"User-Agent": "TinyTalk-Importer/1.0"})
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read().decode("utf-8", errors="ignore")
        parser = TextParser()
        parser.feed(html)
        full_text = " ".join(parser.text)
        sentences = re.split(r"(?<=[.!?]) +", full_text)
        chunks = [" ".join(sentences[i:i + 3]) for i in range(0, min(len(sentences), 15), 3)]
        return [{"user": f"Tell me about information from {url}", "assistant": chunk[:300], "source": url} for chunk in chunks]


def extract_from_file(file_path: Path) -> list[dict[str, str]]:
    """Parse local text, markdown, or jsonl files into knowledge records."""
    if file_path.suffix.lower() == ".jsonl":
        records = []
        for line in file_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records

    content = file_path.read_text(encoding="utf-8")
    lines = content.splitlines()
    records = []
    current_q = None
    current_a = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            if current_q and current_a:
                records.append({"user": current_q, "assistant": " ".join(current_a)[:350], "source": file_path.name})
                current_a = []
            heading = line.lstrip("#").strip()
            current_q = f"What is {heading}?" if not heading.endswith("?") else heading
        else:
            current_a.append(line)

    if current_q and current_a:
        records.append({"user": current_q, "assistant": " ".join(current_a)[:350], "source": file_path.name})

    if not records:
        # Fallback to paragraph chunking
        paragraphs = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 30]
        for idx, p in enumerate(paragraphs, 1):
            records.append({
                "user": f"What is topic section {idx} in {file_path.name}?",
                "assistant": p[:350].strip(),
                "source": file_path.name
            })
    return records


def main():
    parser = argparse.ArgumentParser(description="Import external text or website knowledge into TinyTalk RAG.")
    parser.add_argument("--url", type=str, help="Webpage URL to scrape and index")
    parser.add_argument("--file", type=str, help="Local text or markdown file to ingest")
    parser.add_argument("--output", type=str, default="data/knowledge.jsonl", help="Target JSONL file")
    args = parser.parse_args()

    if not args.url and not args.file:
        parser.error("Specify at least one knowledge source via --url or --file")

    records = []
    if args.url:
        print(f"Fetching and parsing URL: {args.url} ...")
        records.extend(extract_from_url(args.url))

    if args.file:
        fpath = Path(args.file)
        if not fpath.exists():
            raise FileNotFoundError(f"File not found: {fpath}")
        print(f"Parsing local file: {fpath} ...")
        records.extend(extract_from_file(fpath))

    out_path = ROOT / args.output
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Successfully imported {len(records)} knowledge records to {out_path}!")
    print("TinyTalk RAG index will now automatically include these records.")


if __name__ == "__main__":
    main()

