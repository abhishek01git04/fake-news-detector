"""Extract the main article text from a news URL."""

from __future__ import annotations

import httpx
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
}
MAX_CHARS = 6000  # keep LLM cost bounded


def fetch_article(url: str, timeout: float = 15.0) -> tuple[str, str]:
    """Return (title, body_text) for a news article URL."""
    if not url.startswith(("http://", "https://")):
        raise ValueError("URL must start with http:// or https://")
    resp = httpx.get(url, headers=HEADERS, timeout=timeout, follow_redirects=True)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form", "noscript", "figure"]):
        tag.decompose()

    title = (soup.title.string or "").strip() if soup.title else ""
    container = soup.find("article") or soup.find("main") or soup.body or soup
    paragraphs = [p.get_text(" ", strip=True) for p in container.find_all("p")]
    paragraphs = [p for p in paragraphs if len(p) > 40]
    body = "\n".join(paragraphs).strip()
    if not body:
        raise ValueError("Could not extract article text from this page.")
    return title, body[:MAX_CHARS]
