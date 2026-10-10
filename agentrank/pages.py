"""Fetch pages the engines read and cache their main text as markdown (free, no model calls)."""
import requests
import trafilatura

from agentrank.store import cached_call

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0 Safari/537.36"


def _fetch(url: str) -> tuple[dict, float]:
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": UA})
        html = r.text if r.ok and "html" in r.headers.get("content-type", "") else ""
    except requests.RequestException:
        html = ""
    md = trafilatura.extract(html, output_format="markdown", include_tables=True, include_links=False) if html else None
    return {"url": url, "ok": bool(md), "markdown": md or "", "html_len": len(html)}, 0.0


def fetch(url: str) -> dict:
    return cached_call("page", {"url": url}, lambda: _fetch(url), est_cost=0.0)
