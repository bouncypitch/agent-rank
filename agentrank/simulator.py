"""Cheap stand-in for an answer engine's "choose among sources" step.

Given a question and a pool of candidate pages (the ones a real engine read),
a small model writes the answer from those pages only; we read off which brands
it recommends. Swapping one page for an edited variant measures that edit.
"""
import os
import random

from agentrank import pages
from agentrank.engines.claude_engine import client, cost_of
from agentrank.mentions import ranked_brands
from agentrank.store import cached_call

MODEL = os.getenv("SIM_MODEL", "claude-haiku-5-5")
PAGE_CHARS = int(os.getenv("SIM_PAGE_CHARS", "2500"))
POOL_SIZE = int(os.getenv("SIM_POOL_SIZE", "12"))
# v1 said "answer using only these results" and over-trusted vendor listicles (r=-0.05 vs Perplexity).
# v2 mirrors how answer engines actually behave: search results plus their own knowledge.
SIM_VERSION = os.getenv("SIM_VERSION", "v2")
SYSTEMS = {
    "v1": ("You are an AI assistant answering a buyer's question. You searched the web and got the results below. "
           "Answer using only these results, recommend specific products, and cite results by number."),
    "v2": ("You are an AI assistant answering a buyer's question. You searched the web and got the results below. "
           "Combine these results with your own knowledge to recommend specific products, and cite results by number."),
}
SYSTEM = SYSTEMS[SIM_VERSION]


def build_pool(urls: list[str], swap: dict[str, str] | None = None) -> list[tuple[str, str]]:
    """[(url, text)] for the first POOL_SIZE fetchable urls; `swap` replaces a url's text with a variant."""
    pool = []
    for u in urls:
        text = (swap or {}).get(u) or pages.fetch(u)["markdown"]
        if text:
            pool.append((u, text[:PAGE_CHARS]))
        if len(pool) == POOL_SIZE:
            break
    return pool


def _answer(query: str, pool: list[tuple[str, str]]) -> tuple[dict, float]:
    results = "\n\n".join(f"[{i + 1}] {u}\n{t}" for i, (u, t) in enumerate(pool))
    resp = client().messages.create(
        model=MODEL,
        max_tokens=2000,
        output_config={"effort": "low"},
        system=SYSTEM,
        messages=[{"role": "user", "content": f"<search_results>\n{results}\n</search_results>\n\nQuestion: {query}"}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    return {"text": text}, cost_of(MODEL, resp.usage)


def simulate(query: str, pool: list[tuple[str, str]], sample: int, extra_brands: dict | None = None) -> list[str]:
    """One simulated answer, with the pool order shuffled per sample to cancel position bias."""
    order = pool[:]
    random.Random(f"{query}|{sample}").shuffle(order)
    r = cached_call("sim", {"model": MODEL, "v": SIM_VERSION, "query": query, "pool": order}, lambda: _answer(query, order), est_cost=0.003)
    return ranked_brands(r["text"], extra_brands)
