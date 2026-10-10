"""Claude as an AI answer engine: a buyer question in, an answer with web citations out."""
import os

import anthropic

from agentrank.store import cached_call

MODEL = os.getenv("CLAUDE_ENGINE_MODEL", "claude-opus-5-5")
PRICES = {  # USD per 1M tokens (input, output)
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
}
WEB_SEARCH_USD = 0.01  # per search request
# Answer engines ground product recommendations in live search; ask Claude to do the same.
SYSTEM = "You are an AI assistant answering a buyer's question. Search the web for current information before recommending products, and cite your sources."
PROMPT_VERSION = 3

_client = None


def client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def cost_of(model: str, usage) -> float:
    p_in, p_out = PRICES.get(model, (4.0, 20.0))
    tokens_in = (usage.input_tokens or 0) + (usage.cache_creation_input_tokens or 0) + 0.1 * (usage.cache_read_input_tokens or 0)
    searches = getattr(getattr(usage, "server_tool_use", None), "web_search_requests", 0) or 0
    return tokens_in * p_in / 1e6 + (usage.output_tokens or 0) * p_out / 1e6 + searches * WEB_SEARCH_USD


def _ask(query: str) -> tuple[dict, float]:
    messages = [{"role": "user", "content": query}]
    cost, cited, retrieved = 0.0, [], []
    for _ in range(4):  # resume server-side pause_turn a few times at most
        text = []
        resp = client().beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=SYSTEM,
            tools=[{"type": "web_search_20260209", "name": "web_search", "max_uses": 3}],
            messages=messages,
        )
        cost += cost_of(MODEL, resp.usage)
        for block in resp.content:
            if block.type == "web_search_tool_result" and isinstance(block.content, list):
                retrieved += [r.url for r in block.content if getattr(r, "url", None)]
            if block.type == "text":
                text.append(block.text)
                for c in getattr(block, "citations", None) or []:
                    if getattr(c, "url", None):
                        cited.append(c.url)
            else:
                text = []  # keep only the answer written after the last tool step
        if resp.stop_reason != "pause_turn":
            break
        messages = messages + [{"role": "assistant", "content": resp.content}]
    return {"text": "".join(text), "cited": list(dict.fromkeys(cited)), "retrieved": list(dict.fromkeys(retrieved)),
            "stop_reason": resp.stop_reason}, cost


def ask(query: str, sample: int = 0) -> dict:
    return cached_call("engine:claude", {"model": MODEL, "v": PROMPT_VERSION, "query": query, "sample": sample}, lambda: _ask(query), est_cost=0.08)
