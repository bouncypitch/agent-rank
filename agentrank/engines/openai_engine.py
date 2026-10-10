"""OpenAI (ChatGPT-style) as an AI answer engine, via the Responses API with web search."""
import os
import re

from openai import OpenAI

from agentrank.engines.claude_engine import SYSTEM, PROMPT_VERSION
from agentrank.store import cached_call

MODEL = os.getenv("OPENAI_ENGINE_MODEL", "gpt-5.5")
EST_COST = float(os.getenv("OPENAI_EST_COST_PER_CALL", "0.03"))

_client = None


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def _ask(query: str) -> tuple[dict, float]:
    resp = client().responses.create(
        model=MODEL,
        instructions=SYSTEM,
        input=query,
        tools=[{"type": "web_search"}],
        include=["web_search_call.action.sources"],
        reasoning={"effort": "low"},
    )
    cited, retrieved = [], []
    for item in resp.output:
        if item.type == "web_search_call":
            for src in getattr(getattr(item, "action", None), "sources", None) or []:
                if getattr(src, "url", None):
                    retrieved.append(src.url)
        elif item.type == "message":
            for part in item.content:
                for ann in getattr(part, "annotations", None) or []:
                    if getattr(ann, "type", "") == "url_citation":
                        cited.append(ann.url)
    return {"text": resp.output_text, "cited": list(dict.fromkeys(cited)),
            "retrieved": list(dict.fromkeys(retrieved)), "stop_reason": resp.status}, EST_COST


def _clean(url: str) -> str:
    return re.sub(r"[?&]utm_source=openai$", "", url)


def ask(query: str, sample: int = 0) -> dict:
    r = cached_call("engine:openai", {"model": MODEL, "v": PROMPT_VERSION, "query": query, "sample": sample},
                       lambda: _ask(query), est_cost=EST_COST)
    return {**r, "cited": [_clean(u) for u in r["cited"]], "retrieved": [_clean(u) for u in r["retrieved"]]}
