"""Perplexity as an AI answer engine, via its Agent API (search built in)."""
import os
import random
import time

import requests

from agentrank.engines.claude_engine import SYSTEM, PROMPT_VERSION
from agentrank.store import cached_call

PRESET = os.getenv("PERPLEXITY_PRESET", "fast")  # "fast" is the old Sonar tier
EST_COST = float(os.getenv("PERPLEXITY_EST_COST_PER_CALL", "0.01"))
URL = "https://api.perplexity.ai/v1/responses"


def _ask(query: str) -> tuple[dict, float]:
    for attempt in range(8):  # new accounts get a low rate limit; back off on 429
        r = requests.post(URL, timeout=120, headers={"Authorization": f"Bearer {os.getenv('PERPLEXITY_API_KEY')}"},
                          json={"preset": PRESET, "instructions": SYSTEM, "input": query})
        if r.status_code != 429:
            break
        time.sleep(min(60, float(r.headers.get("retry-after") or 2 ** attempt)) + random.random())
    r.raise_for_status()
    d = r.json()
    text, cited, retrieved = [], [], []
    for item in d.get("output", []):
        if item.get("type") == "search_results":
            retrieved += [x["url"] for x in item.get("results", []) if x.get("url")]
        elif item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    text.append(part.get("text", ""))
                    cited += [a["url"] for a in part.get("annotations") or [] if a.get("url")]
    return {"text": "".join(text), "cited": list(dict.fromkeys(cited)), "retrieved": list(dict.fromkeys(retrieved)),
            "stop_reason": d.get("status"), "model": d.get("model")}, EST_COST


def ask(query: str, sample: int = 0) -> dict:
    return cached_call("engine:perplexity", {"preset": PRESET, "v": PROMPT_VERSION, "query": query, "sample": sample},
                       lambda: _ask(query), est_cost=EST_COST)
