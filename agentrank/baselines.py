"""The two comparison versions for each target: a generic LLM rewrite and a size-matched placebo edit."""
import json
from pathlib import Path

from agentrank import optimizer
from agentrank.engines.claude_engine import client, cost_of
from agentrank.store import cached_call

ROOT = Path(__file__).resolve().parent.parent

GENERIC = "Improve this web page."
PLACEBO = ("Rephrase this page sentence by sentence. Keep every fact, the order of content and the structure exactly the "
           "same: do not add, remove or move sections, tables, lists or FAQs, and do not change what is emphasized. "
           "Adjust wording so the result is about {n} characters long.")


def _rewrite(page: str, instruction: str, attempt: int) -> tuple[dict, float]:
    resp = client().messages.create(
        model=optimizer.EDITOR_MODEL, max_tokens=8000, output_config={"effort": "low"}, system=optimizer.EDITOR_SYSTEM,
        messages=[{"role": "user", "content": f"Edit instruction: {instruction}\n\n<original_page>\n{page}\n</original_page>"}],
    )
    return {"page": "".join(b.text for b in resp.content if b.type == "text")}, cost_of(optimizer.EDITOR_MODEL, resp.usage)


def make(page: str, instruction: str, kind: str) -> dict:
    """Rewrite, then run the same claim checker the optimizer faces; retry up to twice if it fails."""
    for attempt in range(3):
        out = cached_call(f"baseline:{kind}", {"model": optimizer.EDITOR_MODEL, "i": instruction, "page": page, "a": attempt},
                          lambda: _rewrite(page, instruction, attempt), est_cost=0.05)["page"]
        bad = optimizer.check(page, out)
        if not bad:
            return {"page": out, "attempts": attempt + 1, "unsupported": []}
    return {"page": out, "attempts": 3, "unsupported": bad}


def build(index: int) -> dict:
    path = next((ROOT / "runs" / "optimized").glob(f"{index:02d}_*.json"))
    res = json.loads(path.read_text())
    original, optimized = res["original"], res["page"]
    return {"target": res["target"], "url": res["url"], "original": original, "optimized": optimized,
            "generic": make(original, GENERIC, "generic"),
            "placebo": make(original, PLACEBO.format(n=len(optimized)), "placebo")}
