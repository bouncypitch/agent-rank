"""Optimize the locked targets in order; the bandit carries over between them.

usage: python -m agentrank.run_optimize --start 0 --count 1 --rounds 4
"""
import argparse
import json
from pathlib import Path
from urllib.parse import urlparse

from agentrank import optimizer, pages, store, validate

ROOT = Path(__file__).resolve().parent.parent
# Broad buyer questions every cobot maker could plausibly win (training split only).
CONTEXT_QUERIES = [
    "Best cobot for a small manufacturer with no robotics experience",
    "What is the most affordable 6-axis cobot for a startup?",
    "What are the best alternatives to Universal Robots?",
    "Which cobot has the best payload-to-price ratio?",
    "Top cobot manufacturers in 2026",
    "What cobot is easiest to program without coding?",
    "Which cobot is best for high-mix low-volume production?",
    "What cobot should a contract manufacturer buy first?",
]


def contexts_for(target: dict) -> list[tuple[str, list[str]]]:
    """Each question's pool = pages real engines read for it, minus the target's own site."""
    domain = urlparse(target["url"]).netloc.removeprefix("www.")
    out = []
    for q in CONTEXT_QUERIES:
        rows = validate.load_probe("perplexity").get(q, []) + validate.load_probe("openai").get(q, [])
        urls = [u for u in validate.pool_urls(rows) if domain not in u]
        out.append((q, urls))
    return out


def main(start: int, count: int, rounds: int):
    targets = json.loads((ROOT / "data" / "targets.json").read_text())["targets"]
    outdir = ROOT / "runs" / "optimized"
    outdir.mkdir(exist_ok=True)
    for i, t in enumerate(targets[start:start + count], start):
        original = pages.fetch(t["url"])["markdown"]
        res = optimizer.optimize(t, original, contexts_for(t), rounds=rounds)
        res.update(index=i, url=t["url"], original=original)
        (outdir / f"{i:02d}_{t['name'].replace(' ', '_')}.json").write_text(json.dumps(res, indent=1))
        print(f"[{i}] {t['name']}: {res['history'][0]['score']:.0%} -> {res['score']:.0%}   (spent ${store.spent():.2f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=4)
    a = ap.parse_args()
    main(a.start, a.count, a.rounds)
