"""Go/no-go: does the simulator recommend the same brands the real engine does?

For each training question, the pool is the pages that engine actually read.
We compare real vs simulated top-3 recommendation rates per (question, brand).

usage: python -m agentrank.validate --engine perplexity --samples 4
"""
import argparse
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scipy.stats import pearsonr, spearmanr

from agentrank import simulator, store

ROOT = Path(__file__).resolve().parent.parent


def load_probe(engine: str, split: str = "train") -> dict[str, list[dict]]:
    rows = [json.loads(l) for l in (ROOT / "runs" / f"probe_{split}.jsonl").read_text().splitlines()]
    by_q = defaultdict(list)
    for r in rows:
        if r["engine"] == engine:
            by_q[r["query"]].append(r)
    return by_q


def pool_urls(rows: list[dict]) -> list[str]:
    """Pages the engine read for this question, most frequently retrieved first."""
    c = Counter(u for r in rows for u in r["retrieved"])
    return [u for u, _ in c.most_common()]


def rates(brand_lists: list[list[str]], k: int = 3) -> Counter:
    c = Counter(b for bl in brand_lists for b in set(bl[:k]))
    return Counter({b: n / len(brand_lists) for b, n in c.items()})


def main(engine: str, samples: int, workers: int = 6):
    by_q = load_probe(engine)
    pairs, per_q = [], []

    def one(q):
        rows = by_q[q]
        pool = simulator.build_pool(pool_urls(rows))
        if len(pool) < 3:
            return q, None, None
        sim = [simulator.simulate(q, pool, s) for s in range(samples)]
        return q, rates([r["brands"] for r in rows]), rates(sim)

    with ThreadPoolExecutor(workers) as ex:
        for q, real, sim in ex.map(one, list(by_q)):
            if real is None:
                continue
            for b in set(real) | set(sim):
                pairs.append((real.get(b, 0.0), sim.get(b, 0.0)))
            top_real = max(real, key=real.get) if real else None
            per_q.append((q, top_real, top_real in sim and sim[top_real] > 0))

    xs, ys = zip(*pairs)
    print(f"{engine}: {len(per_q)} questions, {len(pairs)} (question, brand) pairs")
    print(f"  Pearson r  = {pearsonr(xs, ys)[0]:.2f}   (bar: SimGym r≈0.55)")
    print(f"  Spearman ρ = {spearmanr(xs, ys)[0]:.2f}")
    print(f"  real #1 brand also recommended by simulator: {sum(h for *_, h in per_q) / len(per_q):.0%} of questions")
    print(f"  spent so far: ${store.spent():.2f}")
    return pairs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="perplexity")
    ap.add_argument("--samples", type=int, default=4)
    a = ap.parse_args()
    main(a.engine, a.samples)
