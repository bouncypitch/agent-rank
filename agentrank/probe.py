"""Ask every engine every question N times; record which brands each answer recommends.

usage: python -m agentrank.probe --engines perplexity claude --samples 3 [--split train]
"""
import argparse
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from agentrank import store
from agentrank.engines import claude_engine, openai_engine, perplexity_engine
from agentrank.mentions import ranked_brands

ENGINES = {"claude": claude_engine, "openai": openai_engine, "perplexity": perplexity_engine}
ROOT = Path(__file__).resolve().parent.parent


def run(engines: list[str], samples: int, split: str = "train", workers: int = 8) -> list[dict]:
    queries = json.loads((ROOT / "data" / "queries.json").read_text())[split]
    jobs = [(e, q, s) for e in engines for q in queries for s in range(samples)]
    rows = []

    def one(e, q, s):
        r = ENGINES[e].ask(q, sample=s)
        return {"engine": e, "query": q, "sample": s, "brands": ranked_brands(r["text"]),
                "retrieved": r["retrieved"], "cited": r["cited"]}

    with ThreadPoolExecutor(workers) as pool:
        futs = {pool.submit(one, *j): j for j in jobs}
        for f in as_completed(futs):
            try:
                rows.append(f.result())
            except store.BudgetExceeded as ex:
                print("STOP:", ex)
                pool.shutdown(cancel_futures=True)
                break
            except Exception as ex:  # one failed call shouldn't sink the batch
                print("error", futs[f][:2], repr(ex)[:200])
    out = ROOT / "runs" / f"probe_{split}.jsonl"
    seen = {(r["engine"], r["query"], r["sample"]) for r in rows}
    if out.exists():  # keep earlier rows from other engines/samples
        rows += [r for r in map(json.loads, out.read_text().splitlines()) if (r["engine"], r["query"], r["sample"]) not in seen]
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return rows


def summarize(rows: list[dict], top_k: int = 3):
    by_engine = defaultdict(list)
    for r in rows:
        by_engine[r["engine"]].append(r)
    for e, rs in by_engine.items():
        c = Counter(b for r in rs for b in r["brands"][:top_k])
        print(f"\n== {e}: {len(rs)} answers — share of answers recommending brand in top {top_k}")
        for b, n in c.most_common(15):
            print(f"  {b:26s} {n / len(rs):5.0%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", nargs="+", default=["perplexity"])
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--split", default="train")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    rows = run(a.engines, a.samples, a.split, a.workers)
    summarize([r for r in rows if r["engine"] in a.engines])
    print(f"\nspent so far: ${store.spent():.2f} of ${store.SPEND_CAP_USD:.0f} cap")
