"""Pre-registered final test (see PREREGISTRATION.md, Amendment 1).

The real production model (Claude Opus 5.5, web search off) answers held-out buyer
questions from a pool of real competitor pages (the pages Perplexity retrieved for
that question) plus one version of the target's page. For a given (target, question)
the target sits at the same position for all four versions, so versions are paired;
the position varies across questions.

usage: python -m agentrank.final_eval
"""
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

from agentrank import baselines, optimizer, pages, simulator, store, validate
from agentrank.engines.claude_engine import MODEL, client, cost_of
from agentrank.mentions import ranked_brands
from agentrank.store import cached_call

ROOT = Path(__file__).resolve().parent.parent
N_TARGETS, N_QUESTIONS, POOL = 10, 8, 10
VERSIONS = ["original", "generic", "placebo", "optimized"]


def _answer(query, pool):
    results = "\n\n".join(f"[{i + 1}] {u}\n{t}" for i, (u, t) in enumerate(pool))
    resp = client().beta.messages.create(
        model=MODEL, max_tokens=4000, output_config={"effort": "low"},
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        system=simulator.SYSTEMS["v2"],
        messages=[{"role": "user", "content": f"<search_results>\n{results}\n</search_results>\n\nQuestion: {query}"}],
    )
    return {"text": "".join(b.text for b in resp.content if b.type == "text"), "stop": resp.stop_reason}, cost_of(MODEL, resp.usage)


def main():
    targets = json.loads((ROOT / "data" / "targets.json").read_text())["targets"][:N_TARGETS]
    questions = json.loads((ROOT / "data" / "queries.json").read_text())["heldout"][:N_QUESTIONS]
    probe = validate.load_probe("perplexity", split="heldout")
    versions = {i: baselines.build(i) for i in range(N_TARGETS)}
    (ROOT / "runs" / "final_versions.json").write_text(json.dumps(versions, indent=1))

    jobs = []
    for i, t in enumerate(targets):
        domain = urlparse(t["url"]).netloc.removeprefix("www.")
        for q in questions:
            urls = [u for u in validate.pool_urls(probe[q]) if domain not in u]
            comp = simulator.build_pool(urls)[: POOL - 1]
            pos = random.Random(f"{t['name']}|{q}").randrange(POOL)
            for v in VERSIONS:
                page = versions[i][v] if v in ("original", "optimized") else versions[i][v]["page"]
                pool = comp[:pos] + [(t["url"], page[: simulator.PAGE_CHARS])] + comp[pos:]
                jobs.append((i, t, q, v, pool, pos))

    def run(job):
        i, t, q, v, pool, pos = job
        r = cached_call("final", {"model": MODEL, "query": q, "pool": pool}, lambda: _answer(q, pool), est_cost=0.06)
        brands = ranked_brands(r["text"], optimizer.target_aliases(t["name"]))
        return {"target": t["name"], "index": i, "question": q, "version": v, "position": pos,
                "selected": t["name"] in brands[:3], "rank": brands.index(t["name"]) + 1 if t["name"] in brands else None}

    out = ROOT / "runs" / "final_results.jsonl"
    rows = []
    with ThreadPoolExecutor(6) as ex:
        for row in ex.map(run, jobs):
            rows.append(row)
            out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"{len(rows)} trials written to {out}; spent ${store.spent():.2f}")


if __name__ == "__main__":
    main()
