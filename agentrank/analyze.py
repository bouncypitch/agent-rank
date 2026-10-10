"""Pre-registered analysis of runs/final_results.jsonl (committed before the final test ran).

usage: python -m agentrank.analyze
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
VERSIONS = ["original", "generic", "placebo", "optimized"]
COMPARISONS = ["original", "generic", "placebo"]
ALPHA, N_BOOT, GAIN = 0.05, 10_000, 0.20


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main():
    rows = [json.loads(l) for l in (ROOT / "runs" / "final_results.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r["index"] < 5]  # Amendment 2
    targets = sorted({r["index"] for r in rows})
    rate = {v: np.array([np.mean([r["selected"] for r in rows if r["index"] == i and r["version"] == v]) for i in targets])
            for v in VERSIONS}

    print("Selection rate (target in top-3 recommendations), Claude Opus 5.5, held-out questions")
    for v in VERSIONS:
        k = sum(r["selected"] for r in rows if r["version"] == v)
        n = sum(r["version"] == v for r in rows)
        lo, hi = wilson(k, n)
        print(f"  {v:10s} {k / n:5.1%}  95% CI [{lo:.1%}, {hi:.1%}]  (n={n})")

    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(targets), size=(N_BOOT, len(targets)))
    results = []
    for c in COMPARISONS:
        d = rate["optimized"] - rate[c]
        boot = d[idx].mean(axis=1)
        p = min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))
        results.append((c, d.mean(), boot, p))

    # Holm: sort by p, compare to alpha / (m - rank); CI level adjusted the same way.
    order = sorted(range(len(results)), key=lambda j: results[j][3])
    m, passed_all, still_rejecting = len(results), True, True
    print("\nOptimized minus comparison (paired by target, bootstrap over targets, Holm-corrected)")
    for rank, j in enumerate(order):
        c, mean, boot, p = results[j]
        a = ALPHA / (m - rank)
        lo, hi = np.quantile(boot, [a / 2, 1 - a / 2])
        reject = still_rejecting and p < a and lo > 0
        still_rejecting = still_rejecting and reject
        passed_all &= reject
        print(f"  vs {c:9s} {mean:+.1%}  {100 * (1 - a):.1f}% CI [{lo:+.1%}, {hi:+.1%}]  p={p:.4f}  {'significant' if reject else 'not significant'}")
    print(f"\nPRIMARY (Amendment 1, single engine): {'PASS' if passed_all else 'FAIL'}")

    # Secondary: does the bandit need fewer rounds for later targets?
    rounds = []
    for f in sorted((ROOT / "runs" / "optimized").glob("*.json")):
        h = json.loads(f.read_text())
        base = h["history"][0]["score"]
        hit = next((x["round"] for x in h["history"] if x["score"] - base >= GAIN), len(h["history"]))
        rounds.append((h["index"], hit))
    rounds.sort()
    rho = spearmanr([i for i, _ in rounds], [r for _, r in rounds])[0]
    print(f"SECONDARY: rounds to +{GAIN:.0%} simulated gain by target order {[r for _, r in rounds]}; Spearman ρ = {rho:.2f} (want < 0)")


if __name__ == "__main__":
    main()
