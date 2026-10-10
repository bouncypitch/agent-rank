"""Write runs/public/summary.json: every number the site and deck show, and none of the companies' page text.

usage: python -m agentrank.export_public
"""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Simulator validation, as printed by `python -m agentrank.validate --engine <e>` (training questions, simulator v2).
VALIDATION = {
    "perplexity": {"pair_r": 0.25, "brand_spearman": 0.81, "brand_r_without_ur_fanuc": 0.75, "top1_hit": 0.95},
    "openai": {"pair_r": 0.42, "brand_spearman": 0.91, "brand_r_without_ur_fanuc": 0.91, "top1_hit": 0.94},
    "claude": {"pair_r": 0.09, "brand_spearman": 0.76, "brand_r_without_ur_fanuc": 0.88, "top1_hit": 0.78},
    "perplexity_self_consistency_pair_r": 0.21,
    "simulator_v1_pair_r_perplexity": -0.05,
}


def main():
    probe = [json.loads(l) for l in (ROOT / "runs" / "probe_train.jsonl").read_text().splitlines()]
    shares = {}
    for e in ("claude", "openai", "perplexity"):
        rs = [r for r in probe if r["engine"] == e]
        c = Counter(b for r in rs for b in set(r["brands"][:3]))
        shares[e] = {"answers": len(rs), "pages_read_avg": round(sum(len(r["retrieved"]) for r in rs) / len(rs), 1),
                     "top3_share": {b: round(n / len(rs), 3) for b, n in c.most_common()}}

    final = [json.loads(l) for l in (ROOT / "runs" / "final_results.jsonl").read_text().splitlines()]
    final = [r for r in final if r["index"] < 5]  # Amendment 2
    companies = []
    for f in sorted((ROOT / "runs" / "optimized").glob("*.json")):
        res = json.loads(f.read_text())
        rows = [r for r in final if r["target"] == res["target"]]
        companies.append({
            "name": res["target"], "url": res["url"], "index": res["index"],
            "sim_before": res["history"][0]["score"], "sim_after": res["score"],
            "page_changed": res["page"] != res["original"],
            "rounds": [{"round": h["round"], "best": h["score"], "ops": h["ops"]} for h in res["history"][1:]],
            "today_top3": {e: shares[e]["top3_share"].get(res["target"], 0.0) for e in shares},
            "final": {v: round(sum(r["selected"] for r in rows if r["version"] == v) / max(1, sum(r["version"] == v for r in rows)), 3)
                      for v in ("original", "generic", "placebo", "optimized")} if rows else None,
        })

    overall = {v: round(sum(r["selected"] for r in final if r["version"] == v) / sum(r["version"] == v for r in final), 3)
               for v in ("original", "generic", "placebo", "optimized")}
    out = {"generated": "2026-10-10", "engines": shares, "validation": VALIDATION, "companies": companies,
           "final_test": {"engine": "claude-opus-5-5 (web search off, real competitor pages)", "companies": 5,
                          "heldout_questions": 8, "trials": len(final), "rates": overall, "verdict": "FAIL",
                          "optimized_minus": {"original": {"diff": 0.05, "p": 0.538}, "placebo": {"diff": 0.025, "p": 0.635},
                                              "generic": {"diff": -0.025, "p": 0.868}}}}
    (ROOT / "runs" / "public").mkdir(exist_ok=True)
    (ROOT / "runs" / "public" / "summary.json").write_text(json.dumps(out, indent=1))
    print("wrote runs/public/summary.json:", len(companies), "companies")


if __name__ == "__main__":
    main()
