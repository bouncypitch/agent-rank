"""Build runs/demo.html from the run data: diagnosis, before/after page, optimization trace, final result.

usage: python -m agentrank.demo_page   (then open runs/demo.html)
"""
import html
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OPS = {"answer_first": "Answer first", "specs_table": "Specs table", "use_case_sections": "Use-case sections",
       "customer_evidence": "Customer evidence", "price_signal": "Price signal", "specificity": "Specificity", "buyer_faq": "Buyer FAQ"}


def esc(s):
    return html.escape(s or "")


def probe_shares():
    rows = [json.loads(l) for l in (ROOT / "runs" / "probe_train.jsonl").read_text().splitlines()]
    out = {}
    for e in ("claude", "openai", "perplexity"):
        rs = [r for r in rows if r["engine"] == e]
        c = Counter(b for r in rs for b in set(r["brands"][:3]))
        out[e] = {b: n / len(rs) for b, n in c.items()}
    return out


def final_rates():
    p = ROOT / "runs" / "final_results.jsonl"
    if not p.exists():
        return {}
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    out = {}
    for r in rows:
        out.setdefault(r["target"], {}).setdefault(r["version"], []).append(r["selected"])
    return {t: {v: sum(x) / len(x) for v, x in vs.items()} for t, vs in out.items()}


def company_card(res, shares, final):
    t = res["target"]
    eng = " · ".join(f"{n} {shares[e].get(t, 0):.0%}" for e, n in (("claude", "Claude"), ("openai", "OpenAI"), ("perplexity", "Perplexity")))
    trace = "".join(
        f"<li><b>Round {h['round']}</b>: tried {', '.join(OPS.get(o, o) for o in h['ops'])} → best {h['score']:.0%}</li>"
        for h in res["history"][1:])
    fin = final.get(t)
    fin_html = ""
    if fin:
        bars = "".join(f"<div class='bar'><span>{v}</span><div class='track'><div style='width:{fin.get(v, 0) * 100:.0f}%' class='fill {v}'></div></div><em>{fin.get(v, 0):.0%}</em></div>"
                       for v in ("original", "generic", "placebo", "optimized"))
        fin_html = f"<h4>Real engine (Claude Opus 5.5), held-out questions</h4>{bars}"
    return f"""
<section class="card" id="c{res['index']}">
  <header><h2>{esc(t)}</h2><a href="{esc(res['url'])}">{esc(res['url'])}</a></header>
  <p class="diag"><b>Today, recommended in top 3:</b> {eng}</p>
  <p class="sim"><b>Simulated, when its page is among the sources:</b> {res['history'][0]['score']:.0%} → <b>{res['score']:.0%}</b></p>
  <ul class="trace">{trace}</ul>
  {fin_html}
  <div class="pages">
    <div><h4>Original page</h4><pre>{esc(res['original'][:3500])}</pre></div>
    <div><h4>Optimized page <span class="badge">✓ no invented claims (checker)</span></h4><pre>{esc(res['page'][:3500])}</pre></div>
  </div>
</section>"""


def main():
    shares, final = probe_shares(), final_rates()
    results = [json.loads(f.read_text()) for f in sorted((ROOT / "runs" / "optimized").glob("*.json"))]
    nav = "".join(f"<a href='#c{r['index']}'>{esc(r['target'])}</a>" for r in results)
    cards = "".join(company_card(r, shares, final) for r in results)
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>AgentRank demo</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{{--ink:#121A2B;--paper:#F6F4EE;--card:#FFFDF8;--line:#DDD7CA;--accent:#E2703A;--blue:#3A6FC4;--muted:#4A5568}}
body{{margin:0;background:var(--paper);color:var(--ink);font:18px/1.5 -apple-system,'IBM Plex Sans',Arial,sans-serif}}
.top{{background:var(--ink);color:var(--paper);padding:28px 32px}} .top h1{{margin:0;font-size:34px}} .top p{{margin:6px 0 0;color:#B9C2D3}}
nav{{display:flex;flex-wrap:wrap;gap:8px;padding:16px 32px;border-bottom:1px solid var(--line);background:#EAE6DC}}
nav a{{color:var(--ink);text-decoration:none;background:var(--card);border:1px solid var(--line);border-radius:999px;padding:6px 14px;font-size:15px}}
main{{max-width:1280px;margin:0 auto;padding:24px 16px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:24px;margin:0 0 24px}}
.card header{{display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:8px}} .card h2{{margin:0}}
.card header a{{color:var(--blue);font-size:15px}} .sim b{{color:var(--accent)}}
.trace{{color:var(--muted);font-size:16px}}
.pages{{display:grid;grid-template-columns:1fr 1fr;gap:16px}} @media(max-width:800px){{.pages{{grid-template-columns:1fr}}}}
pre{{white-space:pre-wrap;background:#F2EFE7;border-radius:10px;padding:14px;font-size:13px;max-height:420px;overflow:auto}}
.badge{{background:#E6F0E4;color:#24572A;border-radius:999px;padding:2px 10px;font-size:13px;margin-left:6px}}
.bar{{display:grid;grid-template-columns:110px 1fr 56px;align-items:center;gap:10px;margin:6px 0}} .bar span{{font-size:15px}}
.track{{height:18px;border-radius:9px;background:#ECE7DC}} .fill{{height:18px;border-radius:9px;background:#9C958A}} .fill.optimized{{background:var(--accent)}} .bar em{{font-style:normal;font-size:15px}}
</style></head><body>
<div class="top"><h1>AgentRank — get recommended by the AI agents</h1><p>Cobot companies · real probes of Claude, OpenAI and Perplexity · simulator-guided, checker-verified rewrites</p></div>
<nav>{nav}</nav><main>{cards}</main></body></html>"""
    out = ROOT / "runs" / "demo.html"
    out.write_text(page)
    print(f"wrote {out} with {len(results)} companies")


if __name__ == "__main__":
    main()
