"""Live demo: paste a robotics company URL, watch it get diagnosed and optimized (simulated, with error bars).

usage: .venv/bin/python -m agentrank.app   then open http://127.0.0.1:5050
"""
import json
import math
import queue
import random
import re
import sqlite3
import threading
from pathlib import Path

from flask import Flask, Response, request

from agentrank import optimizer, pages, run_optimize, store

ROOT = Path(__file__).resolve().parent.parent
APP_BANDIT = ROOT / "runs" / "bandit_app.json"
SAMPLES = 4          # 8 questions x 4 = 32 simulated answers per version (2x the optimizer's original 16)
ROUNDS = 3
Z = 1.64             # keep an edit only if its gain clears ~1.64 standard errors (one-sided 95%)
RUN_COST_CAP = 1.50  # stop a single run beyond this much new spend

app = Flask(__name__)


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def probe_mentions(name: str) -> dict:
    """How often each real engine named this company in today's probe answers (read from the cache, free)."""
    db = sqlite3.connect(store.DB_PATH)
    pat = re.compile(r"(?<![\w-])" + re.escape(name) + r"(?![\w-])", re.I)
    out = {}
    for kind, label in (("engine:claude", "Claude"), ("engine:openai", "OpenAI"), ("engine:perplexity", "Perplexity")):
        texts = [json.loads(r)["text"] for (r,) in db.execute("SELECT response FROM calls WHERE kind = ?", (kind,))]
        out[label] = {"answers": len(texts), "mentioned": sum(bool(pat.search(t or "")) for t in texts)}
    return out


KNOWN_NAMES = {t["url"].split("//")[1].removeprefix("www.").split("/")[0]: t["name"]
               for t in json.loads((ROOT / "data" / "targets.json").read_text())["targets"]}
KNOWN_NAMES.update({"kassowrobots.com": "Kassow Robots", "formic.co": "Formic", "kinovarobotics.com": "Kinova"})


def infer_name(url: str) -> str:
    """Company name from our target list, else the page <title> segment that best matches the domain."""
    domain = re.sub(r"^https?://(www\.)?", "", url).split("/")[0].lower()
    if domain in KNOWN_NAMES:
        return KNOWN_NAMES[domain]
    stem = domain.split(".")[0].replace("-", "")
    try:
        html = pages.requests.get(url, timeout=15, headers={"User-Agent": pages.UA}).text[:200_000]
        title = re.search(r"<title[^>]*>(.*?)</title>", html, re.S | re.I)
        parts = [p.strip() for p in re.split(r"[|\-–—:·]", title.group(1))] if title else []
        parts = [p for p in parts if 2 < len(p) < 40]
        best = max(parts, key=lambda p: sum(c in stem for c in p.lower().replace(" ", "")) / len(p), default=None)
        if best:
            return best
    except Exception:
        pass
    return stem.capitalize()


def score(page, target, contexts):
    rate = optimizer.score(page, target, contexts, SAMPLES)
    n = len(contexts) * SAMPLES
    k = round(rate * n)
    lo, hi = wilson(k, n)
    return {"rate": rate, "k": k, "n": n, "lo": lo, "hi": hi}


def significant_gain(new, old):
    se = math.sqrt(new["rate"] * (1 - new["rate"]) / new["n"] + old["rate"] * (1 - old["rate"]) / old["n"]) or 1e-9
    return (new["rate"] - old["rate"]) / se >= Z


def run(url: str, name: str, emit):
    start_spend = store.spent()
    target = {"name": name, "url": url}
    emit("status", {"msg": f"Fetching {url} …"})
    page = pages.fetch(url)
    if not page["ok"] or len(page["markdown"]) < 300:
        return emit("error", {"msg": "Couldn't read enough text from that page (it may render with JavaScript). Try a product or about page."})
    original = page["markdown"]
    emit("page", {"chars": len(original), "preview": original[:600]})
    emit("probe", probe_mentions(name))

    contexts = run_optimize.contexts_for(target)
    emit("status", {"msg": f"Simulating {len(contexts) * SAMPLES} buyer answers with your page among real competitor pages …"})
    best, best_s = original, score(original, target, contexts)
    emit("baseline", best_s)
    attempt = None  # best-scoring edit that passed the checker, kept or not (shown for comparison)

    # Start from what the bandit learned on the 10 locked companies, but keep the app's updates separate.
    bandit = optimizer.Bandit()
    optimizer.BANDIT_PATH, saved_path = APP_BANDIT, optimizer.BANDIT_PATH
    try:
        rng = random.Random(f"{url}|app")
        for rnd in range(1, ROUNDS + 1):
            if store.spent() - start_spend > RUN_COST_CAP:
                emit("status", {"msg": "Per-run cost cap reached; stopping early."})
                break
            ctx = str(optimizer.page_features(best))
            op = bandit.pick(ctx, rng)
            emit("status", {"msg": f"Round {rnd}: bandit chose “{optimizer.OPERATORS[op][:70]}…”"})
            cand = optimizer.edit(best, op)
            bad = optimizer.check(original, cand)
            if bad:
                bandit.update(ctx, op, success=False)
                emit("round", {"round": rnd, "op": op, "rejected": bad[:3]})
                continue
            s = score(cand, target, contexts)
            keep = significant_gain(s, best_s)
            if attempt is None or s["rate"] > attempt["score"]["rate"]:
                attempt = {"page": cand, "score": s, "op": op, "kept": keep}
            bandit.update(ctx, op, success=keep)
            emit("round", {"round": rnd, "op": op, "score": s, "kept": keep})
            if keep:
                best, best_s = cand, s
    finally:
        optimizer.BANDIT_PATH = saved_path
    emit("done", {"original": original, "final": best, "final_score": best_s, "attempt": attempt,
                  "cost": round(store.spent() - start_spend, 2), "spent_total": round(store.spent(), 2)})


@app.post("/api/run")
def api_run():
    body = request.get_json(force=True)
    url, name = (body.get("url") or "").strip(), (body.get("name") or "").strip()
    if url and not re.match(r"^https?://", url):
        url = "https://" + url
    if not re.match(r"^https?://[^/\s]+\.[^/\s]+", url):
        return Response(json.dumps({"event": "error", "msg": "Enter a company website, e.g. doosanrobotics.com"}) + "\n",
                        mimetype="application/x-ndjson")
    q = queue.Queue()

    def emit(event, data):
        q.put(json.dumps({"event": event, **data}))

    def worker():
        try:
            company = name or infer_name(url)
            emit("company", {"name": company, "url": url})
            run(url, company, emit)
        except store.BudgetExceeded as ex:
            emit("error", {"msg": f"Budget cap reached: {ex}"})
        except Exception as ex:  # surface failures in the UI instead of hanging the stream
            emit("error", {"msg": f"{type(ex).__name__}: {ex}"[:300]})
        finally:
            q.put(None)

    threading.Thread(target=worker, daemon=True).start()

    def stream():
        while (item := q.get()) is not None:
            yield item + "\n"
    return Response(stream(), mimetype="application/x-ndjson")


@app.get("/")
def index():
    return (ROOT / "agentrank" / "app.html").read_text()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, threaded=True)
