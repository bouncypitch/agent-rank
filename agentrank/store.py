"""SQLite cache for every paid call, plus a hard spend cap.

Every engine/simulator call goes through `cached_call`: identical requests are
served from disk for free, and new requests are refused once the cap is hit.
"""
import hashlib
import json
import os
import sqlite3
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DB_PATH = Path(__file__).resolve().parent.parent / "runs" / "cache.sqlite"
SPEND_CAP_USD = float(os.getenv("SPEND_CAP_USD", "40"))

_lock = threading.Lock()


class BudgetExceeded(RuntimeError):
    pass


def _conn():
    DB_PATH.parent.mkdir(exist_ok=True)
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.execute("CREATE TABLE IF NOT EXISTS calls (key TEXT PRIMARY KEY, kind TEXT, request TEXT, response TEXT, cost REAL, ts REAL)")
    return c


_db = _conn()


def spent() -> float:
    with _lock:
        return _db.execute("SELECT COALESCE(SUM(cost), 0) FROM calls").fetchone()[0]


def cached_call(kind: str, request: dict, fn, est_cost: float):
    """Return fn()'s result for `request`, from cache if seen before.

    fn must return (result_dict, actual_cost_usd). est_cost is checked against
    the cap before calling so a single call can't blow far past it.
    """
    key = hashlib.sha256(json.dumps({"kind": kind, **request}, sort_keys=True).encode()).hexdigest()
    with _lock:
        row = _db.execute("SELECT response FROM calls WHERE key = ?", (key,)).fetchone()
    if row:
        return json.loads(row[0])
    if spent() + est_cost > SPEND_CAP_USD:
        raise BudgetExceeded(f"spend cap ${SPEND_CAP_USD:.2f} reached (spent ${spent():.2f})")
    result, cost = fn()
    with _lock:
        _db.execute(
            "INSERT OR REPLACE INTO calls VALUES (?, ?, ?, ?, ?, ?)",
            (key, kind, json.dumps(request, sort_keys=True), json.dumps(result), cost, time.time()),
        )
        _db.commit()
    return result
