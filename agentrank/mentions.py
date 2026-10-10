"""Turn an engine's answer text into an ordered list of recommended brands."""
import json
import re
from pathlib import Path

BRANDS = json.loads((Path(__file__).resolve().parent.parent / "data" / "brands.json").read_text())["brands"]
_PATTERNS = {b: re.compile(r"(?<![\w-])(" + "|".join(re.escape(a) for a in sorted(al, key=len, reverse=True)) + r")(?![\w-])", re.I)
             for b, al in BRANDS.items()}


def ranked_brands(text: str, extra: dict[str, list[str]] | None = None) -> list[str]:
    """Brands in order of first appearance. `extra` adds e.g. a target company not in the list."""
    pats = dict(_PATTERNS)
    for b, al in (extra or {}).items():
        pats[b] = re.compile(r"(?<![\w-])(" + "|".join(re.escape(a) for a in al) + r")(?![\w-])", re.I)
    hits = []
    for b, pat in pats.items():
        m = pat.search(text or "")
        if m:
            hits.append((m.start(), b))
    return [b for _, b in sorted(hits)]
