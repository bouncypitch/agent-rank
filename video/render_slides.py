"""Render deck slides (from the deck's project/slides/*.html) and the demo page to 1920x1080 PNGs with headless Chrome."""
import re, subprocess, sys
from pathlib import Path

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
DECK = Path(sys.argv[1])            # folder holding project/slides
OUT = Path(__file__).parent / "frames"
FONTS = '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400..700&family=IBM+Plex+Sans:wght@400;600&display=swap">'


def localize(section: str) -> str:
    # The deck runtime draws x-shape / x-icon; for a local render, arrows become a glyph and icons are dropped.
    section = re.sub(r'<x-shape kind="arrow-right"[^>]*></x-shape>', '<div style="align-self:center;font-size:56px;color:#E2703A;padding:0 8px">&#8594;</div>', section)
    section = re.sub(r"<x-icon[^>]*></x-icon>", "", section)
    section = re.sub(r"<aside>.*?</aside>", "", section, flags=re.S)
    return section.replace("<section", '<section style-box="1"', 1)


def shot(html: str, png: Path):
    tmp = OUT / "_tmp.html"
    tmp.write_text(f"<!doctype html><html><head><meta charset='utf-8'>{FONTS}<style>body{{margin:0}} section{{width:1920px;height:1080px;box-sizing:border-box;position:relative;overflow:hidden}} h1,h2,h3,p,ul,table{{margin:0}} td,th{{padding:12px 20px;border-bottom:1px solid rgba(128,128,128,.35)}} table{{border-collapse:collapse}}</style></head><body>{html}</body></html>")
    subprocess.run([CHROME, "--headless=new", "--hide-scrollbars", "--window-size=1920,1080", "--virtual-time-budget=4000",
                    f"--screenshot={png}", tmp.as_uri()], check=True, capture_output=True)


for f in sorted((DECK / "project" / "slides").glob("*.html")):
    shot(localize(f.read_text()), OUT / f"{f.stem}.png")
    print("rendered", f.stem)
