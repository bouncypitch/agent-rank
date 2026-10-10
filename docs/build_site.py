"""Inline runs/public/summary.json into docs/template.html -> docs/index.html (works on GitHub Pages and from file://)."""
from pathlib import Path
root = Path(__file__).resolve().parent.parent
data = (root / "runs" / "public" / "summary.json").read_text().replace("</", "<\\/")
(root / "docs" / "index.html").write_text((root / "docs" / "template.html").read_text().replace("__DATA__", data))
print("wrote docs/index.html")
