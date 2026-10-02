"""Inline web/data.json into web/template.html -> web/index.html (single self-contained page)."""
from pathlib import Path
W = Path(__file__).resolve().parents[1] / "web"
data = (W / "data.json").read_text().replace("</", "<\\/")
html = (W / "template.html").read_text().replace("/*__DATA__*/", data)
(W / "index.html").write_text(html)
print("web/index.html", round(len(html) / 1e6, 2), "MB")
