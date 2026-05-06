#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MD = ROOT / "analytics" / "reports" / "case_study.md"
DOCS = ROOT / "docs" / "report.html"


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def md_to_html(md: str) -> str:
    o = []
    for block in md.split("\n\n"):
        b = block.strip()
        if not b:
            continue
        if b.startswith("# "):
            o.append(f"<h1>{esc(b[2:])}</h1>")
        elif b.startswith("## "):
            o.append(f"<h2>{esc(b[3:])}</h2>")
        elif b.startswith("- "):
            items = "".join(f"<li>{esc(l[2:])}</li>" for l in b.split("\n") if l.startswith("- "))
            o.append("<ul>" + items + "</ul>")
        else:
            o.append("<p>" + esc(b) + "</p>")
    return "\n".join(o)


def main() -> None:
    body = md_to_html(MD.read_text(encoding="utf-8"))
    gal = "<h2>Artifacts</h2><ul>"
    for fn in ("metrics.json", "classification_report.txt", "confusion_matrix.png", "errors_by_lang.csv", "error_sample.csv"):
        gal += f"<li><a href=\"../modeling/artifacts/{fn}\">{esc(fn)}</a></li>"
    gal += "</ul>"
    DOCS.parent.mkdir(parents=True, exist_ok=True)
    DOCS.write_text(
        f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"/>
<title>Fact-check baseline report</title>
<style>body{{font-family:system-ui;max-width:780px;margin:2rem auto;padding:0 1rem}} a{{color:#4338ca}}</style></head><body>
<p><a href="index.html">← Hub</a></p>
{body}{gal}
</body></html>""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
