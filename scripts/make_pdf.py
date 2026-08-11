"""Render README.md to the submission PDF.

The brief requires the README content to also be part of the submitted PDF, so
this renders from the same source file rather than maintaining a second copy that
would drift.

Markdown -> HTML (python-markdown) -> PDF (WeasyPrint). The CSS below exists
mostly to handle the observations table, whose cells hold several sentences each:
`table-layout: fixed` plus explicit column widths stops the renderer from letting
one long cell blow the table off the page edge.

Usage:  python scripts/make_pdf.py
"""

from __future__ import annotations

from pathlib import Path

import markdown
from weasyprint import CSS, HTML

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "README.md"
OUT = ROOT / "reports" / "ML_Assignment2_Badhmanaban_M_2025AC05386.pdf"

CSS_TEXT = """
@page {
    size: A4;
    margin: 16mm 14mm 18mm 14mm;
    @bottom-center {
        content: counter(page) " / " counter(pages);
        font-family: "DejaVu Sans", sans-serif;
        font-size: 8pt;
        color: #888;
    }
}
body {
    font-family: "DejaVu Sans", sans-serif;
    font-size: 9.2pt;
    line-height: 1.45;
    color: #1a1a1a;
}
h1 {
    font-size: 17pt;
    color: #0f3d64;
    border-bottom: 2.5px solid #0f3d64;
    padding-bottom: 5px;
    margin-bottom: 4px;
}
h2 {
    font-size: 12.5pt;
    color: #0f3d64;
    margin-top: 16px;
    margin-bottom: 6px;
    border-bottom: 1px solid #c8d6e2;
    padding-bottom: 3px;
    /* Keep a required section heading from stranding at the foot of a page. */
    break-after: avoid;
}
h3 { font-size: 10.5pt; color: #24557d; margin-top: 12px; margin-bottom: 4px; break-after: avoid; }
p { margin: 5px 0; text-align: justify; }
code {
    font-family: "DejaVu Sans Mono", monospace;
    font-size: 8pt;
    background: #f0f3f6;
    padding: 1px 3px;
    border-radius: 2px;
}
pre {
    background: #f7f9fb;
    border: 1px solid #dde5ec;
    border-left: 3px solid #0f3d64;
    padding: 7px 9px;
    font-size: 7.6pt;
    line-height: 1.35;
    white-space: pre-wrap;
    word-wrap: break-word;
    break-inside: avoid;
}
pre code { background: none; padding: 0; font-size: 7.6pt; }
blockquote {
    border-left: 3px solid #f0ad4e;
    background: #fffaf2;
    margin: 8px 0;
    padding: 5px 10px;
    font-size: 8.6pt;
}
blockquote p { margin: 3px 0; }
table {
    border-collapse: collapse;
    width: 100%;
    margin: 8px 0;
    font-size: 7.9pt;
    table-layout: fixed;
}
th, td {
    border: 1px solid #c2cfda;
    padding: 4px 5px;
    text-align: left;
    vertical-align: top;
    word-wrap: break-word;
    overflow-wrap: break-word;
    hyphens: auto;
}
th { background: #0f3d64; color: #fff; font-weight: bold; }
/* An inline `code` chip inherits a pale background, which is invisible against
   the dark header fill. Invert it inside th only. */
th code { background: rgba(255, 255, 255, 0.20); color: #fff; }
tr:nth-child(even) td { background: #f5f8fa; }
/* The observations table: one narrow label column, one very wide prose column. */
table.obs th:first-child, table.obs td:first-child { width: 17%; }
table.obs { font-size: 7.6pt; }
/* Metric tables: numerics right-aligned so the decimal points line up, and a
   first column wide enough that "Random Forest (Ensemble)" stays on one line —
   this is the table the assignment is marked on, so it should read cleanly and
   survive a copy-paste without the label splitting across rows. */
table.metrics td:not(:first-child), table.metrics th:not(:first-child) {
    text-align: right;
}
table.metrics th:first-child, table.metrics td:first-child { width: 23%; }
ul, ol { margin: 5px 0 5px 16px; padding-left: 6px; }
li { margin: 2.5px 0; text-align: justify; }
strong { color: #0a2a45; }
hr { border: none; border-top: 1px solid #dde5ec; margin: 14px 0; }
"""


def classify_tables(html: str) -> str:
    """Tag tables so the CSS can size them appropriately.

    python-markdown emits bare <table>; the observations table needs a fixed
    narrow first column while the metric tables want right-aligned numerics.
    Detect by header text rather than position, so reordering the README does
    not silently break the layout.
    """
    out: list[str] = []
    for chunk in html.split("<table>"):
        if not out:
            out.append(chunk)
            continue
        head = chunk[:400]
        if "Observation about model performance" in head:
            cls = "obs"
        elif "MCC" in head or "CV Acc" in head or "Accuracy" in head:
            cls = "metrics"
        else:
            cls = "plain"
        out.append(f'<table class="{cls}">{chunk}')
    return "".join(out)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    body = markdown.markdown(
        SRC.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "sane_lists", "attr_list"],
    )
    body = classify_tables(body)
    html = f"<html><head><meta charset='utf-8'></head><body>{body}</body></html>"
    HTML(string=html, base_url=str(ROOT)).write_pdf(OUT, stylesheets=[CSS(string=CSS_TEXT)])
    print(f"Wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
