"""Build the submission PDF.

The brief fixes the order of the document:

    1. GitHub repository link
    2. Live Streamlit app link
    3. BITS Virtual Lab screenshot
    4. The full README content

All four are rendered in a single WeasyPrint pass rather than generating pieces
and stitching them with pypdf. One pass means one set of page numbers, one style,
and no chance of the parts drifting out of order.

The links in section 1-2 are *parsed out of README.md* rather than restated here,
so the cover page cannot disagree with the body of the document.

Markdown -> HTML (python-markdown) -> PDF (WeasyPrint). Most of the CSS exists to
handle the observations table, whose cells hold several sentences each:
`table-layout: fixed` plus explicit column widths stops one long cell from blowing
the table off the page edge.

Usage
-----
    python scripts/make_pdf.py                      # full submission
    python scripts/make_pdf.py --readme-only        # body only, no cover/evidence
    python scripts/make_pdf.py --screenshots docs/  # screenshots from elsewhere
"""

from __future__ import annotations

import argparse
import base64
import mimetypes
import re
import sys
from pathlib import Path

import markdown
from weasyprint import CSS, HTML

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "README.md"
OUT = ROOT / "reports" / "ML_Assignment2_Badhmanaban_M_2025AC05386.pdf"
LAB_SHOTS = ROOT / "screenshots" / "lab"
APP_SHOTS = ROOT / "screenshots" / "app"

NAME = "Badhmanaban M"
BITS_ID = "2025AC05386"
COURSE = "M.Tech (AIML/DSE), Machine Learning, Assignment 2"

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp"}

CSS_TEXT = """
/* Plain black-and-white print styling. No colour, no fills, no rounded corners:
   a printed report should look like a document, not like a dashboard export.
   Hairline rules and whitespace do the separating instead. */
@page {
    size: A4;
    margin: 20mm 18mm 20mm 18mm;
    @bottom-center {
        content: counter(page);
        font-family: "DejaVu Serif", serif;
        font-size: 9pt;
    }
}
body {
    font-family: "DejaVu Serif", serif;
    font-size: 10pt;
    line-height: 1.5;
    color: #000;
}
h1 {
    font-size: 16pt;
    font-weight: normal;
    margin: 0 0 2px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid #000;
}
h2 {
    font-size: 12pt;
    margin-top: 20px;
    margin-bottom: 6px;
    break-after: avoid;   /* never strand a heading at the foot of a page */
}
h3 { font-size: 10.5pt; font-style: italic; font-weight: normal;
     margin-top: 14px; margin-bottom: 4px; break-after: avoid; }
p { margin: 6px 0; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 8.4pt; }
pre {
    font-family: "DejaVu Sans Mono", monospace;
    font-size: 8pt;
    line-height: 1.4;
    border-left: 2px solid #000;
    padding: 4px 0 4px 10px;
    margin: 8px 0;
    white-space: pre-wrap;
    word-wrap: break-word;
    break-inside: avoid;
}
blockquote {
    margin: 8px 0 8px 14px;
    padding-left: 12px;
    border-left: 1px solid #000;
    font-size: 9.4pt;
}
blockquote p { margin: 3px 0; }

/* Tables: horizontal rules only, in the style of a typeset table. Vertical
   lines and shaded cells are what make a table look machine-produced. */
table {
    border-collapse: collapse;
    width: 100%;
    margin: 10px 0;
    font-size: 8.6pt;
    table-layout: fixed;
    break-inside: auto;
}
th, td {
    border: none;
    border-bottom: 0.5px solid #999;
    padding: 5px 6px 5px 0;
    text-align: left;
    vertical-align: top;
    word-wrap: break-word;
    overflow-wrap: break-word;
}
thead th { border-bottom: 1px solid #000; font-weight: bold; }
thead { display: table-header-group; }   /* repeat header when a table splits */
table.obs th:first-child, table.obs td:first-child { width: 18%; }
table.obs { font-size: 8.4pt; }
table.metrics td:not(:first-child), table.metrics th:not(:first-child) { text-align: right; }
table.metrics th:first-child, table.metrics td:first-child { width: 23%; }

ul, ol { margin: 6px 0 6px 18px; padding-left: 4px; }
li { margin: 4px 0; }
hr { border: none; border-top: 0.5px solid #999; margin: 16px 0; }
a { color: #000; text-decoration: underline; }

/* ---- Cover ---- */
.cover { page-break-after: always; }
.cover h1 { margin-top: 60mm; font-size: 18pt; border-bottom: 1px solid #000; }
.cover .sub { font-size: 10.5pt; margin: 8px 0 4px 0; }
.cover .who { font-size: 10.5pt; margin: 0 0 34px 0; }
.linkrow { margin: 14px 0; }
.linkrow .label { font-size: 9pt; font-style: italic; margin-bottom: 2px; }
.linkrow a {
    font-family: "DejaVu Sans Mono", monospace;
    font-size: 9.2pt;
    word-break: break-all;   /* long URLs wrap instead of running off the page */
}

/* ---- Figures ---- */
.shot { margin: 12px 0; break-inside: avoid; text-align: center; }
.shot img {
    max-width: 100%;
    /* Cap the height as well: a tall terminal capture scaled to full page width
       would otherwise overflow onto a blank following page. */
    max-height: 200mm;
    border: 0.5px solid #999;
}
.shot .cap { font-size: 8.4pt; font-style: italic; margin-top: 5px; text-align: center; }
.figs { page-break-after: always; }
"""


# --------------------------------------------------------------------------- #
# Cover + evidence
# --------------------------------------------------------------------------- #
def extract_links(md_text: str) -> dict[str, str]:
    """Pull the repository and app URLs out of README section c.

    Parsed rather than hard-coded, so the cover page and the body of the report
    cannot disagree. The README stays the single source of truth for both.
    """
    links: dict[str, str] = {}
    for key, label in [("repo", "Repository"), ("app", "Live Streamlit app")]:
        m = re.search(rf"\*\*{re.escape(label)}:\*\*\s*<?(https?://[^\s<>`)]+)>?", md_text)
        if m:
            links[key] = m.group(1).rstrip(">")
    return links


def data_uri(path: Path) -> str:
    """Inline an image as a data URI.

    WeasyPrint can load images off disk through base_url, but inlining makes the
    finished PDF independent of the directory it was built in. That matters
    because this file gets copied around at submission time.
    """
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def caption(path: Path, index: int) -> str:
    """Turn '02-confusion-matrix.png' into 'Figure 2. Confusion matrix'.

    The numeric prefix exists to order the files, so it should not survive into
    the printed caption.
    """
    stem = re.sub(r"^\d+[-_]", "", path.stem).replace("-", " ").replace("_", " ").strip()
    return f"Figure {index}. {stem[:1].upper() + stem[1:]}" if stem else f"Figure {index}."


def figures(shots: list[Path], start: int = 1) -> str:
    return "".join(
        f'<div class="shot"><img src="{data_uri(p)}">'
        f'<div class="cap">{caption(p, start + i)}</div></div>'
        for i, p in enumerate(shots)
    )


def build_cover(links: dict[str, str]) -> str:
    rows = "".join(
        f'<div class="linkrow"><div class="label">{n}. {label}</div>'
        f'<a href="{links[k]}">{links[k]}</a></div>'
        for n, (k, label) in enumerate(
            [("repo", "GitHub repository"), ("app", "Live Streamlit application")], start=1
        )
        if k in links
    )
    return (
        '<div class="cover">'
        "<h1>FIFA World Cup 2026 Player Performance:<br>a comparison of six classifiers</h1>"
        f'<div class="sub">{COURSE}</div>'
        f'<div class="who">{NAME} ({BITS_ID})</div>'
        f"{rows}"
        "</div>"
    )


def build_lab_section(shots: list[Path]) -> str:
    if not shots:
        return ""
    return (
        '<div class="figs">'
        "<h2>3. Evidence of execution on BITS Virtual Lab</h2>"
        "<p>Every model was trained and evaluated on the BITS Virtual Lab virtual "
        "machine, running Rocky Linux 9.5 with Python 3.12.7 and scikit-learn 1.7.2. "
        "The capture below is a single frame from the end of that run, and shows:</p>"
        "<ul>"
        "<li>the host name and the EC2 instance id of the lab machine, which is what "
        "makes this evidence rather than decoration, since neither can be produced "
        "from a personal laptop;</li>"
        "<li>all six models with all six evaluation metrics, exactly as reported in "
        "section (d) below;</li>"
        "<li>the full set of verification checks passing, covering data leakage, "
        "artifact integrity and deployability.</li>"
        "</ul>"
        f"{figures(shots)}</div>"
    )


def build_app_section(shots: list[Path], start: int) -> str:
    if not shots:
        return ""
    return (
        '<div class="figs">'
        "<h2>4. The deployed Streamlit application</h2>"
        "<p>The application is live on Streamlit Community Cloud at the address "
        "given on the cover page. The screenshots that follow show each feature "
        "the brief asks for:</p>"
        "<ul>"
        "<li>a file upload control for supplying a test set as CSV;</li>"
        "<li>a dropdown for choosing between the six trained models;</li>"
        "<li>the six evaluation metrics for whichever model is selected;</li>"
        "<li>a confusion matrix together with a per-class classification report;</li>"
        "<li>a table scoring all six models on the same test data at once.</li>"
        "</ul>"
        f"{figures(shots, start)}</div>"
    )


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


def collect_screenshots(folder: Path) -> list[Path]:
    """Every image in `folder`, in filename order.

    Sorted so the ordering is deterministic and controllable: prefix files
    `01-`, `02-` if you want a specific sequence.
    """
    if not folder.is_dir():
        return []
    return sorted(
        (p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file()),
        key=lambda p: p.name.lower(),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lab-shots", type=Path, default=LAB_SHOTS,
                    help="folder holding the BITS Virtual Lab capture(s)")
    ap.add_argument("--app-shots", type=Path, default=APP_SHOTS,
                    help="folder holding the Streamlit app screenshots")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--readme-only", action="store_true",
                    help="render just the README body, with no cover page or figures")
    ap.add_argument("--allow-missing-screenshots", action="store_true",
                    help="build even if a screenshot folder is empty")
    args = ap.parse_args()

    md_text = SRC.read_text(encoding="utf-8")
    links = extract_links(md_text)
    lab = [] if args.readme_only else collect_screenshots(args.lab_shots)
    app = [] if args.readme_only else collect_screenshots(args.app_shots)

    if not args.readme_only:
        missing = [k for k in ("repo", "app") if k not in links]
        if missing:
            sys.exit(
                f"ERROR: could not find the {missing} link(s) in README section c.\n"
                "Expected lines of this form:\n"
                "  - **Repository:** <https://github.com/user/repo>\n"
                "  - **Live Streamlit app:** <https://app.streamlit.app/>"
            )
        empty = [d for d, s in [(args.lab_shots, lab), (args.app_shots, app)] if not s]
        if empty and not args.allow_missing_screenshots:
            names = ", ".join(str(d) for d in empty)
            sys.exit(
                f"ERROR: no images found in: {names}\n\n"
                "The report is meant to carry the lab evidence and the app screenshots,\n"
                "so a build without them would look complete while quietly dropping marks.\n\n"
                "  screenshots/lab/   one frame from the end of scripts/lab_run.sh\n"
                "  screenshots/app/   one capture per tab of the deployed app\n\n"
                "Images are embedded in filename order, so prefix them 01-, 02- to\n"
                "control the sequence. See screenshots/README.md for what to capture.\n"
                "To build anyway, pass --allow-missing-screenshots."
            )

    body = classify_tables(
        markdown.markdown(md_text, extensions=["tables", "fenced_code", "sane_lists", "attr_list"])
    )
    parts = "" if args.readme_only else (
        build_cover(links) + build_lab_section(lab) + build_app_section(app, len(lab) + 1)
    )
    html = f"<html><head><meta charset='utf-8'></head><body>{parts}{body}</body></html>"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=html, base_url=str(ROOT)).write_pdf(args.out, stylesheets=[CSS(string=CSS_TEXT)])

    try:
        shown = args.out.resolve().relative_to(ROOT)
    except ValueError:          # --out pointed somewhere outside the repository
        shown = args.out
    print(f"Wrote {shown} ({args.out.stat().st_size / 1024:.0f} KB)")
    if args.readme_only:
        print("  README body only (--readme-only)")
        return
    print(f"  1. GitHub repository    {links['repo']}")
    print(f"  2. Live Streamlit app   {links['app']}")
    print(f"  3. Lab evidence         {len(lab)} image(s): "
          f"{', '.join(p.name for p in lab) or 'NONE'}")
    print(f"  4. App screenshots      {len(app)} image(s): "
          f"{', '.join(p.name for p in app) or 'NONE'}")
    print("  5. Report body from README.md")


if __name__ == "__main__":
    main()
