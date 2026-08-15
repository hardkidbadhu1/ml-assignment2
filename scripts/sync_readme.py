"""Rewrite the two result tables in README.md from reports/metrics.csv.

Why this exists: the metrics themselves are seeded and reproduce exactly, but the
fit times are wall-clock and change on every run. After any retrain, the README
is therefore stale in a way that is easy to miss, because five of the six columns
still look right. Transcribing 66 numbers by hand is the wrong answer.

The script rewrites, in place, everything between the table header and the blank
line that follows it, for:

  * the comparison table       (ML Model Name | Accuracy | AUC | ...)
  * the diagnostics table      (Model | Train Acc | Test Acc | Gap | ...)

Everything else in the README, including the prose observations, is left alone.
Run it after a retrain, then re-read any sentence that quotes a fit time.

Usage
-----
    python scripts/sync_readme.py            # rewrite in place
    python scripts/sync_readme.py --check    # exit 1 if stale, change nothing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
METRICS = ROOT / "reports" / "metrics.csv"

COMPARISON_HEADER = "| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |"
DIAGNOSTIC_HEADER = "| Model | Train Acc | Test Acc | Gap | CV Acc (mean ± sd) | Fit time (s) |"

# Row order in the report. Fixed rather than taken from the CSV so the table
# always reads in the order the assignment brief lists the models.
ORDER = [
    "Logistic Regression",
    "Decision Tree",
    "kNN",
    "Naive Bayes (Gaussian)",
    "Random Forest (Ensemble)",
    "SVM (RBF)",
]


def comparison_rows(df: pd.DataFrame) -> list[str]:
    """Six metric columns. The winning row is bolded, decided by MCC.

    MCC rather than accuracy because the classes run from 34.6% to 11.5%, and
    accuracy would let a model that ignores Goalkeepers look respectable.
    """
    best = df["MCC"].idxmax()
    rows = []
    for name in ORDER:
        r = df.loc[name]
        vals = [f"{r[c]:.4f}" for c in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]]
        if name == best:
            vals = [f"**{v}**" for v in vals]
        rows.append(f"| {name} | " + " | ".join(vals) + " |")
    return rows


def diagnostic_rows(df: pd.DataFrame) -> list[str]:
    """Train/test/gap/CV/fit-time. Gaps above 0.03 are bolded as the overfitting
    signal the write-up points at."""
    rows = []
    for name in ORDER:
        r = df.loc[name]
        gap = r["Train Accuracy"] - r["Accuracy"]
        gap_s = f"**{gap:.3f}**" if gap > 0.03 else f"{gap:.3f}"
        rows.append(
            f"| {name} | {r['Train Accuracy']:.4f} | {r['Accuracy']:.4f} | {gap_s} | "
            f"{r['CV Accuracy (mean)']:.4f} ± {r['CV Accuracy (std)']:.4f} | "
            f"{r['Fit time (s)']:.2f} |"
        )
    return rows


def replace_table(lines: list[str], header: str, new_rows: list[str]) -> list[str]:
    """Swap the body of the table whose header line matches, leaving the header
    and the separator row untouched."""
    try:
        i = next(n for n, line in enumerate(lines) if line.strip() == header)
    except StopIteration:
        sys.exit(f"ERROR: could not find this table header in README.md:\n  {header}")
    start = i + 2  # skip the header and the |---|---| separator
    end = start
    while end < len(lines) and lines[end].lstrip().startswith("|"):
        end += 1
    return lines[:start] + new_rows + lines[end:]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="report whether the README is stale without rewriting it")
    args = ap.parse_args()

    if not METRICS.exists():
        sys.exit(f"ERROR: {METRICS} not found. Run scripts/train_fifa.sh first.")

    df = pd.read_csv(METRICS, index_col=0)
    missing = [m for m in ORDER if m not in df.index]
    if missing:
        sys.exit(f"ERROR: {missing} missing from metrics.csv. Found: {list(df.index)}")

    original = README.read_text(encoding="utf-8")
    lines = original.splitlines()
    lines = replace_table(lines, COMPARISON_HEADER, comparison_rows(df))
    lines = replace_table(lines, DIAGNOSTIC_HEADER, diagnostic_rows(df))
    updated = "\n".join(lines) + "\n"

    if updated == original:
        print("README tables already match reports/metrics.csv. Nothing to do.")
        return

    if args.check:
        sys.exit(
            "README tables are STALE with respect to reports/metrics.csv.\n"
            "Run: python scripts/sync_readme.py"
        )

    README.write_text(updated, encoding="utf-8")
    best = df["MCC"].idxmax()
    print("Updated the comparison and diagnostics tables in README.md.")
    print(f"  Best by MCC: {best} ({df.loc[best, 'MCC']:.4f})")
    # Built outside the f-string: nested same-type quotes inside an f-string
    # expression only became legal in Python 3.12 (PEP 701), and this should run
    # anywhere.
    times = ", ".join("{} {:.2f}s".format(m.split()[0], df.loc[m, "Fit time (s)"]) for m in ORDER)
    print(f"  Fit times:   {times}")
    print("\nNow re-read any prose that quotes a fit time. Currently that is:")
    print("  * the Naive Bayes row  ('fastest to fit at ...')")
    print("  * the SVM row          ('roughly eleven times the fit cost, ... against ...')")
    print("  * the winner row       ('fits roughly eleven times faster than the SVM')")
    print("  * section b            ('15.72 seconds against 0.19 seconds')")


if __name__ == "__main__":
    main()
