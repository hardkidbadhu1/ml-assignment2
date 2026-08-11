"""
Generate a throwaway mixed-type dataset for local smoke-testing only.

Purpose: verify the train -> artifact -> Streamlit wiring end to end before your
real dataset is in place, and after any change to `ml_pipeline.py`. It exercises
the awkward paths — NaNs in a numeric column, a boolean feature, a low-cardinality
integer feature that must be treated as categorical, and heavy class imbalance.

    python scripts/make_synthetic_smoke_data.py
    python model/train.py --data data/_smoke.csv --target converted --skip-cv

This is NOT the assignment dataset. Do not submit results from it.
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "data" / "_smoke.csv"


def main(n: int = 1500, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "duration": rng.gamma(2, 30, n).round(2),
            "page_views": rng.poisson(8, n),
            "bounce_rate": rng.beta(2, 5, n).round(4),
            "exit_rate": rng.beta(2, 6, n).round(4),
            "page_value": rng.exponential(5, n).round(3),
            "special_day": rng.choice([0.0, 0.2, 0.4, 0.6], n),
            "account_age_days": rng.integers(1, 2000, n),
            "sessions": rng.integers(1, 40, n),
            "avg_cart_value": rng.normal(120, 45, n).round(2),
            "discount_pct": rng.integers(0, 5, n),          # low-card int -> categorical
            "month": rng.choice(["Jan", "Feb", "Mar", "Nov", "Dec"], n),
            "visitor_type": rng.choice(["New", "Returning", "Other"], n, p=[0.25, 0.70, 0.05]),
            "region": rng.choice(list("ABCDEFGH"), n),
            "weekend": rng.choice([True, False], n),        # bool -> categorical
        }
    )
    df.loc[rng.choice(n, 40, replace=False), "avg_cart_value"] = np.nan  # exercise the imputer

    logit = (
        0.2 * df.page_value
        + 0.03 * df.page_views
        - 6 * df.bounce_rate
        + 0.4 * (df.visitor_type == "Returning")
        + 0.5 * df.weekend
        - 3.0
    )
    df["converted"] = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "yes", "no")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {OUT} — {df.shape[0]} rows, {df.shape[1] - 1} features")
    print(df["converted"].value_counts(normalize=True).round(3).to_string())


if __name__ == "__main__":
    main()
