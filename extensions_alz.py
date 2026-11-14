"""
Extensions for the OASIS-2 dementia analysis. Run AFTER analysis_alz.py works.

    python extensions_alz.py --csv data/oasis_longitudinal.csv

1. First-visit-only sensitivity analysis: one row per person, so no repeated
   measures at all. Do the conclusions hold?
2. Calibration: are predicted probabilities trustworthy? (out-of-fold, grouped)
3. Paired comparison: does the full model beat the age-only model on the SAME
   folds? Reports the per-fold AUC difference with a bootstrap 95% interval.
   (Folds are not fully independent, so treat the interval as indicative.)
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from analysis_alz import REPEATS, SEED, SPLITS, load, model

OUT = Path("results_alz")


def first_visit(df, feats):
    d = df.sort_values(["Subject ID", "Visit"]).groupby("Subject ID").head(1)
    X, y = d[feats], d["y"].values
    aucs = []
    for rep in range(REPEATS):
        for tr, te in StratifiedKFold(SPLITS, shuffle=True, random_state=SEED + rep).split(X, y):
            m = model("lr").fit(X.iloc[tr], y[tr])
            aucs.append(roc_auc_score(y[te], m.predict_proba(X.iloc[te])[:, 1]))
    return np.array(aucs), len(d)


def grouped_oof(df, feats):
    """Out-of-fold probabilities plus per-fold AUCs for full and age-only models."""
    X, y, g = df[feats], df["y"].values, df["Subject ID"].values
    oof = np.zeros(len(df))
    full, age = [], []
    for rep in range(REPEATS):
        sgk = StratifiedGroupKFold(SPLITS, shuffle=True, random_state=SEED + rep)
        for tr, te in sgk.split(X, y, g):
            m_full = model("lr").fit(X.iloc[tr], y[tr])
            m_age = model("lr").fit(X.iloc[tr][["Age"]], y[tr])
            p = m_full.predict_proba(X.iloc[te])[:, 1]
            full.append(roc_auc_score(y[te], p))
            age.append(roc_auc_score(y[te], m_age.predict_proba(X.iloc[te][["Age"]])[:, 1]))
            if rep == 0:
                oof[te] = p
    return oof, np.array(full), np.array(age)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    df, feats = load(args.csv, include_cognitive=False)
    summary = {}

    # 1. First-visit sensitivity
    a, n = first_visit(df, feats)
    summary["first_visit_only"] = {
        "n_subjects": int(n), "auc_mean": round(float(a.mean()), 3),
        "range95": [round(float(np.percentile(a, 2.5)), 3), round(float(np.percentile(a, 97.5)), 3)]}
    print("First-visit-only LR AUC:", summary["first_visit_only"])

    # 2 and 3. Grouped out-of-fold predictions
    oof, full, age = grouped_oof(df, feats)
    y = df["y"].values
    frac_pos, mean_pred = calibration_curve(y, oof, n_bins=8, strategy="quantile")
    summary["brier_score_full_lr"] = round(float(brier_score_loss(y, oof)), 4)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    ax.plot(mean_pred, frac_pos, "o-", color="#4C78A8", label="Logistic regression")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraction with dementia")
    ax.set_title(f"Calibration (Brier = {summary['brier_score_full_lr']})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "calibration.png", dpi=200)
    plt.close(fig)

    diff = full - age
    rng = np.random.default_rng(SEED)
    boots = [rng.choice(diff, len(diff)).mean() for _ in range(5000)]
    summary["full_vs_age_only"] = {
        "mean_auc_difference": round(float(diff.mean()), 3),
        "bootstrap_ci95": [round(float(np.percentile(boots, 2.5)), 3),
                           round(float(np.percentile(boots, 97.5)), 3)],
        "share_of_folds_full_better": round(float((diff > 0).mean()), 2)}
    print("Full vs age-only:", summary["full_vs_age_only"])

    (OUT / "extensions_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved to {OUT.resolve()}")


if __name__ == "__main__":
    main()
