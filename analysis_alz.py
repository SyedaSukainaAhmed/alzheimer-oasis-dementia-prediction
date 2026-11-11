"""
Leakage-aware prediction of dementia from demographic and MRI-derived measures
(OASIS-2 longitudinal data).

Expected input: oasis_longitudinal.csv with columns
    Subject ID, Group, Visit, M/F, Age, EDUC, SES, MMSE, CDR, eTIV, nWBV, ASF

    python analysis_alz.py --csv data/oasis_longitudinal.csv

Design choices (each one avoids a common mistake):
  * Subjects have several visits. Rows from the same person must never be in
    both train and test, so cross-validation is grouped by Subject ID.
    The script also runs naive row-wise CV to show how much it inflates the score.
  * CDR and MMSE are excluded by default. CDR is the clinical rating used to
    define dementia, so using it to predict dementia is circular.
  * ASF is excluded because it is a deterministic function of eTIV (redundant).
  * 'Converted' subjects (non-demented at first, demented later) are removed so
    the label is unambiguous.
  * Missing values (SES) are imputed inside each fold, not before splitting.
  * An age-only baseline shows whether brain volume (nWBV) adds information
    beyond age.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
OUT = Path("results_alz")
REPEATS = 5
SPLITS = 5


def load(path, include_cognitive):
    df = pd.read_csv(path)
    df = df[df["Group"].isin(["Demented", "Nondemented"])].copy()
    df["y"] = (df["Group"] == "Demented").astype(int)
    df["Male"] = (df["M/F"] == "M").astype(int)
    feats = ["Age", "EDUC", "SES", "Male", "eTIV", "nWBV"]
    if include_cognitive:
        feats += ["MMSE", "CDR"]
    return df, feats


def model(kind):
    imp = ("impute", SimpleImputer(strategy="median"))
    if kind == "lr":
        return Pipeline([imp, ("scale", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=5000, random_state=SEED))])
    return Pipeline([imp, ("clf", RandomForestClassifier(
        n_estimators=300, min_samples_leaf=3, random_state=SEED, n_jobs=-1))])


def run_cv(df, feats, kind, grouped=True):
    X, y, g = df[feats], df["y"].values, df["Subject ID"].values
    aucs, perms = [], []
    for rep in range(REPEATS):
        if grouped:
            splitter = StratifiedGroupKFold(SPLITS, shuffle=True, random_state=SEED + rep)
            splits = splitter.split(X, y, g)
        else:
            splitter = StratifiedKFold(SPLITS, shuffle=True, random_state=SEED + rep)
            splits = splitter.split(X, y)
        for tr, te in splits:
            m = model(kind).fit(X.iloc[tr], y[tr])
            aucs.append(roc_auc_score(y[te], m.predict_proba(X.iloc[te])[:, 1]))
            if grouped and kind == "rf" and len(feats) > 1:
                r = permutation_importance(m, X.iloc[te], y[te], scoring="roc_auc",
                                           n_repeats=5, random_state=SEED, n_jobs=-1)
                perms.append(r.importances_mean)
    return np.array(aucs), perms


def summarise(a):
    return {"mean": round(float(a.mean()), 3),
            "ci95_low": round(float(np.percentile(a, 2.5)), 3),
            "ci95_high": round(float(np.percentile(a, 97.5)), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--include-cognitive", action="store_true",
                    help="Also use MMSE and CDR (circular for dementia labels; for comparison only)")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    df, feats = load(args.csv, args.include_cognitive)
    print(f"{len(df)} visits from {df['Subject ID'].nunique()} subjects; "
          f"dementia rate {df['y'].mean():.2f}; features: {feats}")

    summary = {"n_visits": int(len(df)), "n_subjects": int(df["Subject ID"].nunique()),
               "features": feats, "results": {}}
    rows = {}
    for label, kind, fs, grouped in [
        ("Age only (LR), grouped CV", "lr", ["Age"], True),
        ("All features (LR), grouped CV", "lr", feats, True),
        ("All features (RF), grouped CV", "rf", feats, True),
        ("All features (RF), NAIVE row-wise CV", "rf", feats, False),
    ]:
        a, p = run_cv(df, fs, kind, grouped)
        rows[label] = a
        summary["results"][label] = summarise(a)
        print(f"{label:42s} AUC {a.mean():.3f}  (95% range {np.percentile(a,2.5):.3f} to {np.percentile(a,97.5):.3f})")
        if label.startswith("All features (RF), grouped"):
            perms = p

    # Figure 1: model comparison
    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(rows)
    means = [rows[n].mean() for n in names]
    err = [[rows[n].mean() - np.percentile(rows[n], 2.5) for n in names],
           [np.percentile(rows[n], 97.5) - rows[n].mean() for n in names]]
    ax.barh(names[::-1], means[::-1], xerr=[e[::-1] for e in err],
            color=["#E15759", "#59A14F", "#4C78A8", "#9C9C9C"][::-1], capsize=4)
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel("ROC-AUC (mean, 95% range over folds)")
    ax.set_title("Dementia prediction: effect of grouping and of age alone")
    fig.tight_layout()
    fig.savefig(OUT / "model_comparison.png", dpi=200)
    plt.close(fig)

    # Figure 2: permutation importance (grouped RF)
    arr = np.vstack(perms)
    mean, sd = arr.mean(0), arr.std(0)
    order = np.argsort(mean)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.barh([feats[i] for i in order], mean[order], xerr=sd[order], color="#59A14F")
    ax.set_xlabel("Drop in ROC-AUC when shuffled (held-out subjects)")
    ax.set_title("Permutation importance, Random Forest")
    fig.tight_layout()
    fig.savefig(OUT / "feature_importance.png", dpi=200)
    plt.close(fig)
    summary["permutation_importance"] = {feats[i]: round(float(mean[i]), 4) for i in order[::-1]}
    print("Permutation importance:", summary["permutation_importance"])

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved to {OUT.resolve()}")


if __name__ == "__main__":
    main()
