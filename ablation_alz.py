"""
Feature ablation and sex check (grouped CV by subject).

    python ablation_alz.py --csv data/oasis_longitudinal.csv

Question: which features carry the signal, and is the result driven by the
sex imbalance between groups (more men among demented subjects)?

Part A: logistic regression with different feature sets (grouped CV).
Part B: model trained and tested within each sex separately.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from analysis_alz import REPEATS, SEED, SPLITS, load, model

OUT = Path("results_alz")


def grouped_auc(df, feats):
    X, y, g = df[feats], df["y"].values, df["Subject ID"].values
    aucs = []
    for rep in range(REPEATS):
        sgk = StratifiedGroupKFold(SPLITS, shuffle=True, random_state=SEED + rep)
        for tr, te in sgk.split(X, y, g):
            if len(set(y[te])) < 2:
                continue
            m = model("lr").fit(X.iloc[tr], y[tr])
            aucs.append(roc_auc_score(y[te], m.predict_proba(X.iloc[te])[:, 1]))
    a = np.array(aucs)
    return {"mean": round(float(a.mean()), 3),
            "range95": [round(float(np.percentile(a, 2.5)), 3), round(float(np.percentile(a, 97.5)), 3)],
            "n_folds": int(len(a))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    df, _ = load(args.csv, include_cognitive=False)

    sets = {
        "nWBV only": ["nWBV"],
        "nWBV + eTIV": ["nWBV", "eTIV"],
        "nWBV + eTIV + Age": ["nWBV", "eTIV", "Age"],
        "nWBV + eTIV + Age + EDUC": ["nWBV", "eTIV", "Age", "EDUC"],
        "All without sex": ["Age", "EDUC", "SES", "eTIV", "nWBV"],
        "All with sex": ["Age", "EDUC", "SES", "Male", "eTIV", "nWBV"],
        "Sex only": ["Male"],
    }
    res = {"ablation": {}, "by_sex": {}}
    print("Part A: feature sets (logistic regression, grouped CV)")
    for name, fs in sets.items():
        res["ablation"][name] = grouped_auc(df, fs)
        r = res["ablation"][name]
        print(f"  {name:28s} AUC {r['mean']:.3f}  ({r['range95'][0]:.3f} to {r['range95'][1]:.3f})")

    print("Part B: within each sex (features without sex)")
    fs = ["Age", "EDUC", "SES", "eTIV", "nWBV"]
    for label, val in [("Women", 0), ("Men", 1)]:
        d = df[df["Male"] == val]
        res["by_sex"][label] = {"n_subjects": int(d["Subject ID"].nunique()),
                                "dementia_rate": round(float(d["y"].mean()), 2),
                                **grouped_auc(d, fs)}
        r = res["by_sex"][label]
        print(f"  {label:6s} n={r['n_subjects']} subjects, dementia rate {r['dementia_rate']}, "
              f"AUC {r['mean']:.3f} ({r['range95'][0]:.3f} to {r['range95'][1]:.3f})")

    names = list(res["ablation"])
    means = [res["ablation"][n]["mean"] for n in names]
    lo = [res["ablation"][n]["mean"] - res["ablation"][n]["range95"][0] for n in names]
    hi = [res["ablation"][n]["range95"][1] - res["ablation"][n]["mean"] for n in names]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh(names[::-1], means[::-1], xerr=[lo[::-1], hi[::-1]], color="#4C78A8", capsize=3)
    ax.axvline(0.5, color="grey", linestyle="--")
    ax.set_xlim(0.2, 1.0)
    ax.set_xlabel("ROC-AUC (mean, 95% range over folds)")
    ax.set_title("Which features carry the signal? (grouped CV)")
    fig.tight_layout()
    fig.savefig(OUT / "feature_ablation.png", dpi=200)
    plt.close(fig)
    (OUT / "ablation_summary.json").write_text(json.dumps(res, indent=2))
    print(f"Saved to {OUT.resolve()}")


if __name__ == "__main__":
    main()
