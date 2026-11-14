"""
Descriptive statistics for the OASIS-2 analysis cohort (after exclusions).

    python descriptives_alz.py --csv data/oasis_longitudinal.csv

Prints and saves the numbers quoted in the README: cohort size, mean age at
first visit and share of men in each group (one row per subject).
"""
import argparse
import json
from pathlib import Path

from analysis_alz import load

OUT = Path("results_alz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    df, _ = load(args.csv, include_cognitive=False)
    first = df.sort_values(["Subject ID", "Visit"]).groupby("Subject ID").head(1)

    summary = {"n_visits": int(len(df)), "n_subjects": int(first.shape[0]), "by_group": {}}
    for group in ["Nondemented", "Demented"]:
        d = first[first["Group"] == group]
        summary["by_group"][group] = {
            "n_subjects": int(len(d)),
            "mean_age_first_visit": round(float(d["Age"].mean()), 1),
            "percent_male": round(float(100 * d["Male"].mean()), 1),
            "mean_education_years": round(float(d["EDUC"].mean()), 1),
            "mean_nWBV": round(float(d["nWBV"].mean()), 3),
        }
        r = summary["by_group"][group]
        print(f"{group:12s} n={r['n_subjects']:3d}  mean age {r['mean_age_first_visit']}  "
              f"{r['percent_male']}% male  education {r['mean_education_years']} y  "
              f"nWBV {r['mean_nWBV']}")
    print(f"Cohort: {summary['n_visits']} visits from {summary['n_subjects']} subjects")

    (OUT / "descriptives_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved to {OUT.resolve()}")


if __name__ == "__main__":
    main()
