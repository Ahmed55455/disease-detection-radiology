"""Step 2: cost-sensitive decision threshold and the urgency rule (model used in the app).

A) Abnormal-vs-Normal threshold. Score a = 1 - P(Normal). A case is flagged "abnormal" when
   a >= t. Cost = C_FN x FN + C_FP x FP, with C_FP = 1 and C_FN = cost ratio (a missed sick
   patient is assumed to cost more than a false alarm). Several ratios are reported.
B) Urgency rule: urgency = sum(probability x weight), 0 to 1 (config.WEIGHT_SETS).
   HIGH if urgency >= h; h is chosen with the same cost rule (true disease = should be HIGH).
   HIGH/STANDARD is a label for the radiologist and for reporting; the worklist itself is
   ordered by the continuous urgency score.

Only TEST-set probabilities are available, and the test set must not be used to tune and report
at the same time. So the test predictions are split 50/50 (stratified by true class, fixed seed):
thresholds are chosen on the "tuning half" and all reported numbers come from the
"reporting half". (Better: validation-set predictions from Member 1, if they become available.)

Input : data/predictions/<APP_MODEL>_test_probs.csv
Output: analysis/evaluation/results/
          threshold_results.csv, urgency_threshold.csv, urgency_scores.csv, cost_curve.png
Run   : python analysis/evaluation/step2_threshold_urgency.py   (from the repository root)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (APP_MODEL, CLASSES, COST_RATIOS, MAIN_COST_RATIO, MAIN_WEIGHTS, PRED_DIR,
                    PROB_COLS, RESULTS, SEED, WEIGHT_SETS)

GRID = np.round(np.arange(0.01, 1.00, 0.01), 2)


def urgency(df, weights):
    return sum(df[PROB_COLS[c]] * weights[c] for c in CLASSES)


def split_halves(df):
    """Stratified 50/50 split by true class, fixed seed."""
    rng = np.random.default_rng(SEED)
    tune = np.zeros(len(df), dtype=bool)
    for c in CLASSES:
        idx = np.flatnonzero(df["label"].to_numpy() == c)
        rng.shuffle(idx)
        tune[idx[: len(idx) // 2]] = True
    return df[tune], df[~tune]


def confusion(sick, flagged):
    tp = int(np.sum(sick & flagged)); fn = int(np.sum(sick & ~flagged))
    fp = int(np.sum(~sick & flagged)); tn = int(np.sum(~sick & ~flagged))
    return tp, fn, fp, tn


def rates(sick, flagged, ratio):
    tp, fn, fp, tn = confusion(sick, flagged)
    return {"sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp),
            "false_negatives": fn, "false_positives": fp,
            "cost_per_100_cases": 100 * (ratio * fn + fp) / len(sick)}


def best_threshold(score, sick, ratio):
    """Threshold on the grid with the lowest cost (ties: the lower threshold, safer)."""
    costs = [ratio * np.sum(sick & (score < t)) + np.sum(~sick & (score >= t)) for t in GRID]
    return float(GRID[int(np.argmin(costs))]), costs


def main():
    path = PRED_DIR / f"{APP_MODEL}_test_probs.csv"
    if not path.exists():
        sys.exit(f"Missing {path}\nDownload it from Drive (folder 'predictions') into data/predictions/")
    df = pd.read_csv(path)
    df["abnormal_score"] = 1 - df[PROB_COLS["Normal"]]
    for name, w in WEIGHT_SETS.items():
        df[f"urgency [{name}]"] = urgency(df, w)
    tune, report = split_halves(df)
    sick_t, sick_r = tune["label"] != "Normal", report["label"] != "Normal"
    print(f"{APP_MODEL}: {len(df)} test images -> tuning half {len(tune)}, reporting half {len(report)}\n")

    # A) abnormal-vs-normal threshold
    rows, curves = [], {}
    default = rates(sick_r.to_numpy(), (report["pred"] != "Normal").to_numpy(), MAIN_COST_RATIO)
    rows.append({"rule": "argmax (default: predicted class != Normal)", "cost_ratio_fn_fp": MAIN_COST_RATIO,
                 "threshold": None, **default})
    for ratio in COST_RATIOS:
        t, curves[ratio] = best_threshold(tune["abnormal_score"].to_numpy(), sick_t.to_numpy(), ratio)
        r = rates(sick_r.to_numpy(), (report["abnormal_score"] >= t).to_numpy(), ratio)
        rows.append({"rule": "1 - P(Normal) >= threshold", "cost_ratio_fn_fp": ratio, "threshold": t, **r})
    thr = pd.DataFrame(rows)
    thr.to_csv(RESULTS / "threshold_results.csv", index=False, float_format="%.4f")
    print("A) ABNORMAL vs NORMAL threshold (chosen on tuning half, measured on reporting half)")
    print(thr.to_string(index=False), "\n")

    fig, ax = plt.subplots(figsize=(6, 4))
    for ratio, c in curves.items():
        c = np.array(c, dtype=float)
        ax.plot(GRID, 100 * c / len(tune), label=f"FN cost = {ratio} x FP")
    ax.set_xlabel("Threshold on 1 - P(Normal)")
    ax.set_ylabel("Cost per 100 cases (tuning half)")
    ax.set_title("Cost-sensitive threshold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "cost_curve.png", dpi=130)
    plt.close(fig)

    # B) urgency HIGH threshold, for each weight set
    urows = []
    for name in WEIGHT_SETS:
        col = f"urgency [{name}]"
        h, _ = best_threshold(tune[col].to_numpy(), sick_t.to_numpy(), MAIN_COST_RATIO)
        r = rates(sick_r.to_numpy(), (report[col] >= h).to_numpy(), MAIN_COST_RATIO)
        per_class = {f"HIGH share, true {c}": float(np.mean(report.loc[report["label"] == c, col] >= h))
                     for c in CLASSES}
        urows.append({"weights": name, "cost_ratio_fn_fp": MAIN_COST_RATIO, "high_threshold": h,
                      **r, **per_class})
    urg = pd.DataFrame(urows)
    urg.to_csv(RESULTS / "urgency_threshold.csv", index=False, float_format="%.4f")
    print(f"B) URGENCY HIGH threshold (cost ratio {MAIN_COST_RATIO}:1, measured on reporting half)")
    print(urg.to_string(index=False), "\n")

    # Urgency scores for every test image (input of the simulation)
    h_main = float(urg.loc[urg["weights"] == MAIN_WEIGHTS, "high_threshold"].iloc[0])
    out = df[["path", "label", "pred", *PROB_COLS.values()]].copy()
    for name in WEIGHT_SETS:
        out[f"urgency [{name}]"] = df[f"urgency [{name}]"].round(6)
    out["priority (main)"] = np.where(df[f"urgency [{MAIN_WEIGHTS}]"] >= h_main, "HIGH", "STANDARD")
    out["half"] = np.where(df.index.isin(tune.index), "tuning", "reporting")
    out.to_csv(RESULTS / "urgency_scores.csv", index=False)
    print(f"Main rule: weights {WEIGHT_SETS[MAIN_WEIGHTS]}, HIGH if urgency >= {h_main}")
    print(f"Saved results in {RESULTS}")


if __name__ == "__main__":
    main()
