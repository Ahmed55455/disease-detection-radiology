from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

THRESH = 0.98  # our own choice, see the band check below


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - m) / d, (c + m) / d


overlap = pd.read_csv("data/external_overlap.csv")
print("Correlation bands (sanity check for the threshold):")
for lo, hi in [(0.999, 1.01), (0.99, 0.999), (0.98, 0.99), (0.97, 0.98), (0.95, 0.97)]:
    n = ((overlap["corr"] >= lo) & (overlap["corr"] < hi)).sum()
    print(f"  {lo} <= corr < {hi}: {n}")

seen = {Path(p).as_posix() for p in overlap.loc[overlap["corr"] >= THRESH, "external_path"]}
rows = []

for name in ["resnet50", "efficientnet_b0", "densenet121"]:
    r = pd.read_csv(f"data/predictions/external_{name}_probs.csv")
    r["seen"] = r["path"].map(lambda p: Path(p).as_posix() in seen)
    r["y_true"] = (r["label"] == "PNEUMONIA").astype(int)
    r["y_pred"] = (r["pred4"] != "Normal").astype(int)
    r["p_pneu"] = 1.0 - r["p_Normal"]

    if name == "resnet50":
        print("\nComposition:")
        print(pd.crosstab([r["seen"], r["split"]], r["label"]).to_string())

    for subset, sub in [("all", r),
                        ("seen_in_our_data", r[r["seen"]]),
                        ("clean_not_seen", r[~r["seen"]])]:
        tp = int(((sub.y_true == 1) & (sub.y_pred == 1)).sum())
        fn = int(((sub.y_true == 1) & (sub.y_pred == 0)).sum())
        tn = int(((sub.y_true == 0) & (sub.y_pred == 0)).sum())
        fp = int(((sub.y_true == 0) & (sub.y_pred == 1)).sum())
        s_lo, s_hi = wilson(tp, tp + fn)
        c_lo, c_hi = wilson(tn, tn + fp)
        both = sub.y_true.nunique() == 2
        rows.append({
            "model": name, "subset": subset, "n": len(sub),
            "n_normal": tn + fp, "n_pneumonia": tp + fn,
            "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
            "sens_95ci": f"{s_lo:.3f}-{s_hi:.3f}",
            "specificity": tn / (tn + fp) if tn + fp else np.nan,
            "spec_95ci": f"{c_lo:.3f}-{c_hi:.3f}",
            "auc": roc_auc_score(sub.y_true, sub.p_pneu) if both else np.nan,
        })

out = pd.DataFrame(rows)
out.to_csv("analysis/external_clean_generalization.csv", index=False)
print()
print(out.round(3).to_string(index=False))

# What does the model call the clean pneumonia images?
for name in ["resnet50", "efficientnet_b0", "densenet121"]:
    r = pd.read_csv(f"data/predictions/external_{name}_probs.csv")
    r["seen"] = r["path"].map(lambda p: Path(p).as_posix() in seen)
    clean = r[~r["seen"]]
    print(f"\n{name}: 4-class predictions on clean images")
    print(pd.crosstab(clean["label"], clean["pred4"]).to_string())