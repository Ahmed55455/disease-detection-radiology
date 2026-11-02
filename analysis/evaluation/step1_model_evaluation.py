"""Step 1: statistical evaluation and paired comparison of the 3 models (same 3,167 test images).

Per model : confusion matrix, per-class precision / recall / F1 / ROC-AUC (one-vs-rest),
            accuracy, macro-F1, macro-recall, with 95% bootstrap confidence intervals.
Per pair  : exact McNemar test on right/wrong per image (paired), Bonferroni correction for
            3 comparisons, and the 95% bootstrap CI of the macro-F1 difference.

Input : data/predictions/<model>_test_probs.csv (Drive, folder "predictions")
Output: analysis/evaluation/results/
          model_summary.csv, per_class_metrics.csv, pairwise_comparison.csv,
          confusion_<model>.png, roc_<model>.png
Run   : python analysis/evaluation/step1_model_evaluation.py   (from the repository root)

These results describe the seed-42 checkpoints only (docs/model_findings.md, section 3.2).
"""
import math
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, f1_score, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (ALPHA, CLASSES, MODELS, N_BOOT, PRED_DIR, PROB_COLS, RESULTS, SEED)


def load_predictions(model):
    path = PRED_DIR / f"{model}_test_probs.csv"
    if not path.exists():
        sys.exit(f"Missing {path}\nDownload it from Drive (folder 'predictions') into data/predictions/")
    df = pd.read_csv(path)
    missing = {"path", "label", "pred", *PROB_COLS.values()} - set(df.columns)
    if missing:
        sys.exit(f"{path.name}: missing columns {missing}")
    return df.set_index("path")


def summary_scores(y, p):
    """y, p: integer class codes (0-3). Fast version for the bootstrap."""
    cm = np.bincount(y * 4 + p, minlength=16).reshape(4, 4)
    tp = np.diag(cm).astype(float)
    rec = np.divide(tp, cm.sum(1), out=np.zeros(4), where=cm.sum(1) > 0)
    prec = np.divide(tp, cm.sum(0), out=np.zeros(4), where=cm.sum(0) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(4), where=(prec + rec) > 0)
    return tp.sum() / len(y), float(f1.mean()), float(rec.mean()), rec


def ci(values):
    lo, hi = np.percentile(values, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return float(lo), float(hi)


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value (b = A right & B wrong, c = A wrong & B right)."""
    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def plot_confusion(model, y, p):
    cm = confusion_matrix(y, p, labels=CLASSES)
    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(4), [c.replace("_", " ") for c in CLASSES], rotation=30, ha="right")
    ax.set_yticks(range(4), [c.replace("_", " ") for c in CLASSES])
    for i in range(4):
        for j in range(4):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix: {model}")
    fig.tight_layout()
    fig.savefig(RESULTS / f"confusion_{model}.png", dpi=130)
    plt.close(fig)


def plot_roc(model, df):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for c in CLASSES:
        fpr, tpr, _ = roc_curve(df["label"] == c, df[PROB_COLS[c]])
        auc = roc_auc_score(df["label"] == c, df[PROB_COLS[c]])
        ax.plot(fpr, tpr, label=f"{c.replace('_', ' ')} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC one-vs-rest: {model}")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(RESULTS / f"roc_{model}.png", dpi=130)
    plt.close(fig)


def main():
    data = {m: load_predictions(m) for m in MODELS}
    common = sorted(set.intersection(*(set(d.index) for d in data.values())))
    data = {m: d.loc[common] for m, d in data.items()}
    y = data[MODELS[0]]["label"].to_numpy()
    n = len(y)
    print(f"{n} test images common to the 3 models\n")

    rng = np.random.default_rng(SEED)
    boots = [rng.integers(0, n, n) for _ in range(N_BOOT)]

    summary, per_class, boot_f1 = [], [], {}
    code = {c: k for k, c in enumerate(CLASSES)}
    yc = np.array([code[v] for v in y])
    for m in MODELS:
        df, p = data[m], data[m]["pred"].to_numpy()
        pc_ = np.array([code[v] for v in p])
        acc, f1, mrec, _ = summary_scores(yc, pc_)
        b = [summary_scores(yc[i], pc_[i]) for i in boots]
        boot_f1[m] = np.array([x[1] for x in b])
        rec_boot = np.array([x[3] for x in b])
        prob = df[[PROB_COLS[c] for c in CLASSES]].to_numpy()
        macro_auc = roc_auc_score(y, prob, multi_class="ovr", average="macro", labels=CLASSES)
        summary.append({"model": m, "accuracy": acc, "accuracy_ci": ci([x[0] for x in b]),
                        "macro_f1": f1, "macro_f1_ci": ci(boot_f1[m]), "macro_recall": mrec,
                        "macro_roc_auc": macro_auc})
        pr, rc, fs, sup = precision_recall_fscore_support(y, p, labels=CLASSES, zero_division=0)
        for k, c in enumerate(CLASSES):
            per_class.append({"model": m, "class": c, "support": int(sup[k]), "precision": pr[k],
                              "recall": rc[k], "recall_ci": ci(rec_boot[:, k]), "f1": fs[k],
                              "roc_auc": roc_auc_score(y == c, df[PROB_COLS[c]])})
        plot_confusion(m, y, p)
        plot_roc(m, df)

    pairs = list(combinations(MODELS, 2))
    pairwise = []
    for a, b2 in pairs:
        ok_a = data[a]["pred"].to_numpy() == y
        ok_b = data[b2]["pred"].to_numpy() == y
        nab, nba = int(np.sum(ok_a & ~ok_b)), int(np.sum(~ok_a & ok_b))
        pval = mcnemar_exact(nab, nba)
        diff = boot_f1[a] - boot_f1[b2]
        f1a = next(s["macro_f1"] for s in summary if s["model"] == a)
        f1b = next(s["macro_f1"] for s in summary if s["model"] == b2)
        pairwise.append({"model_a": a, "model_b": b2, "a_right_b_wrong": nab, "a_wrong_b_right": nba,
                         "mcnemar_p": pval, "bonferroni_threshold": ALPHA / len(pairs),
                         "significant_after_correction": pval < ALPHA / len(pairs),
                         "macro_f1_diff_a_minus_b": f1a - f1b, "diff_ci": ci(diff)})

    def fmt_ci(t):
        return f"[{t[0]:.4f}, {t[1]:.4f}]"

    out = pd.DataFrame(summary)
    for col in ("accuracy_ci", "macro_f1_ci"):
        out[col] = out[col].map(fmt_ci)
    out.to_csv(RESULTS / "model_summary.csv", index=False, float_format="%.4f")
    pc = pd.DataFrame(per_class)
    pc["recall_ci"] = pc["recall_ci"].map(fmt_ci)
    pc.to_csv(RESULTS / "per_class_metrics.csv", index=False, float_format="%.4f")
    pw = pd.DataFrame(pairwise)
    pw["diff_ci"] = pw["diff_ci"].map(fmt_ci)
    pw.to_csv(RESULTS / "pairwise_comparison.csv", index=False, float_format="%.4f")

    pd.set_option("display.width", 160)
    print("MODELS\n", out.to_string(index=False), "\n")
    print("PER CLASS\n", pc.to_string(index=False), "\n")
    print("PAIRED COMPARISON\n", pw.to_string(index=False))
    print(f"\nSaved results and figures in {RESULTS}")


if __name__ == "__main__":
    main()
