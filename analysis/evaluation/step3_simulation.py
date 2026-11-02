"""Step 3: discrete-event simulation (SimPy) of a radiology reading queue, FIFO vs AI priority.

Queue  : N_RADIOLOGISTS radiologist(s), non-preemptive (a reading in progress is not interrupted).
FIFO   : cases are read in arrival order.
AI     : the waiting case with the highest CONTINUOUS urgency score is read next (ties: arrival).
         This is NOT a two-tier queue; HIGH/STANDARD is only a label. The app orders the same way.
Cases  : the 3,167 test images with their TRUE class and the app model's urgency score
         (results/urgency_scores.csv from step 2), so model errors are included.
Runs   : N_RUNS replications; in each run FIFO and AI get the SAME arrival and reading times.
         Results: mean and 95% confidence interval over the runs.

Scenarios
  A  actual model, main weights, normal load (utilization about 0.75)   <- main result
  B  weaker model (specificity 0.5, sensitivity about 0.98), main weights, normal load
  C  actual model, alternative weights W2 and W3 (sensitivity analysis of the urgency rule)
  D  actual model, main weights, high load (utilization about 0.90)

Weaker model (B): errors are injected at random into the actual predictions. A true Normal case that
becomes a false alarm gets an urgency drawn from the cases the model predicted Viral Pneumonia (the
most common error on new pediatric images, docs/model_findings.md s.5); a missed disease case gets an
urgency drawn from the cases predicted Normal.

Output: analysis/evaluation/results/simulation_summary.csv, simulation_waits.png
Run   : python analysis/evaluation/step3_simulation.py   (from the repository root, after step 2)
ALL TIME PARAMETERS ARE ASSUMPTIONS, NOT HOSPITAL MEASUREMENTS.
"""
import math
import random
import statistics as st
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import simpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (CLASSES, HIGH_LOAD_INTERARRIVAL_MIN, MAIN_WEIGHTS, MEAN_INTERARRIVAL_MIN,
                    MEAN_READING_MIN, N_RADIOLOGISTS, N_RUNS, RESULTS, SEED, WEAK_SENSITIVITY,
                    WEAK_SPECIFICITY, WEIGHT_SETS)

ORDER_CLASSES = ["COVID", "Lung_Opacity", "Viral Pneumonia", "Normal"]


def load_cases():
    path = RESULTS / "urgency_scores.csv"
    thr = RESULTS / "urgency_threshold.csv"
    if not path.exists() or not thr.exists():
        sys.exit("Run step2_threshold_urgency.py first.")
    df = pd.read_csv(path)
    high = dict(zip(*pd.read_csv(thr)[["weights", "high_threshold"]].T.values))
    return df, high


def weak_model(urg, labels, preds, rng):
    """Inject errors so specificity ~ WEAK_SPECIFICITY and sensitivity ~ WEAK_SENSITIVITY."""
    urg = urg.copy()
    is_normal_pred = preds == "Normal"
    sick = labels != "Normal"
    sens0 = np.mean(~is_normal_pred[sick])
    spec0 = np.mean(is_normal_pred[~sick])
    p_fa = max(0.0, 1 - WEAK_SPECIFICITY / spec0)
    p_miss = max(0.0, 1 - WEAK_SENSITIVITY / sens0)
    vp_pool = urg[preds == "Viral Pneumonia"]
    normal_pool = urg[is_normal_pred]
    new_pred_normal = is_normal_pred.copy()
    for i in range(len(urg)):
        if not sick[i] and is_normal_pred[i] and rng.random() < p_fa:
            urg[i], new_pred_normal[i] = rng.choice(vp_pool), False
        elif sick[i] and not is_normal_pred[i] and rng.random() < p_miss:
            urg[i], new_pred_normal[i] = rng.choice(normal_pool), True
    sens = np.mean(~new_pred_normal[sick])
    spec = np.mean(new_pred_normal[~sick])
    return urg, sens, spec


def simulate(urg, arrivals, readings, strategy):
    """Waiting time (minutes) of each case."""
    env = simpy.Environment()
    rad = simpy.PriorityResource(env, capacity=N_RADIOLOGISTS)
    wait = np.zeros(len(urg))

    def case(i):
        yield env.timeout(arrivals[i])
        with rad.request(priority=0 if strategy == "FIFO" else -urg[i]) as req:
            yield req
            wait[i] = env.now - arrivals[i]
            yield env.timeout(readings[i])

    for i in range(len(urg)):
        env.process(case(i))
    env.run()
    return wait


def metrics(wait, readings, labels, urg, high):
    m = {"mean wait, all cases": wait.mean()}
    for c in ORDER_CLASSES:
        m[f"mean wait, true {c}"] = wait[labels == c].mean()
    m["mean time to diagnosis, true COVID"] = (wait + readings)[labels == "COVID"].mean()
    std = urg < high
    sick_std = (labels != "Normal") & std
    m["true disease labeled STANDARD: count"] = int(sick_std.sum())
    m["true disease labeled STANDARD: mean wait"] = wait[sick_std].mean() if sick_std.any() else math.nan
    m["95th percentile wait, all cases"] = float(np.percentile(wait, 95))
    m["95th percentile wait, true Normal"] = float(np.percentile(wait[labels == "Normal"], 95))
    return m


def ci(values):
    v = [x for x in values if not math.isnan(x)]
    if not v:
        return math.nan, math.nan
    half = 1.96 * st.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else 0.0
    return st.mean(v), half


def main():
    df, high = load_cases()
    labels = df["label"].to_numpy()
    preds = df["pred"].to_numpy()
    urg_col = {w: df[f"urgency [{w}]"].to_numpy() for w in WEIGHT_SETS}
    other = [w for w in WEIGHT_SETS if w != MAIN_WEIGHTS]

    scenarios = [  # (name, weights, model, interarrival)
        ("A: actual model", MAIN_WEIGHTS, "actual", MEAN_INTERARRIVAL_MIN),
        ("B: weaker model (spec 0.5)", MAIN_WEIGHTS, "weak", MEAN_INTERARRIVAL_MIN),
        *[(f"C: weights {w}", w, "actual", MEAN_INTERARRIVAL_MIN) for w in other],
        ("D: high load (util 0.9)", MAIN_WEIGHTS, "actual", HIGH_LOAD_INTERARRIVAL_MIN),
    ]
    res = {}          # (scenario, strategy) -> metric -> list over runs
    quality = {"actual": [], "weak": []}

    for run in range(N_RUNS):
        rng = random.Random(SEED + run)
        readings = np.array([rng.expovariate(1 / MEAN_READING_MIN) for _ in labels])
        gaps = np.array([rng.expovariate(1.0) for _ in labels])      # unit-rate gaps, scaled per load
        weak_urg, sens, spec = weak_model(urg_col[MAIN_WEIGHTS], labels, preds,
                                          random.Random(10_000 + SEED + run))
        quality["weak"].append((sens, spec))
        fifo_done = {}
        for name, w, model, inter in scenarios:
            arrivals = np.cumsum(gaps * inter)
            urg = weak_urg if model == "weak" else urg_col[w]
            if inter not in fifo_done:                                  # FIFO does not use the model
                fifo_done[inter] = simulate(urg, arrivals, readings, "FIFO")
            for strat, wait in (("FIFO", fifo_done[inter]),
                                ("AI priority", simulate(urg, arrivals, readings, "AI"))):
                for k, v in metrics(wait, readings, labels, urg, high[w]).items():
                    res.setdefault((name, strat), {}).setdefault(k, []).append(v)
        print(f"run {run + 1}/{N_RUNS} done")

    sick = labels != "Normal"
    sens_a = float(np.mean(preds[sick] != "Normal"))
    spec_a = float(np.mean(preds[~sick] == "Normal"))
    sens_b, spec_b = np.mean([q[0] for q in quality["weak"]]), np.mean([q[1] for q in quality["weak"]])
    print(f"\nModel quality (disease vs Normal): actual sens {sens_a:.3f} spec {spec_a:.3f} | "
          f"weaker sens {sens_b:.3f} spec {spec_b:.3f}")

    rows = []
    for (name, strat), ms in res.items():
        for k, v in ms.items():
            mean, half = ci(v)
            rows.append({"scenario": name, "strategy": strat, "metric": k,
                         "mean": round(mean, 3), "ci95_half_width": round(half, 3)})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "simulation_summary.csv", index=False)

    table = out.assign(value=out["mean"].map("{:.2f}".format) + " ± " +
                       out["ci95_half_width"].map("{:.2f}".format))
    wide = table.pivot_table(index="metric", columns=["scenario", "strategy"], values="value",
                             aggfunc="first", sort=False)
    pd.set_option("display.width", 250, "display.max_columns", 20)
    for name, *_ in scenarios:
        print(f"\n=== {name} ({N_RUNS} runs, mean ± 95% CI, minutes) ===")
        print(wide[name].to_string())

    # Figure: mean wait by true class, FIFO vs AI, scenarios A and B
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, name in zip(axes, [scenarios[0][0], scenarios[1][0]]):
        x = np.arange(len(ORDER_CLASSES))
        for k, strat in enumerate(("FIFO", "AI priority")):
            vals = [ci(res[(name, strat)][f"mean wait, true {c}"]) for c in ORDER_CLASSES]
            ax.bar(x + (k - 0.5) * 0.38, [v[0] for v in vals], 0.38, yerr=[v[1] for v in vals],
                   capsize=3, label=strat, color=["#9aa5b1", "#c0392b"][k])
        ax.set_xticks(x, [c.replace("_", " ") for c in ORDER_CLASSES])
        ax.set_title(name)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Mean waiting time (min)")
    axes[0].legend()
    fig.suptitle("Waiting time by TRUE class (30 runs, 95% CI). Times are assumptions.", fontsize=10)
    fig.tight_layout()
    fig.savefig(RESULTS / "simulation_waits.png", dpi=130)
    print(f"\nSaved {RESULTS / 'simulation_summary.csv'} and simulation_waits.png")


if __name__ == "__main__":
    main()
