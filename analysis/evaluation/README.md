# analysis/evaluation — statistics, urgency rule and simulation (Member 3 role)

The system assists radiologists and does not replace them. It is limited by its training data.

## Setup (once)

```powershell
pip install simpy pandas scikit-learn matplotlib
```
Download from Google Drive (folder `predictions`) into `data/predictions/` (not in GitHub):
`resnet50_test_probs.csv`, `densenet121_test_probs.csv`, `efficientnet_b0_test_probs.csv`.

## Run (from the repository root, in this order)

```powershell
python analysis/evaluation/step1_model_evaluation.py     # about 5 s
python analysis/evaluation/step2_threshold_urgency.py    # about 2 s
python analysis/evaluation/step3_simulation.py           # about 20 s
```
All outputs go to `analysis/evaluation/results/`. All settings (weights, cost ratios, simulation
times) are in `config.py`.

## What each step does

| Step | Question | Method | Outputs |
|---|---|---|---|
| 1 | How good is each model, and is the difference between models real? | Confusion matrix, per-class precision/recall/F1/ROC-AUC, 95% bootstrap CIs; exact McNemar test (paired, same images) with Bonferroni correction; bootstrap CI of the macro-F1 difference | `model_summary.csv`, `per_class_metrics.csv`, `pairwise_comparison.csv`, `confusion_*.png`, `roc_*.png` |
| 2 | Where should we put the decision threshold, and which cases are HIGH? | Cost-sensitive threshold (missed sick patient costs 1, 2, 5 or 10 x a false alarm) on 1 - P(Normal); same cost rule for the HIGH threshold of the urgency score. Tuned on one half of the test predictions, reported on the other half | `threshold_results.csv`, `urgency_threshold.csv`, `urgency_scores.csv`, `cost_curve.png` |
| 3 | Does AI priority reduce the waiting time of truly sick patients? | SimPy, FIFO vs AI priority, 30 runs, same times for both strategies, waits by TRUE class; scenarios: actual model, weaker model (specificity 0.5), alternative weights, high load | `simulation_summary.csv`, `simulation_waits.png` |

## Decisions and assumptions (state them in the report)

- **Urgency rule (main):** `urgency = 1.0 P(COVID) + 0.7 P(Viral Pneumonia) + 0.5 P(Lung Opacity) + 0.1 P(Normal)`,
  range 0-1, the weights agreed in `docs/interfaces.md` and already used by the app. The weights are
  assumptions, so two alternatives (all diseases equal; COVID first) are tested in step 3.
- **Queue:** ordered by the continuous urgency score (not two tiers). HIGH/STANDARD is a label only.
  The app uses the same order.
- **Cost ratio:** main 5:1 (false negative : false positive); 1, 2 and 10 also reported.
- **Threshold tuning:** only test-set probabilities exist, so they are split 50/50 (stratified, seed 42):
  tune on one half, report on the other. Validation-set probabilities from Member 1 would be better.
- **Simulation times:** one radiologist; reading time exponential, mean 6 min; arrivals exponential,
  mean 8 min (utilization 0.75) and 6.67 min (utilization 0.9). Assumptions, not hospital data.
- **Weaker model:** errors injected at random to reach specificity 0.5 (new pediatric images,
  `docs/model_findings.md` section 5). If the actual sensitivity is already below 0.98, no misses are added.
- Results describe the seed-42 checkpoints and the original test set only.
