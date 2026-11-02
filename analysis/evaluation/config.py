"""Shared settings for the evaluation, urgency rule and simulation (Member 3 work).

Every number here that is not measured is an ASSUMPTION and must be named as such in the report.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PRED_DIR = REPO / "data" / "predictions"        # *_test_probs.csv from Drive (not in GitHub)
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

CLASSES = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]     # docs/interfaces.md order
PROB_COLS = {c: f"p_{c}" for c in CLASSES}                          # p_COVID, p_Lung_Opacity, ...
DISEASES = ["COVID", "Lung_Opacity", "Viral Pneumonia"]             # "abnormal" = not Normal
MODELS = ["resnet50", "densenet121", "efficientnet_b0"]
APP_MODEL = "efficientnet_b0"                                        # model used in the app (provisional)

SEED = 42
N_BOOT = 2000            # bootstrap resamples for confidence intervals
ALPHA = 0.05

# ---- Urgency rule ----------------------------------------------------------------
# urgency = sum(probability x weight), range 0 to 1 (fixed in docs/interfaces.md).
# MAIN = the weights already agreed in docs/interfaces.md and used by the app (no change needed
# in the app). Two alternatives are tested in the simulation (sensitivity analysis), because the
# weights are assumptions, not clinical facts.
WEIGHT_SETS = {
    "W1 main (interfaces.md)": {"COVID": 1.0, "Viral Pneumonia": 0.7, "Lung_Opacity": 0.5, "Normal": 0.1},
    "W2 any disease equal":    {"COVID": 1.0, "Viral Pneumonia": 1.0, "Lung_Opacity": 1.0, "Normal": 0.0},
    "W3 COVID first":          {"COVID": 1.0, "Viral Pneumonia": 0.4, "Lung_Opacity": 0.4, "Normal": 0.0},
}
MAIN_WEIGHTS = "W1 main (interfaces.md)"

# ---- Cost-sensitive threshold ----------------------------------------------------
# Missing a sick patient (false negative) is assumed to cost more than a false alarm.
# The ratio is an assumption, so several ratios are reported; MAIN_COST_RATIO is used downstream.
COST_RATIOS = [1, 2, 5, 10]
MAIN_COST_RATIO = 5

# ---- Simulation (ASSUMPTIONS, not hospital measurements) ---------------------------
N_RUNS = 30
MEAN_READING_MIN = 6.0           # mean reading time per X-ray, exponential
MEAN_INTERARRIVAL_MIN = 8.0      # mean time between arrivals, exponential (utilization 0.75)
HIGH_LOAD_INTERARRIVAL_MIN = 6.67  # second load level (utilization about 0.90)
N_RADIOLOGISTS = 1
WEAK_SPECIFICITY = 0.50          # weaker-model scenario (new pediatric images, model_findings.md s.5)
WEAK_SENSITIVITY = 0.98
