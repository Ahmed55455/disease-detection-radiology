from pathlib import Path

import pandas as pd

overlap = pd.read_csv("data/external_overlap.csv")
seen = {Path(p).as_posix() for p in overlap.loc[overlap["corr"] >= 0.98, "external_path"]}

for name in ["resnet50", "efficientnet_b0", "densenet121"]:
    r = pd.read_csv(f"data/predictions/external_{name}_probs.csv")
    r = r[~r["path"].map(lambda p: Path(p).as_posix() in seen)].copy()
    r["pred_pneu"] = r["pred4"] != "Normal"
    r["correct"] = (r["label"] == "PNEUMONIA") == r["pred_pneu"]
    print(f"\n{name}: clean images only (NORMAL rate = specificity, PNEUMONIA rate = sensitivity)")
    print(r.groupby(["split", "label"])["correct"].agg(["size", "mean"]).round(3).to_string())
    