import re
from pathlib import Path

import pandas as pd

ov = pd.read_csv("data/external_overlap.csv")
m = ov[ov["corr"] >= 0.98].copy()
m["fname"] = m["external_path"].map(lambda p: Path(p).name)

split_of = {}
for s in ["train", "val", "test"]:
    for p in pd.read_csv(f"data/splits/{s}.csv")["path"]:
        split_of[p] = s
m["our_split"] = m["match_path"].map(split_of)


def vp_id(f):
    x = re.match(r"person(\d+)_", f)
    return x.group(1) if x else None


def normal_id(f):
    x = re.search(r"IM-(\d{4})", f)
    if not x:
        return None
    return ("N2-" if f.startswith("NORMAL2") else "IM-") + x.group(1)


for cls, idfun in [("Viral Pneumonia", vp_id), ("Normal", normal_id)]:
    g = m[m["match_label"] == cls].copy()
    g["gid"] = g["fname"].map(idfun)
    print(f"\n===== {cls} =====")
    print(f"images {len(g)} | id parsed {int(g['gid'].notna().sum())} | distinct ids {g['gid'].nunique()}")
    g = g.dropna(subset=["gid"])
    train_ids = set(g.loc[g["our_split"] == "train", "gid"])
    test_g = g[g["our_split"] == "test"].copy()
    test_g["leaked"] = test_g["gid"].isin(train_ids)
    print(f"test images {len(test_g)} | whose id also appears in our train split: {int(test_g['leaked'].sum())}")

    for name in ["resnet50", "efficientnet_b0", "densenet121"]:
        t = pd.read_csv(f"data/predictions/{name}_test_probs.csv")
        t = t.merge(test_g[["match_path", "leaked"]], left_on="path", right_on="match_path")
        t["correct"] = t["pred"] == t["label"]
        print(f"\n{name}: recall on test images")
        print(t.groupby("leaked")["correct"].agg(["size", "mean"]).round(3).to_string())