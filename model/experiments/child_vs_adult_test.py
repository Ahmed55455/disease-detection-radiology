import pandas as pd

overlap = pd.read_csv("data/external_overlap.csv")
# our images that also appear in the pediatric dataset (corr >= 0.98)
child = set(overlap.loc[overlap["corr"] >= 0.98, "match_path"])

for name in ["resnet50", "efficientnet_b0", "densenet121"]:
    t = pd.read_csv(f"data/predictions/{name}_test_probs.csv")
    t["from_pediatric_set"] = t["path"].isin(child)
    t["correct"] = t["pred"] == t["label"]
    g = t.groupby(["label", "from_pediatric_set"])["correct"].agg(["size", "mean"])
    print(f"\n{name}: recall on OUR test split")
    print(g.round(3).to_string())