from pathlib import Path

import pandas as pd

ov = pd.read_csv("data/external_overlap.csv")
m = ov[ov["corr"] >= 0.98].copy()
m["fname"] = m["external_path"].map(lambda p: Path(p).name)

print("Sample external file names per class:")
for lab, g in m.groupby("match_label"):
    print(" ", lab, g["fname"].sample(5, random_state=0).tolist())

split_of = {}
for s in ["train", "val", "test"]:
    for p in pd.read_csv(f"data/splits/{s}.csv")["path"]:
        split_of[p] = s
m["our_split"] = m["match_path"].map(split_of)

vp = m[m["match_label"] == "Viral Pneumonia"].copy()
vp["person"] = vp["fname"].str.extract(r"person(\d+)_", expand=False)
print("\nViral Pneumonia matched images:", len(vp),
      "| person id parsed:", int(vp["person"].notna().sum()),
      "| distinct persons:", vp["person"].nunique())

vp = vp.dropna(subset=["person"])
if len(vp):
    sizes = vp.groupby("person").size()
    print("Persons with more than 1 image:", int((sizes > 1).sum()))
    splits_per_person = vp.groupby("person")["our_split"].nunique()
    leaking = splits_per_person[splits_per_person > 1].index
    print("Persons spread over more than one of OUR splits:", len(leaking))
    print("Images belonging to those persons:", int(vp["person"].isin(leaking).sum()))
    test_p = set(vp.loc[vp.our_split == "test", "person"])
    train_p = set(vp.loc[vp.our_split == "train", "person"])
    print(f"Test persons that also appear in our train split: {len(test_p & train_p)} of {len(test_p)}")
else:
    print("Could not parse person ids. Send me the sample names above.")