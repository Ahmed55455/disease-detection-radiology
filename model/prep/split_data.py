import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

SEED = 42
df = pd.read_csv("data/all_images_clean.csv")[["path", "label"]]

# 70% train, 30% temp
train_df, temp_df = train_test_split(
    df, test_size=0.30, stratify=df["label"], random_state=SEED
)
# split temp into 15% val, 15% test
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df["label"], random_state=SEED
)

out = Path("data/splits")
out.mkdir(parents=True, exist_ok=True)
train_df.to_csv(out / "train.csv", index=False)
val_df.to_csv(out / "val.csv", index=False)
test_df.to_csv(out / "test.csv", index=False)

for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
    print(f"\n{name}: {len(part)} images")
    print(part["label"].value_counts())
    print((part["label"].value_counts(normalize=True) * 100).round(1))

# sanity check: no overlap between splits
assert not set(train_df.path) & set(val_df.path)
assert not set(train_df.path) & set(test_df.path)
assert not set(val_df.path) & set(test_df.path)
print("\nNo overlap between splits: OK")