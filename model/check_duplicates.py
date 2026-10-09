import hashlib
from pathlib import Path
import pandas as pd
from tqdm import tqdm

df = pd.read_csv("data/all_images.csv")

def file_hash(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()

tqdm.pandas(desc="Hashing")
df["hash"] = df["path"].progress_apply(file_hash)

dup_mask = df.duplicated(subset="hash", keep=False)
dups = df[dup_mask]

print("\nTotal images           :", len(df))
print("Unique hashes          :", df["hash"].nunique())
print("Images in duplicate groups:", len(dups))

if len(dups) > 0:
    groups = dups.groupby("hash")["label"].nunique()
    print("Duplicate groups with DIFFERENT labels:", (groups > 1).sum())
    print(dups.sort_values("hash").head(10))

# Keep one copy of each exact duplicate
clean = df.drop_duplicates(subset="hash", keep="first")
clean.to_csv("data/all_images_clean.csv", index=False)
print("\nAfter removing exact duplicates:", len(clean))
print(clean["label"].value_counts())