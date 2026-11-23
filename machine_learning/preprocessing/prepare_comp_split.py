import pandas as pd
import tldextract
from pathlib import Path
import json

from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parents[1]

df = pd.read_excel(ROOT / "data" / "All_Features_threshold90.xlsx")

df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)

def registered_domain(url):
    ext = tldextract.extract(url)

    if ext.suffix:
        return f"{ext.domain}.{ext.suffix}".lower()

    return ext.domain.lower()

groups = df["url"].apply(registered_domain)

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, val_idx = next(
    splitter.split(df, df["label"], groups=groups)
)

train = df.iloc[train_idx].copy()
val = df.iloc[val_idx].copy()

mean_phishing_length = train.loc[
    train["label"] == 1, "url_length"
].mean()

mean_domain_length = train.loc[
    train["label"] == 1, "domain_length"
].mean()

print("Training records:", len(train))
print("Validation records:", len(val))
print("Training label distribution:", train["label"].value_counts().to_dict())
print("Validation label distribution:", val["label"].value_counts().to_dict())
print("Mean phishing URL length:", mean_phishing_length)
print("Mean phishing domain length:", mean_domain_length)

for dataset in [train, val]:
    dataset["url_more_than_mean"] = (
        dataset["url_length"] >= mean_phishing_length
    ).astype(int)

    dataset["domain_more_than_mean"] = (
        dataset["domain_length"] >= mean_domain_length
    ).astype(int)

thresholds = {
    "mean_phishing_length": float(mean_phishing_length),
    "mean_domain_length": float(mean_domain_length)
}

with open(ROOT / "data" / "training_thresholds.json", "w") as f:
    json.dump(thresholds, f, indent=4)

train.to_csv(ROOT / "data" / "comp_train.csv", index=False)
val.to_csv(ROOT / "data" / "comp_validation.csv", index=False)

train_domains = set(train["url"].apply(registered_domain))
val_domains = set(val["url"].apply(registered_domain))

print("Overlapping domains:", len(train_domains & val_domains))