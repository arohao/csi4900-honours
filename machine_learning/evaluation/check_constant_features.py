import json
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

train = pd.read_csv(ROOT / "data" / "comp_train.csv")
phresh = pd.read_csv(
    ROOT / "data" / "phreshphish_domain_disjoint.csv"
)

schema = json.loads(
    (ROOT / "phishing_pipeline" / "feature_schema.json").read_text()
)

print("\nConstant features in training:\n")

for feature in schema:
    if train[feature].nunique() == 1:
        print(
            f"{feature}: "
            f"Train={train[feature].iloc[0]}, "
            f"Phresh unique={phresh[feature].nunique()}"
        )

print("\nFeatures present in PhreshPhish but always zero in training:\n")

for feature in schema:
    if (
        train[feature].eq(0).all()
        and phresh[feature].ne(0).any()
    ):
        print(
            f"{feature}: "
            f"{phresh[feature].ne(0).sum()} nonzero external samples"
        )
original = pd.read_excel(
    ROOT / "data" / "All_Features_threshold90.xlsx"
)

print("\nOriginal brand_subdomain distribution:")
print(original["brand_subdomain"].value_counts())

print("\nBy phishing label:")
print(pd.crosstab(
    original["label"],
    original["brand_subdomain"]
))