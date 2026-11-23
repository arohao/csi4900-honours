import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXTRACTOR_DIR = ROOT / "original_extractor"

sys.path.insert(0, str(EXTRACTOR_DIR))
os.chdir(EXTRACTOR_DIR)

import feature_Generator_with_brands as original

df = pd.read_excel(ROOT / "data" / "All_Features_threshold90.xlsx")

sample = df.groupby("label", group_keys=False).sample(
    n=250,
    random_state=42
)

def extract_brand_features(url):
    domain = original.extract_domain(url)

    return {
        "brand_subdomain": original.brand_in_subdomain(domain),
        "brand_substring_domain": original.brand_as_substring_in_domain(domain),
        "brand_typo_dom": original.brand_typo_in_domain(
            domain, original.brand_names
        ),
        "fake_tld": original.brand_with_fake_tld(domain),
        "brand_in_path_query": original.is_brand_in_path_query(url)
    }

results = []

for i, (_, row) in enumerate(sample.iterrows(), start=1):
    features = extract_brand_features(row["url"])
    results.append(features)

    if i % 10 == 0:
        print(f"Processed {i}/{len(sample)} URLs", flush=True)

predictions = pd.DataFrame(results, index=sample.index)

for feature in predictions.columns:
    matches = np.isclose(
        predictions[feature].astype(float),
        sample[feature].astype(float),
        rtol=1e-9,
        atol=1e-12
    )

    print(
        f"{feature}: {matches.mean() * 100:.2f}% agreement "
        f"({(~matches).sum()} mismatches)"
    )

    if not matches.all():
        mismatches = sample.loc[~matches, ["url", feature]].copy()
        mismatches["calculated"] = predictions.loc[
            ~matches, feature
        ]

        print(mismatches.head(5).to_string(index=False))

feature = "brand_subdomain"

mismatches = sample[
    predictions[feature] != sample[feature]
]

for url in mismatches["url"]:
    print(f"\nURL: {url}")
    print(f"Extracted domain: {original.extract_domain(url)}")
    print(f"Calculated: {extract_brand_features(url)[feature]}")

feature = "brand_subdomain"

mask = predictions[feature] != sample[feature]
mismatches = sample.loc[mask]

print("\nMismatch labels:")
print(mismatches["label"].value_counts())

print("\nOriginal values:")
print(mismatches[feature].value_counts())

print("\nCalculated values:")
print(predictions.loc[mask, feature].value_counts())