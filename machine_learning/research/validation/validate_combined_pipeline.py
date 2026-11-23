import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = ROOT / "phishing_pipeline"

sys.path.insert(0, str(PIPELINE_DIR))

from feature_pipeline import extract_features, compute_training_thresholds

df = pd.read_excel(ROOT / "data" / "All_Features_threshold90.xlsx")

sample = df.groupby("label", group_keys=False).sample(
    n=50,
    random_state=42
)

schema = json.loads(
    (PIPELINE_DIR / "feature_schema.json").read_text()
)

url_features = schema[:47]
html_features = schema[47:]

thresholds = compute_training_thresholds(
    df[["url", "label"]].to_dict("records")
)

print("URL features:", len(url_features))
print("HTML features:", len(html_features))
print("Thresholds:", thresholds)

predictions = []

for i, (_, row) in enumerate(sample.iterrows(), start=1):
    result = extract_features(
        url=row["url"],
        html="<html><body></body></html>",
        **thresholds
    )

    predictions.append(result)

    if i % 10 == 0:
        print(f"Processed {i}/{len(sample)} URLs")

predictions = pd.DataFrame(predictions, index=sample.index)

print("\nFeature validation results:\n")

for feature in url_features:
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