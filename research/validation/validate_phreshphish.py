import json
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

df = pd.read_csv(
    ROOT / "data" / "phreshphish_preliminary.csv"
)

schema = json.loads(
    (ROOT / "phishing_pipeline" / "feature_schema.json").read_text()
)

print("Records:", len(df))
print("Columns:", len(df.columns))
print("Labels:", df["label"].value_counts().to_dict())

missing_features = set(schema) - set(df.columns)
extra_features = set(df.columns) - set(schema)

print("Missing features:", missing_features)
print("Metadata columns:", extra_features)

features = df[schema]

print("Missing values:", features.isna().sum().sum())
print("Infinite values:", np.isinf(features.to_numpy(dtype=float)).sum())
print("Constant features:", features.columns[features.nunique() <= 1].tolist())
print("Duplicate URLs:", df["url"].duplicated().sum())
print("Duplicate SHA256:", df["sha256"].duplicated().sum())