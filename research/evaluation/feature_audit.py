
import json
import numpy as np
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

results = []

for feature in schema:
    train_values = train[feature].astype(float)
    phresh_values = phresh[feature].astype(float)

    train_mean = train_values.mean()
    phresh_mean = phresh_values.mean()

    train_std = train_values.std()
    phresh_std = phresh_values.std()

    pooled_std = np.sqrt(
        (train_std**2 + phresh_std**2) / 2
    )

    if pooled_std > 0:
        smd = abs(train_mean - phresh_mean) / pooled_std
    else:
        smd = 0.0

    results.append({
        "Feature": feature,
        "Type": "URL" if feature in schema[:47] else "HTML",
        "Train Mean": train_mean,
        "Phresh Mean": phresh_mean,
        "Train Std": train_std,
        "Phresh Std": phresh_std,
        "SMD": smd,
        "Train Unique": train_values.nunique(),
        "Phresh Unique": phresh_values.nunique()
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    "SMD", ascending=False
)

print("\nTop 20 features with largest distribution differences:\n")

print(
    results_df[
        ["Feature", "Type", "Train Mean", "Phresh Mean", "SMD"]
    ].head(20).round(4).to_string(index=False)
)

print("\nConstant features in PhreshPhish:\n")

print(
    results_df.loc[
        results_df["Phresh Unique"] == 1,
        ["Feature", "Type", "Train Unique"]
    ].to_string(index=False)
)

print("\nAverage SMD by feature type:\n")

print(results_df.groupby("Type")["SMD"].mean())

output_dir = ROOT / "results"
output_dir.mkdir(parents=True, exist_ok=True)

results_df.to_csv(
    output_dir / "feature_distribution_audit.csv",
    index=False
)

print("\nSaved feature_distribution_audit.csv")
