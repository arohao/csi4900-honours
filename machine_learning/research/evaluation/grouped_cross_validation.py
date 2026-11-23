
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tldextract

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
import sys

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT))

from phishing_pipeline.feature_pipeline import (
    compute_training_thresholds,
    load_original,
)


train = pd.read_csv(ROOT / "data" / "comp_train.csv")

schema = json.loads(
    (ROOT / "phishing_pipeline" / "feature_schema.json").read_text()
)

feature_sets = {
    "URL-only": schema[:47],
    "URL + HTML": schema,
}

extractor = tldextract.TLDExtract(suffix_list_urls=())

def get_domain(url):
    result = extractor(str(url))
    return result.top_domain_under_public_suffix or result.domain or str(url)

groups = train["url"].apply(get_domain)
y = train["label"]

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

results = []

original = load_original()

def update_threshold_features(X_train, X_val, train_rows):
    thresholds = compute_training_thresholds(
        train_rows[["url", "label"]].to_dict("records")
    )

    X_train = X_train.copy()
    X_val = X_val.copy()

    for X in (X_train, X_val):
        if "url_more_than_mean" in X.columns:
            url_lengths = train.loc[X.index, "url"].astype(str).str.len()

            X["url_more_than_mean"] = (
                url_lengths > thresholds["mean_phishing_length"]
            ).astype(int)

        if "domain_more_than_mean" in X.columns:
            domain_lengths = train.loc[X.index, "url"].astype(str).apply(
                lambda url: len(original.extract_domain(url))
            )

            X["domain_more_than_mean"] = (
                domain_lengths > thresholds["mean_domain_length"]
            ).astype(int)

    return X_train, X_val

for feature_set_name, columns in feature_sets.items():
    X = train[columns]

    models = {
        "Logistic Regression": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                random_state=42,
            ),
        ),
        "Random Forest": lambda: RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        ),
    }

    for model_name, create_model in models.items():
        print(f"\n{feature_set_name} | {model_name}")

        for fold, (train_idx, val_idx) in enumerate(
            cv.split(X, y, groups),
            start=1,
        ):
            X_train = X.iloc[train_idx].copy()
            X_val = X.iloc[val_idx].copy()

            y_train = y.iloc[train_idx]
            y_val = y.iloc[val_idx]

            X_train, X_val = update_threshold_features(
                X_train,
                X_val,
                train.iloc[train_idx]
            )
            train_groups = set(groups.iloc[train_idx])
            val_groups = set(groups.iloc[val_idx])

            assert train_groups.isdisjoint(val_groups)

            model = create_model()
            model.fit(X_train, y_train)

            predictions = model.predict(X_val)

            phishing_index = list(model.classes_).index(1)
            probabilities = model.predict_proba(X_val)[:, phishing_index]

            metrics = {
                "Accuracy": accuracy_score(y_val, predictions),
                "Precision": precision_score(
                    y_val, predictions, zero_division=0
                ),
                "Recall": recall_score(
                    y_val, predictions, zero_division=0
                ),
                "F1": f1_score(
                    y_val, predictions, zero_division=0
                ),
                "ROC-AUC": roc_auc_score(y_val, probabilities)
                    if y_val.nunique() == 2 else np.nan,
                "PR-AUC": average_precision_score(
                    y_val, probabilities
                ) if y_val.nunique() == 2 else np.nan,
            }

            results.append({
                "Features": feature_set_name,
                "Model": model_name,
                "Fold": fold,
                "Train Samples": len(train_idx),
                "Validation Samples": len(val_idx),
                "Validation Domains": len(val_groups),
                **metrics,
            })

            print(
                f"Fold {fold}: "
                f"F1={metrics['F1']:.4f}, "
                f"ROC-AUC={metrics['ROC-AUC']:.4f}"
            )

results_df = pd.DataFrame(results)

metric_columns = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC-AUC",
    "PR-AUC",
]

summary = results_df.groupby(
    ["Features", "Model"]
)[metric_columns].agg(["mean", "std"])

output_dir = ROOT / "results"
output_dir.mkdir(parents=True, exist_ok=True)

results_df.to_csv(
    output_dir / "grouped_cv_folds.csv",
    index=False,
)

summary.to_csv(
    output_dir / "grouped_cv_summary.csv",
)

print("\n5-Fold Cross-Validation Summary:\n")
print(summary.round(4).to_string())

print("\nResults saved.")
