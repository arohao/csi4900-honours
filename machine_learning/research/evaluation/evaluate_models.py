
import json
from pathlib import Path

import numpy as np
import pandas as pd

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
    average_precision_score
)

ROOT = Path(__file__).resolve().parents[1]

train = pd.read_csv(ROOT / "data" / "comp_train.csv")
validation = pd.read_csv(ROOT / "data" / "comp_validation.csv")
phresh = pd.read_csv(
    ROOT / "data" / "phreshphish_domain_disjoint.csv"
)

schema = json.loads(
    (ROOT / "phishing_pipeline" / "feature_schema.json").read_text()
)

url_features = schema[:47]
combined_features = schema

feature_sets = {
    "URL-only": url_features,
    "URL + HTML": combined_features
}

def evaluate(model, X, y):
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y, predictions),
        "Precision": precision_score(y, predictions, zero_division=0),
        "Recall": recall_score(y, predictions, zero_division=0),
        "F1": f1_score(y, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y, probabilities),
        "PR-AUC": average_precision_score(y, probabilities)
    }

    return metrics, predictions, probabilities

results = []

for feature_set_name, columns in feature_sets.items():
    X_train = train[columns]
    y_train = train["label"]

    X_val = validation[columns]
    y_val = validation["label"]

    X_phresh = phresh[columns]
    y_phresh = phresh["label"]

    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                random_state=42
            )
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        )
    }

    for model_name, model in models.items():
        print(f"Training {model_name} ({feature_set_name})...")

        model.fit(X_train, y_train)

        for dataset_name, X, y in [
            ("CompPhish Validation", X_val, y_val),
            ("PhreshPhish Domain-Disjoint", X_phresh, y_phresh)
        ]:
            metrics, predictions, probabilities = evaluate(model, X, y)
            prediction_dir = ROOT / "results" / "predictions"
            prediction_dir.mkdir(parents=True, exist_ok=True)

            filename = (
                dataset_name.lower().replace(" ", "_")
                + "_"
                + feature_set_name.lower().replace(" ", "_")
                + "_"
                + model_name.lower().replace(" ", "_")
                + ".csv"
            )

            pd.DataFrame({
                "label": y.to_numpy(),
                "prediction": predictions,
                "probability": probabilities
            }).to_csv(prediction_dir / filename, index=False)
            
            results.append({
                "Dataset": dataset_name,
                "Features": feature_set_name,
                "Model": model_name,
                **metrics
            })

results_df = pd.DataFrame(results)

print("\nEvaluation Results:\n")
print(results_df.round(4).to_string(index=False))

output_dir = ROOT / "results"
output_dir.mkdir(parents=True, exist_ok=True)

results_df.to_csv(
    output_dir / "domain_disjoint_model_results.csv",
    index=False
)

print("\nResults saved successfully.")
