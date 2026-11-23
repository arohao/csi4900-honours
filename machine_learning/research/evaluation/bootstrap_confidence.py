import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = ROOT / "results" / "predictions"

def bootstrap_f1_difference(url_file, combined_file, n_bootstrap=5000):
    url_df = pd.read_csv(PRED_DIR / url_file)
    combined_df = pd.read_csv(PRED_DIR / combined_file)

    assert np.array_equal(
        url_df["label"].to_numpy(),
        combined_df["label"].to_numpy()
    ), "Labels do not match!"

    y = url_df["label"].to_numpy()
    pred_url = url_df["prediction"].to_numpy()
    pred_combined = combined_df["prediction"].to_numpy()

    observed = (
        f1_score(y, pred_combined)
        - f1_score(y, pred_url)
    )

    rng = np.random.default_rng(42)
    differences = []

    for _ in range(n_bootstrap):
        indices = rng.choice(len(y), size=len(y), replace=True)

        differences.append(
            f1_score(y[indices], pred_combined[indices])
            - f1_score(y[indices], pred_url[indices])
        )

    lower, upper = np.percentile(differences, [2.5, 97.5])

    return observed, lower, upper


models = {
    "Logistic Regression": "logistic_regression",
    "Random Forest": "random_forest"
}

for name, filename in models.items():
    observed, lower, upper = bootstrap_f1_difference(
        f"phreshphish_domain-disjoint_url-only_{filename}.csv",
        f"phreshphish_domain-disjoint_url_+_html_{filename}.csv"
    )

    print(f"\n{name}")
    print(f"F1 improvement: {observed:.4f}")
    print(f"95% CI: [{lower:.4f}, {upper:.4f}]")