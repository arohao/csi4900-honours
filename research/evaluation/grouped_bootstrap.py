
import numpy as np
import pandas as pd
import tldextract

from pathlib import Path
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = ROOT / "results" / "predictions"

dataset = pd.read_csv(
    ROOT / "data" / "phreshphish_domain_disjoint.csv"
)

def registered_domain(url):
    ext = tldextract.extract(url)

    if ext.suffix:
        return f"{ext.domain}.{ext.suffix}".lower()

    return ext.domain.lower()


def grouped_bootstrap(url_file, combined_file, n_bootstrap=5000):
    url_df = pd.read_csv(PRED_DIR / url_file)
    combined_df = pd.read_csv(PRED_DIR / combined_file)

    assert len(dataset) == len(url_df) == len(combined_df)

    y = dataset["label"].to_numpy()

    assert np.array_equal(y, url_df["label"].to_numpy())
    assert np.array_equal(y, combined_df["label"].to_numpy())

    pred_url = url_df["prediction"].to_numpy()
    pred_combined = combined_df["prediction"].to_numpy()

    domains = dataset["url"].apply(registered_domain).to_numpy()

    unique_domains = np.unique(domains)

    domain_indices = {
        domain: np.where(domains == domain)[0]
        for domain in unique_domains
    }

    observed = (
        f1_score(y, pred_combined)
        - f1_score(y, pred_url)
    )

    rng = np.random.default_rng(42)
    differences = []

    for _ in range(n_bootstrap):
        sampled_domains = rng.choice(
            unique_domains,
            size=len(unique_domains),
            replace=True
        )

        indices = np.concatenate([
            domain_indices[domain]
            for domain in sampled_domains
        ])

        difference = (
            f1_score(y[indices], pred_combined[indices], zero_division=0)
            - f1_score(y[indices], pred_url[indices], zero_division=0)
        )

        differences.append(difference)

    lower, upper = np.percentile(differences, [2.5, 97.5])

    return observed, lower, upper, len(unique_domains)


models = {
    "Logistic Regression": "logistic_regression",
    "Random Forest": "random_forest"
}

for name, filename in models.items():
    observed, lower, upper, n_domains = grouped_bootstrap(
        f"phreshphish_domain-disjoint_url-only_{filename}.csv",
        f"phreshphish_domain-disjoint_url_+_html_{filename}.csv"
    )

    print(f"\n{name}")
    print(f"Unique domains: {n_domains}")
    print(f"F1 improvement: {observed:.4f}")
    print(f"95% grouped CI: [{lower:.4f}, {upper:.4f}]")
