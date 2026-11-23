"""Paired domain-cluster bootstrap of delta F1 on frozen final prediction files."""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from common import RESULTS


def main():
    predictions = pd.read_csv(RESULTS / 'final_predictions.csv', dtype={'sha256': str})
    rng = np.random.default_rng(20261123)
    output = []
    for subset_name in ['all_external', 'domain_disjoint']:
        for model in ['logistic_regression', 'random_forest']:
            sub = predictions[predictions['model'] == model]
            if subset_name == 'domain_disjoint':
                sub = sub[sub['domain_disjoint']]
            url = sub[sub['features'] == 'url_only'].set_index('sha256').sort_index()
            combined = sub[sub['features'] == 'url_html'].set_index('sha256').sort_index()
            if not url.index.equals(combined.index):
                raise ValueError('Non-matching paired sample IDs')
            if not np.array_equal(url['label'].to_numpy(), combined['label'].to_numpy()):
                raise ValueError('Paired labels do not match')
            y = url['label'].to_numpy()
            pred_url = url['prediction'].to_numpy()
            pred_both = combined['prediction'].to_numpy()
            domains = url['registered_domain'].to_numpy()
            unique, inverse = np.unique(domains, return_inverse=True)
            group_rows = [np.flatnonzero(inverse == k) for k in range(len(unique))]
            deltas = np.empty(5000, dtype=float)
            for b in range(len(deltas)):
                drawn = rng.integers(0, len(unique), size=len(unique))
                ix = np.concatenate([group_rows[k] for k in drawn])
                deltas[b] = f1_score(y[ix], pred_both[ix], zero_division=0) - f1_score(y[ix], pred_url[ix], zero_division=0)
            point = f1_score(y, pred_both) - f1_score(y, pred_url)
            low, high = np.percentile(deltas, [2.5, 97.5])
            output.append({'subset': subset_name, 'model': model, 'n': len(y),
                           'domains': len(unique), 'delta_f1': point,
                           'ci_low': low, 'ci_high': high, 'bootstrap_resamples': 5000})
    out = RESULTS / 'final_grouped_confidence_intervals.csv'
    if out.exists():
        raise FileExistsError(f'{out} already exists')
    pd.DataFrame(output).to_csv(out, index=False)
    print(pd.DataFrame(output).round(4).to_string(index=False))


if __name__ == '__main__':
    main()
