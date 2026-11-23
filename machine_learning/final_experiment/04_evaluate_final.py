"""One-shot external evaluation of frozen models; URL-disjoint and domain-disjoint reports."""
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score)

from common import ROOT, OUT, RESULTS, load_schema, normalized_url, registered_domain, digest_file


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    output = RESULTS / 'final_metrics.csv'
    if output.exists():
        raise FileExistsError(f'{output} already exists; frozen evaluation cannot be overwritten')
    manifest = pd.read_csv(OUT / 'final_manifest.csv', dtype={'sha256': str})
    df = pd.read_csv(OUT / 'phreshphish_final.csv', dtype={'sha256': str})
    frozen_dir = RESULTS / 'frozen_models'
    meta = json.loads((frozen_dir / 'freeze_metadata.json').read_text())
    schema = load_schema()
    if meta['schema'] != schema:
        raise ValueError('Frozen model feature schema does not match')
    if not df['sha256'].is_unique or set(df['sha256']) != set(manifest['sha256']):
        raise ValueError('Final extracted IDs do not match the frozen manifest')
    if len(df) != len(manifest):
        raise ValueError('Final extracted sample count differs from manifest')
    if not (df.set_index('sha256')['label'].sort_index() == manifest.set_index('sha256')['label'].sort_index()).all():
        raise ValueError('Final labels differ from manifest')
    if not np.isfinite(df[schema].to_numpy(dtype=float)).all():
        raise ValueError('Invalid or missing numeric feature values')
    train_path = ROOT / 'data' / 'comp_train.csv'
    val_path = ROOT / 'data' / 'comp_validation.csv'
    if digest_file(train_path) != meta['train_sha256'] or digest_file(val_path) != meta['validation_sha256']:
        raise ValueError('CompPhish split files have changed since model freezing')
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    known_urls = set(train['url'].map(normalized_url)) | set(val['url'].map(normalized_url))
    known_domains = set(train['url'].map(registered_domain))
    if df['url'].map(normalized_url).isin(known_urls).any():
        raise ValueError('Exact URL overlap with CompPhish detected: stop evaluation')
    df['registered_domain'] = df['url'].map(registered_domain)
    df['domain_disjoint'] = ~df['registered_domain'].isin(known_domains)
    # Primary analysis uses all newly selected, exactly URL-disjoint external records.
    # Secondary analysis excludes registered domains present in CompPhish training.
    subsets = {'all_external': df, 'domain_disjoint': df[df['domain_disjoint']].copy()}
    rows, predictions = [], []
    for fs_name, cols in [('url_only', schema[:47]), ('url_html', schema)]:
        for model_name in ['logistic_regression', 'random_forest']:
            key = f'{fs_name}_{model_name}'
            model = joblib.load(frozen_dir / f'{key}.joblib')
            proba_all = model.predict_proba(df[cols])[:, 1]
            pred_all = (proba_all >= 0.5).astype(int)
            for i, item in enumerate(df.itertuples(index=False)):
                predictions.append({'sha256': item.sha256, 'url': item.url,
                                    'registered_domain': item.registered_domain,
                                    'domain_disjoint': item.domain_disjoint,
                                    'label': item.label, 'model': model_name,
                                    'features': fs_name, 'prediction': int(pred_all[i]),
                                    'probability': float(proba_all[i])})
            for subset_name, subset in subsets.items():
                indices = df.index.get_indexer(subset.index)
                y = subset['label'].to_numpy()
                if len(np.unique(y)) != 2:
                    raise ValueError(f'{subset_name} lacks both classes')
                pred, proba = pred_all[indices], proba_all[indices]
                rows.append({'subset': subset_name, 'features': fs_name, 'model': model_name,
                             'n': len(y), 'phish_n': int(y.sum()),
                             'accuracy': accuracy_score(y, pred),
                             'precision': precision_score(y, pred, zero_division=0),
                             'recall': recall_score(y, pred, zero_division=0),
                             'f1': f1_score(y, pred, zero_division=0),
                             'roc_auc': roc_auc_score(y, proba),
                             'pr_auc': average_precision_score(y, proba)})
    pd.DataFrame(predictions).to_csv(RESULTS / 'final_predictions.csv', index=False)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print('Saved:', output)


if __name__ == '__main__':
    main()
