"""Freeze the same four previously used configurations without consulting final data."""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

from common import ROOT, OUT, RESULTS, load_schema, digest_file


def main():
    out = RESULTS / 'frozen_models'
    if out.exists():
        raise FileExistsError(f'{out} exists; do not silently replace frozen models')
    out.mkdir(parents=True)
    train_path = ROOT / 'data' / 'comp_train.csv'
    val_path = ROOT / 'data' / 'comp_validation.csv'
    train, val = pd.read_csv(train_path), pd.read_csv(val_path)
    schema = load_schema()
    feature_sets = {'url_only': schema[:47], 'url_html': schema}
    configs = {
        'logistic_regression': lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42)),
        'random_forest': lambda: RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    }
    rows = []
    for fs_name, columns in feature_sets.items():
        for model_name, constructor in configs.items():
            model = constructor()
            model.fit(train[columns], train['label'])
            pred = model.predict(val[columns])
            proba = model.predict_proba(val[columns])[:, 1]
            name = f'{fs_name}_{model_name}'
            joblib.dump(model, out / f'{name}.joblib')
            rows.append({'model': name, 'validation_f1': f1_score(val['label'], pred),
                         'validation_roc_auc': roc_auc_score(val['label'], proba),
                         'validation_pr_auc': average_precision_score(val['label'], proba)})
            print(name, rows[-1])
    (out / 'freeze_metadata.json').write_text(json.dumps({
        'train_sha256': digest_file(train_path), 'validation_sha256': digest_file(val_path),
        'schema': schema, 'sklearn_version': sklearn.__version__,
        'threshold': 0.5, 'model_settings': 'LR StandardScaler max_iter=2000; RF n_estimators=300; random_state=42',
        'validation_metrics': rows
    }, indent=2))
    print('Frozen artifacts saved at:', out)


if __name__ == '__main__':
    main()
