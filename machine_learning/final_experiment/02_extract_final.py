"""Second streamed pass; extract only selected frozen sample IDs into 70-feature CSV."""
import csv
import json
import sys

import pandas as pd
from datasets import load_dataset

from common import ROOT, OUT, PIPE, load_schema, digest_file

sys.path.insert(0, str(PIPE))
from feature_pipeline import extract_features


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / 'final_manifest.csv'
    out_path = OUT / 'phreshphish_final.csv'
    if out_path.exists():
        raise FileExistsError(f'{out_path} already exists; refusing to overwrite frozen data')
    selected = pd.read_csv(manifest_path, dtype={'sha256': str})
    meta = json.loads((OUT / 'selection_metadata.json').read_text())
    thresholds = json.loads((ROOT / 'data' / 'training_thresholds.json').read_text())
    schema = load_schema()
    wanted = dict(zip(selected['sha256'], selected['url']))
    done = set()
    failures = []
    dataset = load_dataset(meta['repository'], split=meta['split'],
                           revision=meta['revision'], streaming=True)
    with out_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['sha256', 'url', 'label'] + schema)
        writer.writeheader()
        for rownum, row in enumerate(dataset, start=1):
            sha = row.get('sha256')
            if sha not in wanted or sha in done:
                continue
            done.add(sha)
            try:
                if row['url'] != wanted[sha]:
                    raise ValueError('URL differs from frozen manifest')
                if not isinstance(row['html'], str):
                    raise ValueError('Missing HTML string')
                features = extract_features(row['url'], row['html'], **thresholds)
                if set(features) != set(schema):
                    raise ValueError('Wrong feature schema')
                writer.writerow({'sha256': sha, 'url': row['url'],
                                 'label': 1 if row['label'] == 'phish' else 0,
                                 **features})
            except Exception as exc:
                failures.append({'sha256': sha, 'error': f'{type(exc).__name__}: {exc}'})
            if len(done) % 100 == 0:
                print(f'Attempted {len(done)}/{len(wanted)} selected records', flush=True)
            if len(done) == len(wanted):
                break
    (OUT / 'extraction_failures.json').write_text(json.dumps(failures, indent=2))
    if len(done) != len(wanted) or failures:
        raise RuntimeError(f'Incomplete extraction: found {len(done)}/{len(wanted)}, errors={len(failures)}. '
                           'Do not evaluate this sample without accounting for the missing records.')
    extracted = pd.read_csv(out_path)
    assert len(extracted) == len(wanted)
    assert not extracted[schema].isna().any().any()
    print('Extracted:', len(extracted), 'labels:', extracted['label'].value_counts().to_dict())
    print('Output SHA256:', digest_file(out_path))


if __name__ == '__main__':
    main()
