"""Select globally hash-ranked, class-balanced, untouched PhreshPhish test records.

One streaming pass over the entire publisher test split. This can transfer many GB.
Selection only saves metadata; HTML is not retained in memory.
"""
import argparse
import hashlib
import heapq
import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi

from common import ROOT, OUT, load_excluded_samples, normalized_url, registered_domain

REPO = 'phreshphish/phreshphish'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--per-class', type=int, default=1500)
    parser.add_argument('--seed', type=int, default=20261123)
    args = parser.parse_args()
    if args.per_class < 1:
        raise ValueError('per-class must be positive')

    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / 'final_manifest.csv'
    if manifest_path.exists():
        raise FileExistsError(f'{manifest_path} exists; refusing to replace frozen selection')

    excluded_urls, excluded_shas = load_excluded_samples()
    revision = HfApi().dataset_info(REPO).sha
    print('Pinned PhreshPhish revision:', revision, flush=True)
    dataset = load_dataset(REPO, split='test', revision=revision, streaming=True)
    # Hold at most k small metadata records per class; hash rank is deterministic.
    heaps = {'benign': [], 'phish': []}
    seen_sha = set()
    seen_urls = set()
    valid = 0
    for index, row in enumerate(dataset, start=1):
        label = row.get('label')
        url, sha = row.get('url'), row.get('sha256')
        if label not in heaps or not isinstance(url, str) or not isinstance(sha, str):
            continue
        norm = normalized_url(url)
        if not norm or norm in excluded_urls or sha in excluded_shas:
            continue
        if sha in seen_sha or norm in seen_urls:
            continue
        seen_sha.add(sha)
        seen_urls.add(norm)
        valid += 1
        score = int.from_bytes(hashlib.sha256(f'{args.seed}:{sha}'.encode()).digest(), 'big')
        entry = (-score, sha, url, label)
        heap = heaps[label]
        if len(heap) < args.per_class:
            heapq.heappush(heap, entry)
        elif entry[0] > heap[0][0]:
            heapq.heapreplace(heap, entry)
        if index % 10000 == 0:
            print(f'Scanned {index:,} test rows; eligible unique={valid:,}', flush=True)

    if any(len(h) < args.per_class for h in heaps.values()):
        raise ValueError('Insufficient eligible records for one or both classes')

    records = []
    for label, heap in heaps.items():
        for negscore, sha, url, _ in heap:
            records.append({'sha256': sha, 'url': url,
                            'label': 1 if label == 'phish' else 0,
                            'registered_domain': registered_domain(url)})
    records.sort(key=lambda r: (r['label'], r['sha256']))
    pd.DataFrame(records).to_csv(manifest_path, index=False)
    meta = {'repository': REPO, 'split': 'test', 'revision': revision,
            'seed': args.seed, 'per_class': args.per_class,
            'selected': len(records), 'eligible_unique': valid,
            'selection': 'smallest SHA256(seed:sample_sha256) ranks within each class',
            'excluded': 'all available preliminary PhreshPhish sample URLs/SHAs and CompPhish train+validation URLs'}
    (OUT / 'selection_metadata.json').write_text(json.dumps(meta, indent=2))
    print(f'Saved {len(records)} sample IDs to {manifest_path}')
    print('IMPORTANT: Do not change this manifest after looking at evaluation results.')


if __name__ == '__main__':
    main()
