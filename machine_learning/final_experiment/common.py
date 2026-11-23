from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pandas as pd
import tldextract

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'final_experiment'
RESULTS = ROOT / 'results' / 'final_experiment'
PIPE = ROOT / 'phishing_pipeline'


def normalized_url(raw: str) -> str:
    p = urlsplit(str(raw).strip())
    return urlunsplit((p.scheme.lower(), p.netloc.lower(), p.path.rstrip('/') or '/', p.query, ''))


def registered_domain(raw: str) -> str:
    e = tldextract.extract(raw)
    return (f'{e.domain}.{e.suffix}' if e.suffix else e.domain).lower()


def load_schema() -> list[str]:
    schema = json.loads((PIPE / 'feature_schema.json').read_text())
    if len(schema) != 70 or len(set(schema)) != 70:
        raise ValueError('Expected exactly 70 unique feature columns')
    return schema


def load_excluded_samples() -> tuple[set[str], set[str]]:
    urls, shas = set(), set()
    for name in ['phreshphish_sample.csv', 'phreshphish_preliminary.csv',
                 'phreshphish_preliminary_clean.csv', 'phreshphish_domain_disjoint.csv']:
        path = ROOT / 'data' / name
        if not path.exists():
            continue
        for chunk in pd.read_csv(path, usecols=lambda c: c in {'url', 'sha256'}, chunksize=10000):
            if 'url' in chunk:
                urls.update(chunk['url'].dropna().map(normalized_url))
            if 'sha256' in chunk:
                shas.update(chunk['sha256'].dropna().astype(str))
    for name in ['comp_train.csv', 'comp_validation.csv']:
        path = ROOT / 'data' / name
        if not path.exists():
            raise FileNotFoundError(f'Missing required CompPhish split: {path}')
        for chunk in pd.read_csv(path, usecols=['url'], chunksize=10000):
            urls.update(chunk['url'].dropna().map(normalized_url))
    return urls, shas


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
