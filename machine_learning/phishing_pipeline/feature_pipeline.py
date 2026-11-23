from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ORIGINAL_DIR = ROOT / "original_extractor"
SCHEMA_PATH = ROOT / "feature_schema.json"


def load_original():
    sys.path.insert(0, str(ORIGINAL_DIR))
    try:
        import os
        previous = Path.cwd()
        try:
            os.chdir(ORIGINAL_DIR)
            module = importlib.import_module("feature_Generator_with_brands")
        finally:
            os.chdir(previous)
    finally:
        sys.path.remove(str(ORIGINAL_DIR))

    module.file_path_su = str(ORIGINAL_DIR / "shortUrls.txt")
    module.file_path_fhs = str(ORIGINAL_DIR / "free_hosting_services.txt")
    return module


_original = None


def extract_features(url: str, html: str, mean_phishing_length: float, mean_domain_length: float) -> dict:
    global _original
    if _original is None:
        _original = load_original()
    if not isinstance(url, str) or not isinstance(html, str):
        raise TypeError("url and html must be strings")
    values = _original.extract_features(
        serial_number=0,
        url=url,
        label=0,
        html=html,
        mean_phishing_length=mean_phishing_length,
        mean_domain_length=mean_domain_length,
    )
    values["presence_of_url_shortner"] = values.pop("presence_of_url_shortener")
    expected = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    actual = set(values) - {"serial_number", "url", "label"}
    if actual != set(expected):
        raise ValueError(f"Feature schema mismatch: missing={set(expected)-actual}, extra={actual-set(expected)}")
    return {column: values[column] for column in expected}


def compute_training_thresholds(training_data):
    global _original
    if _original is None:
        _original = load_original()
    phishing_urls = [str(row["url"]) for row in training_data if int(row["label"]) == 1]
    if not phishing_urls:
        raise ValueError("Training data must contain phishing-labelled URLs")
    return {
        "mean_phishing_length": sum(len(u) for u in phishing_urls) / len(phishing_urls),
        "mean_domain_length": sum(len(_original.extract_domain(u)) for u in phishing_urls) / len(phishing_urls),
    }
