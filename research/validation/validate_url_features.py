import math
import re
import tldextract
import ipaddress
import pandas as pd
import numpy as np
from collections import Counter
from urllib.parse import urlparse, parse_qs
from pathlib import Path
from pathlib import Path

RISKY_EXTENSIONS = [
    ".exe", ".msi", ".bat", ".cmd", ".vbs", ".ps1",
    ".jar", ".scr", ".app", ".dmg", ".pkg", ".sh",
    ".run", ".bin", ".py", ".air", ".apk", ".elf",
    ".zip", ".rar", ".7z"
]

SPECIAL_CHARS = [
    "@", ",", "$", "%20", "&", "=", "%", "?", "_", "-", ";"
]

def extract_domain(url):
    domain = urlparse(url).netloc
    if domain.startswith("www."):
        domain = domain[4:]
    return domain

def get_hostname(url):
    extracted = tldextract.extract(url)
    return ".".join(
        part for part in [
            extracted.subdomain,
            extracted.domain,
            extracted.suffix
        ] if part
    ).lower()

def has_downloadable_extension(url):
    return int(any(ext in url.lower() for ext in RISKY_EXTENSIONS))

def has_port(url):
    return int(bool(urlparse(url).port))

def tokenize_url(url):
    return re.split(r"\W+", url)

ROOT = Path(__file__).resolve().parents[1]
EXTRACTOR_DIR = ROOT / "original_extractor"

def read_reference_file(filename):
    with open(EXTRACTOR_DIR / filename, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

SHORTENED_DOMAINS = read_reference_file("shortUrls.txt")
FREE_HOSTING_DOMAINS = read_reference_file("free_hosting_services.txt")

def is_shortened_url(url):
    domain = urlparse(url).netloc.lower()
    return int(domain in SHORTENED_DOMAINS)

def is_free_hosting(url):
    domain = urlparse(url).netloc.lower()
    return int(any(domain.endswith(free) for free in FREE_HOSTING_DOMAINS))

def url_contains_ip(url):
    try:
        hostname = get_hostname(url)

        if hostname is None:
            return 0

        ipv4_pattern = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")
        hex_ipv4_pattern = re.compile(r"^(0x[0-9a-fA-F]{1,2}\.){3}0x[0-9a-fA-F]{1,2}$")
        int_ip_pattern = re.compile(r"^\d{8,10}$")
        ipv6_pattern = re.compile(r"^([a-fA-F0-9:]+)$")

        return int(any([
            ipv4_pattern.match(hostname),
            hex_ipv4_pattern.match(hostname),
            int_ip_pattern.match(hostname),
            ipv6_pattern.match(hostname)
        ]))

    except Exception:
        return 0


def extract_features(url, mean_phishing_length, mean_domain_length):
    parsed = urlparse(url)
    hostname = get_hostname(url)
    words_raw = tokenize_url(url)
    char_counts = Counter(url)

    url_entropy = -sum(
        (count / len(url)) * math.log2(count / len(url))
        for count in char_counts.values()
    ) if url else 0.0

    double_slash_positions = [
        m.start() for m in re.finditer("//", url)
    ]

    features = {
        "url_length": len(url),
        "domain_length": len(extract_domain(url)),
        "dots_path": parsed.path.count("."),
        "no_of_at": url.count("@"),
        "no_of_comma": url.count(","),
        "no_of_dollar": url.count("$"),
        "no_of_semicolon": url.count(";"),
        "no_of_space": url.count(" ") + url.count("@"),
        "no_of_ampersand": url.count("&"),
        "no_of_dots": url.count("."),
        "no_of_equal": url.count("="),
        "no_of_percent": url.count("%"),
        "no_of_underscore": url.count("_"),
        "no_of_hyphen": url.count("-"),
        "no_of_questionmark": url.count("?"),
        "no_of_colon": url.count(":"),
        "no_of_slashes_inpath": parsed.path.count("/"),
        "special_characters_count": sum(
            url.count(c) for c in SPECIAL_CHARS
        ),
        "has_asterisk": int("*" in url),
        "has_or": int("|" in url),
        "has_embedded_url": int(
            url.count("http://") + url.count("https://") > 1
        ),
        "uses_https": int(parsed.scheme == "https"),
        "num_query_params": len(parse_qs(parsed.query)),

        "numeric_proportion": (
            sum(c.isdigit() for c in url) / len(url)
            if url else 0
        ),
        "url_entropy": url_entropy,
        "hostname_digit_ratio": (
            len(re.findall(r"[0-9]", hostname)) / len(hostname)
            if hostname else 0
        ),
        "has_redirections": int(
            len(double_slash_positions) > 1
            and double_slash_positions[-1] > 6
        ),
        "has_punycode": int("xn--" in hostname),
        "count_www": sum(
            "www" in word for word in re.split(r"\W+", url)
        ),
        "count_com": sum(
            "com" in word for word in re.split(r"\W+", url)
        ),

        "url_has_high_risk_extension": has_downloadable_extension(url),
        "has_explicit_port": has_port(url),
        "total_words_url": len(words_raw),
        "average_length_of_words": (
            sum(len(word) for word in words_raw) / len(words_raw)
            if words_raw else 0
        ),
        "longest_word_length": max(
            map(len, words_raw), default=0
        ),
        "shortest_word_length": min(
            map(len, words_raw), default=0
        ),
    }

    features.update({
        "url_contains_ip": url_contains_ip(url),
        "presence_of_url_shortner": is_shortened_url(url),
        "presence_of_free_hosting": is_free_hosting(url),
        "url_more_than_mean": int(len(url) >= mean_phishing_length),
        "domain_more_than_mean": int(
            len(extract_domain(url)) >= mean_domain_length
        ),
    })

    return features

ROOT = Path(__file__).resolve().parents[1]

df = pd.read_excel(ROOT / "data" / "All_Features_threshold90.xlsx")

phishing_df = df[df["label"] == 1]

mean_phishing_length = phishing_df["url"].apply(len).mean()

mean_domain_length = phishing_df["url"].apply(
    lambda url: len(extract_domain(url))
).mean()

print(f"Mean phishing URL length: {mean_phishing_length:.4f}")
print(f"Mean phishing domain length: {mean_domain_length:.4f}")

predictions = df["url"].apply(
    lambda url: extract_features(
        url,
        mean_phishing_length,
        mean_domain_length
    )
).apply(pd.Series)

for feature in predictions.columns:
    if feature not in df.columns:
        print(f"{feature}: SKIPPED (column not found)")
        continue

    matches = np.isclose(
        predictions[feature].astype(float),
        df[feature].astype(float),
        rtol=1e-9,
        atol=1e-12
    )

    print(
        f"{feature}: {matches.mean() * 100:.2f}% agreement "
        f"({(~matches).sum()} mismatches)"
    )

print(df.columns.tolist())

def find_best_threshold(df, length_column, feature_column):
    lengths = df[length_column].to_numpy()
    actual = df[feature_column].to_numpy()

    best_threshold = None
    best_mismatches = len(df) + 1

    for threshold in sorted(set(lengths)):
        predicted = (lengths >= threshold).astype(int)
        mismatches = (predicted != actual).sum()

        if mismatches < best_mismatches:
            best_threshold = threshold
            best_mismatches = mismatches

    print(f"\n{feature_column}")
    print(f"Best threshold: {best_threshold}")
    print(f"Mismatches: {best_mismatches}")
    print(f"Agreement: {(1 - best_mismatches / len(df)) * 100:.2f}%")

find_best_threshold(
    df,
    "url_length",
    "url_more_than_mean"
)

find_best_threshold(
    df,
    "domain_length",
    "domain_more_than_mean"
)

for feature, threshold, length_col in [
    ("url_more_than_mean", 50, "url_length"),
    ("domain_more_than_mean", 27, "domain_length")
]:
    predicted = (df[length_col] >= threshold).astype(int)
    mismatches = df[predicted != df[feature]].copy()

    print(f"\n{feature}")
    print("Mismatch labels:")
    print(mismatches["label"].value_counts())

    print("\nMismatch examples:")
    print(
        mismatches[
            ["url", "label", length_col, feature]
        ].head(10).to_string(index=False)
    )