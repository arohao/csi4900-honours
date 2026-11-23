import pandas as pd
import tldextract
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[1]

comp = pd.read_csv(ROOT / "data" / "comp_train.csv")
phresh = pd.read_csv(ROOT / "data" / "phreshphish_preliminary.csv")

def normalize_url(url):
    url = str(url).strip()
    parsed = urlsplit(url)

    return urlunsplit((
        parsed.scheme.lower(),
        parsed.netloc.lower(),
        parsed.path.rstrip("/") or "/",
        parsed.query,
        ""
    ))

comp_urls = set(comp["url"].map(normalize_url))
phresh_urls = set(phresh["url"].map(normalize_url))

overlap = comp_urls & phresh_urls

print("CompPhish unique URLs:", len(comp_urls))
print("PhreshPhish unique URLs:", len(phresh_urls))
print("Overlapping URLs:", len(overlap))

for url in list(overlap)[:10]:
    print(url)

def registered_domain(url):
    ext = tldextract.extract(url)

    if ext.suffix:
        return f"{ext.domain}.{ext.suffix}".lower()

    return ext.domain.lower()

comp_domains = set(comp["url"].map(registered_domain))
phresh_domains = set(phresh["url"].map(registered_domain))

domain_overlap = comp_domains & phresh_domains

print("\nCompPhish registered domains:", len(comp_domains))
print("PhreshPhish registered domains:", len(phresh_domains))
print("Overlapping registered domains:", len(domain_overlap))

print("\nExamples:")
for domain in sorted(domain_overlap)[:15]:
    print(domain)