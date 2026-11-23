import pandas as pd
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[1]

train = pd.read_csv(ROOT / "data" / "comp_train.csv")
phresh = pd.read_csv(ROOT / "data" / "phreshphish_preliminary.csv")

def normalize_url(url):
    parsed = urlsplit(str(url).strip())

    return urlunsplit((
        parsed.scheme.lower(),
        parsed.netloc.lower(),
        parsed.path.rstrip("/") or "/",
        parsed.query,
        ""
    ))

train_urls = set(train["url"].map(normalize_url))

overlap_mask = phresh["url"].map(normalize_url).isin(train_urls)

print("Overlapping records:", overlap_mask.sum())

clean = phresh.loc[~overlap_mask].copy()

print("Remaining records:", len(clean))
print("Label distribution:", clean["label"].value_counts().to_dict())

clean.to_csv(
    ROOT / "data" / "phreshphish_preliminary_clean.csv",
    index=False
)