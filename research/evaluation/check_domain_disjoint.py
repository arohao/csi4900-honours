import pandas as pd
import tldextract
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

train = pd.read_csv(ROOT / "data" / "comp_train.csv")
phresh = pd.read_csv(
    ROOT / "data" / "phreshphish_preliminary_clean.csv"
)

def registered_domain(url):
    ext = tldextract.extract(url)

    if ext.suffix:
        return f"{ext.domain}.{ext.suffix}".lower()

    return ext.domain.lower()

train_domains = set(train["url"].apply(registered_domain))

phresh["registered_domain"] = phresh["url"].apply(
    registered_domain
)

disjoint = phresh[
    ~phresh["registered_domain"].isin(train_domains)
].copy()

print("Original samples:", len(phresh))
print("Domain-disjoint samples:", len(disjoint))
print("Excluded samples:", len(phresh) - len(disjoint))
print("Label distribution:", disjoint["label"].value_counts().to_dict())

disjoint.drop(columns=["registered_domain"]).to_csv(
    ROOT / "data" / "phreshphish_domain_disjoint.csv",
    index=False
)