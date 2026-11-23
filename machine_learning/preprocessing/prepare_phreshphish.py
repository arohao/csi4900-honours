
import sys
import csv
import json
from pathlib import Path
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "phishing_pipeline"))
from feature_pipeline import extract_features

SCHEMA = json.loads(
    (ROOT / "phishing_pipeline" / "feature_schema.json").read_text()
)

OUTPUT = ROOT / "data" / "phreshphish_preliminary.csv"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

dataset = load_dataset(
    "phreshphish/phreshphish",
    split="test",
    streaming=True
).shuffle(seed=42, buffer_size=100)

thresholds = {}

with open(ROOT / "data" / "training_thresholds.json") as f:
    thresholds = json.load(f)

counts = {"benign": 0, "phish": 0}
limits = {"benign": 500, "phish": 500}

with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["url", "label", "sha256"] + SCHEMA
    )
    writer.writeheader()

    for row in dataset:
        label = row["label"]

        if label not in counts or counts[label] >= limits[label]:
            continue

        try:
            url = row["url"]
            html = row["html"]

            if not isinstance(url, str) or not isinstance(html, str):
                continue

            features = extract_features(
                url=url,
                html=html,
                **thresholds
            )

            record = {
                "url": url,
                "label": 1 if label == "phish" else 0,
                "sha256": row["sha256"]
            }

            record.update(features)

            writer.writerow(record)
            counts[label] += 1
            if sum(counts.values()) % 50 == 0:
                print(
                    f"Processed {sum(counts.values())}/100 "
                    f"(benign={counts['benign']}, phish={counts['phish']})",
                    flush=True
                )

        except Exception as e:
            print(f"Skipped record: {type(e).__name__}: {e}")

        if all(counts[k] >= limits[k] for k in counts):
            break

print(f"Saved to {OUTPUT}")
print(f"Final counts: {counts}")