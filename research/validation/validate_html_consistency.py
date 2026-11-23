
import sys
import json
import numpy as np
import pandas as pd

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "phishing_pipeline"))

from feature_pipeline import extract_features as original_extract
from validate_html_features import extract_html_features as reconstructed_extract

schema = json.loads(
    (ROOT / "phishing_pipeline" / "feature_schema.json").read_text()
)

html_features = schema[47:]

thresholds = json.loads(
    (ROOT / "data" / "training_thresholds.json").read_text()
)

test_cases = [
    {
        "name": "Empty HTML",
        "url": "https://example.com",
        "html": "<html><body></body></html>"
    },
    {
        "name": "Normal webpage",
        "url": "https://example.com",
        "html": """
        <html>
            <head>
                <title>Example Home</title>
                <link rel="icon" href="/favicon.ico">
            </head>
            <body>
                <header>Example</header>
                <a href="/about">About</a>
                <img src="/logo.png">
                <footer>© Example 2026</footer>
            </body>
        </html>
        """
    },
    {
        "name": "Phishing-style login",
        "url": "https://secure-login-example.net",
        "html": """
        <html>
            <head><title>Bank Login</title></head>
            <body>
                <header>Welcome to Bank</header>
                <form action="https://other.com/login">
                    <input type="text">
                    <input type="password">
                </form>
                <script>eval("test")</script>
                <div style="display:none">Hidden</div>
            </body>
        </html>
        """
    },
    {
        "name": "Mixed links",
        "url": "https://example.com/account",
        "html": """
        <html><body>
            <a href="/home">Internal</a>
            <a href="https://other.com">External</a>
            <a href="">Empty</a>
            <a href="#">Self</a>
            <a href="javascript:void(0)">Null</a>
            <img src="/logo.png">
            <form action="/login"></form>
            <form action="https://other.com/login"></form>
        </body></html>
        """
    },
    {
        "name": "Suspicious HTML",
        "url": "https://example.com",
        "html": """
        <html><body>
            <a href="/download/setup.exe">Download</a>
            <a href="https://bit.ly/test">Short URL</a>
            <script src="https://external.com/script.js"></script>
            <button onclick="alert(1)">Click</button>
            <div class="hidden">Hidden content</div>
        </body></html>
        """
    },
    {
        "name": "Brand mismatch",
        "url": "https://example.com",
        "html": """
        <html>
            <head>
                <title>Another Company</title>
                <link rel="icon"
                      href="https://other.com/favicon.ico">
            </head>
            <body>
                <header>Another Company</header>
                <footer>© Another Company</footer>
                <a href="https://facebook.com/othercompany">
                    Facebook
                </a>
            </body>
        </html>
        """
    },
    {
        "name": "Relative script",
        "url": "https://example.com",
        "html": """
        <html><body>
            <script src="/js/app.js"></script>
        </body></html>
        """
    },
    {
        "name": "Missing title",
        "url": "https://example.com",
        "html": "<html><head></head><body>Hello</body></html>"
    }
]

results = []

for test in test_cases:
    name = test["name"]
    url = test["url"]
    html = test["html"]

    try:
        original = original_extract(
            url=url,
            html=html,
            **thresholds
        )

        reconstructed = reconstructed_extract(html, url)

        for feature in html_features:
            expected = original[feature]
            actual = reconstructed[feature]

            match = bool(np.isclose(
                float(expected),
                float(actual),
                rtol=1e-9,
                atol=1e-12
            ))

            results.append({
                "Test": name,
                "Feature": feature,
                "Original": expected,
                "Reconstructed": actual,
                "Match": match
            })

    except Exception as e:
        print(f"ERROR in {name}: {type(e).__name__}: {e}")
        raise

results_df = pd.DataFrame(results)

print("\nHTML Feature Consistency Results:\n")

for feature in html_features:
    subset = results_df[
        results_df["Feature"] == feature
    ]

    matches = subset["Match"].sum()
    total = len(subset)

    print(
        f"{feature}: "
        f"{matches / total * 100:.2f}% agreement "
        f"({total - matches} mismatches)"
    )

print("\nOverall Results:")
print("Total comparisons:", len(results_df))
print("Total mismatches:", (~results_df["Match"]).sum())

mismatches = results_df[
    ~results_df["Match"]
]

if not mismatches.empty:
    print("\nMismatched features:")
    print(mismatches.to_string(index=False))

output = ROOT / "results" / "html_consistency_results.csv"
output.parent.mkdir(parents=True, exist_ok=True)

results_df.to_csv(output, index=False)

print(f"\nSaved to {output}")
