# CompPhish feature pipeline

This package wraps the supplied original CompPhish extractor; it does not reproduce the user's independent validation implementations. The source files are unchanged. The wrapper corrects the original `presence_of_url_shortener` output name to the dataset's `presence_of_url_shortner` and enforces the original spreadsheet's feature-column ordering.

Install: `python3 -m pip install -r requirements.txt`

Run a smoke test: `python3 smoke_test.py`

Use `from feature_pipeline import extract_features, compute_training_thresholds`.

`extract_features(url, html, mean_phishing_length, mean_domain_length)` accepts the HTML string for the webpage and returns an ordered dict of all feature columns. The original source must remain in `original_extractor` beside the wrapper. Do not use an empty HTML string when evaluating URL+HTML detection; obtain a reliable HTML snapshot or mark the webpage unavailable and define an explicit missing-data policy.

`compute_training_thresholds(training_data)` accepts an iterable of mappings with `url` and `label` fields. Fit thresholds on the CompPhish training partition only, not the independent evaluation set.

The original extractor unpickles the bundled gibberish model, so use only trusted copies of these source files. Its brand and URL-list lookups can be expensive, and its network-sensitive hostname handling inherits the environment's `tldextract` suffix-list version. Record and pin package versions for reported experiments.

The supplied workbook contains 70 feature columns: 47 URL-based and 23 HTML-based, plus `serial_number`, `url`, and `label`. Earlier progress messages mistakenly counted 69 total features.

The wrapper was syntax-checked in this environment, but a full execution and model comparison require installing `tldextract`, `thefuzz`, and `python-Levenshtein` locally. Existing feature disagreements remain unresolved, and passing controlled HTML tests does not prove historical CompPhish equivalence.
