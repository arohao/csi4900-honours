from feature_pipeline import extract_features

thresholds = {"mean_phishing_length": 52.1861, "mean_domain_length": 26.9864}
html = '<html><head><title>Example</title></head><body><form action="/login"></form><img src="/logo.png"></body></html>'
result = extract_features("https://example.com/login", html, **thresholds)
print(f"Features: {len(result)}")
print({name: result[name] for name in ("url_length", "no_of_forms", "no_of_Images", "presence_of_url_shortner")})
