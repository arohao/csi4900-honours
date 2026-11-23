import os
import re
import tldextract
from pathlib import Path
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup

def get_hostname(url):
    extracted = tldextract.extract(url)

    return ".".join(
        part for part in [
            extracted.subdomain,
            extracted.domain,
            extracted.suffix
        ] if part
    ).lower()

ROOT = Path(__file__).resolve().parents[1]
EXTRACTOR_DIR = ROOT / "original_extractor"

RISKY_EXTENSIONS = {
    ".exe", ".msi", ".bat", ".cmd", ".vbs", ".ps1",
    ".jar", ".scr", ".app", ".dmg", ".pkg", ".sh",
    ".run", ".bin", ".py", ".air", ".apk", ".elf",
    ".zip", ".rar", ".7z"
}

with open(EXTRACTOR_DIR / "shortUrls.txt", encoding="utf-8") as f:
    SHORTENED_DOMAINS = {
        line.strip() for line in f if line.strip()
    }

def contains_executable_link(soup, url):
    tags = soup.find_all(["a", "link", "script", "img", "iframe", "form"])

    for tag in tags:
        attr = tag.get("href") or tag.get("src") or tag.get("action")

        if not attr:
            continue

        full_url = urljoin(url, attr.strip())
        path = urlparse(full_url).path
        ext = os.path.splitext(path)[1].lower()

        if ext in RISKY_EXTENSIONS:
            return 1

    return 0


def is_shortened_url(url):
    return int(urlparse(url).netloc.lower() in SHORTENED_DOMAINS)


def has_hidden_elements(soup):
    hidden_flag = 0

    for tag in soup.find_all(style=True):
        style = tag["style"].lower()

        if "display:none" in style or "visibility:hidden" in style:
            hidden_flag = 1

    for tag in soup.find_all(class_=True):
        if any(
            cls.lower() in ["hidden", "hide", "invisible"]
            for cls in tag.get("class", [])
        ):
            hidden_flag = 1

    return hidden_flag


def external_script_detected(soup, url):
    hostname = urlparse(url).hostname or ""

    return int(any(
        urlparse(script.get("src", "")).netloc not in hostname
        for script in soup.find_all("script")
        if script.get("src")
    ))


def detect_suspicious_js(html):
    patterns = [
        r"eval\s*\(",
        r"atob\s*\(",
        r"(\+.*?){3,}",
        r"addEventListener\s*\(\s*['\"]keydown['\"]",
        r"window\.location",
        r"setTimeout\s*\(",
        r"setInterval\s*\("
    ]

    return int(any(re.search(pattern, html) for pattern in patterns))

def extract_branding_features(soup, url):
    hostname = get_hostname(url)
    parts = hostname.split(".")
    domain_core = parts[-2] if len(parts) >= 2 else parts[0]

    footer = soup.find("footer")
    footer_text = footer.get_text(strip=True).lower() if footer else ""

    header = soup.find("header")
    header_text = header.get_text(strip=True).lower() if header else ""

    is_footer_mismatch = int(
        bool(footer_text) and domain_core not in footer_text
    )

    is_header_mismatch = int(
        bool(header_text) and domain_core not in header_text
    )

    content_text = soup.get_text(" ", strip=True)
    symbols = r"(\u00A9|\u2122|\u00AE)"
    matches = list(re.finditer(symbols, content_text))

    copyright_mismatch = 1

    for match in matches:
        start = max(0, match.start() - 50)
        end = match.end() + 50
        window = content_text[start:end].lower()

        if domain_core in window:
            copyright_mismatch = 0
            break

    social_domains = [
        "facebook.com",
        "twitter.com",
        "linkedin.com",
        "instagram.com"
    ]

    social_links = [
        tag.get("href", "")
        for tag in soup.find_all("a", href=True)
    ]

    found_social = any(
        any(sd in link for sd in social_domains)
        for link in social_links
    )

    matching_links = [
        link for link in social_links
        if domain_core in link
    ]

    mismatched_social = int(
        found_social and len(matching_links) == 0
    )

    base_domain = urlparse(url).netloc.lower()

    favicon_links = soup.find_all(
        "link",
        rel=lambda r: r and "icon" in r.lower()
    )

    favicon_mismatch = 0

    for link in favicon_links:
        href = link.get("href", "")
        full_url = urljoin(url, href)

        if urlparse(full_url).netloc.lower() != base_domain:
            favicon_mismatch = 1
            break

    title_tag = (
        soup.title.string.strip().lower()
        if soup.title and soup.title.string
        else ""
    )

    generic_titles = {"index", "home", "page", ""}

    inline_handlers = int(any(
        tag.get("onclick") or tag.get("onmouseover")
        for tag in soup.find_all(["a", "button"])
    ))

    external_form = int(any(
        urlparse(urljoin(url, form.get("action", ""))).netloc.lower()
        != hostname
        for form in soup.find_all("form")
        if form.get("action")
    ))

    return {
        "is_footer_mismatch": is_footer_mismatch,
        "is_header_mismatch": is_header_mismatch,
        "is_copyright_mismatch": copyright_mismatch,
        "is_socials_mismatch": mismatched_social,
        "is_favicon_mismatch": favicon_mismatch,
        "has_missing_title": int(title_tag in generic_titles),
        "title_mismatch_with_domain": int(domain_core not in title_tag),
        "has_inline_event_handlers": inline_handlers,
        "form_action_external": external_form
    }

def extract_html_features(html, url):
    soup = BeautifulSoup(html, "html.parser")

    href_links = soup.find_all(href=True)
    src_links = soup.find_all(src=True)
    form_links = soup.find_all("form")

    empty_patterns = {"", "null", "javascript:void(0)"}
    self_pattern = "#"

    base_domain = urlparse(url).netloc

    internal_count = 0
    external_count = 0
    empty_count = 0
    self_count = 0

    tags = href_links + src_links

    for tag in tags:
        link = tag.get("href") or tag.get("src") or ""
        link_clean = link.strip().lower()

        if link_clean == self_pattern:
            self_count += 1
            continue

        elif link_clean in empty_patterns:
            empty_count += 1
            continue

        full_url = urljoin(url, link_clean)
        netloc = urlparse(full_url).netloc.lower()

        if netloc == base_domain:
            internal_count += 1
        else:
            external_count += 1

    for form in form_links:
        action = form.get("action", "").strip().lower()

        if action == self_pattern:
            self_count += 1
            continue

        elif action in empty_patterns or action == "":
            empty_count += 1
            continue

        full_url = urljoin(url, action)
        netloc = urlparse(full_url).netloc.lower()

        if netloc == base_domain:
            internal_count += 1
        else:
            external_count += 1

    total_links = (
        internal_count
        + external_count
        + empty_count
        + self_count
    )

    shortened_count = 0

    for tag in href_links + src_links:
        link = tag.get("href") or tag.get("src") or ""
        link_clean = link.strip().lower()

        if link_clean in {"", "null", "javascript:void(0)", "#"}:
            continue

        full_url = urljoin(url, link_clean)
        shortened_count += is_shortened_url(full_url)

    for form in form_links:
        action = form.get("action", "").strip().lower()

        if action in {"", "null", "javascript:void(0)", "#"}:
            continue

        full_url = urljoin(url, action)
        shortened_count += is_shortened_url(full_url)

    features = {
        "no_of_forms": len(form_links),
        "no_of_Images": len(soup.find_all("img")),
        "no_of_hyperlinks": total_links,
        "no_of_external_links": external_count,
        "no_of_internal_links": internal_count,
        "no_of_null_links": empty_count + self_count,
        "no_of_selfreference_links": self_count,
        "no_of_empty_links": empty_count,
        "EmptyToExternalLinksRatio": (
            empty_count / external_count
            if external_count else 0
        ),
        "Risky_extensions_In_Html": contains_executable_link(soup, url),
        "presence_of_shorten_url_html": int(shortened_count > 0),
        "hidden_elements_html": has_hidden_elements(soup),
        "Script_loaded_from_ext_domain": external_script_detected(soup, url),
        "suspicious_inline_js": detect_suspicious_js(html)
    }
    features.update(extract_branding_features(soup, url))
    return features

test_cases = [
    {
        "name": "Empty page",
        "html": "<html><body></body></html>",
        "url": "https://example.com",
        "expected": {
            "no_of_forms": 0,
            "no_of_Images": 0,
            "no_of_hyperlinks": 0,
            "no_of_external_links": 0,
            "no_of_internal_links": 0,
            "no_of_null_links": 0,
            "no_of_selfreference_links": 0,
            "no_of_empty_links": 0,
            "EmptyToExternalLinksRatio": 0
        }
    },
    {
        "name": "Mixed links",
        "html": """
        <html>
            <body>
                <a href="/home">Home</a>
                <a href="https://other.com">External</a>
                <a href="">Empty</a>
                <a href="#">Self</a>
                <img src="/logo.png">
                <form action="https://other.com/login"></form>
                <form></form>
            </body>
        </html>
        """,
        "url": "https://example.com/account",
        "expected": {
            "no_of_forms": 2,
            "no_of_Images": 1,
            "no_of_hyperlinks": 7,
            "no_of_external_links": 2,
            "no_of_internal_links": 2,
            "no_of_null_links": 3,
            "no_of_selfreference_links": 1,
            "no_of_empty_links": 2,
            "EmptyToExternalLinksRatio": 1.0
        }
    },
    {
        "name": "Internal links only",
        "html": """
        <html>
            <body>
                <a href="/about">About</a>
                <a href="/contact">Contact</a>
                <img src="/image.png">
            </body>
        </html>
        """,
        "url": "https://example.com",
        "expected": {
            "no_of_forms": 0,
            "no_of_Images": 1,
            "no_of_hyperlinks": 3,
            "no_of_external_links": 0,
            "no_of_internal_links": 3,
            "no_of_null_links": 0,
            "no_of_selfreference_links": 0,
            "no_of_empty_links": 0,
            "EmptyToExternalLinksRatio": 0
        }
    }
]

for test in test_cases:
    result = extract_html_features(test["html"], test["url"])

    passed = all(
        result.get(feature) == expected
        for feature, expected in test["expected"].items()
    )

    print(f"{test['name']}: {'PASS' if passed else 'FAIL'}")

    if not passed:
        print("Expected:", test["expected"])
        print("Actual:", result)

suspicious_tests = [
    {
        "name": "Risky executable",
        "html": '<a href="/downloads/setup.exe">Download</a>',
        "url": "https://example.com",
        "feature": "Risky_extensions_In_Html",
        "expected": 1
    },
    {
        "name": "Shortened URL",
        "html": '<a href="https://bit.ly/example">Click</a>',
        "url": "https://example.com",
        "feature": "presence_of_shorten_url_html",
        "expected": 1
    },
    {
        "name": "Hidden element",
        "html": '<div style="display:none">Hidden</div>',
        "url": "https://example.com",
        "feature": "hidden_elements_html",
        "expected": 1
    },
    {
        "name": "External script",
        "html": '<script src="https://external.com/app.js"></script>',
        "url": "https://example.com",
        "feature": "Script_loaded_from_ext_domain",
        "expected": 1
    },
    {
        "name": "Suspicious JavaScript",
        "html": '<script>eval("test")</script>',
        "url": "https://example.com",
        "feature": "suspicious_inline_js",
        "expected": 1
    },
    {
        "name": "Normal HTML",
        "html": "<html><body><p>Hello</p></body></html>",
        "url": "https://example.com",
        "feature": None,
        "expected": 0
    }
]

for test in suspicious_tests:
    result = extract_html_features(test["html"], test["url"])

    if test["feature"] is None:
        actual = sum(
            result[feature]
            for feature in [
                "Risky_extensions_In_Html",
                "presence_of_shorten_url_html",
                "hidden_elements_html",
                "Script_loaded_from_ext_domain",
                "suspicious_inline_js"
            ]
        )
    else:
        actual = result[test["feature"]]

    print(
        f"{test['name']}: "
        f"{'PASS' if actual == test['expected'] else 'FAIL'}"
    )

branding_tests = [
    {
        "name": "Footer mismatch",
        "html": "<footer>Welcome to another company</footer>",
        "feature": "is_footer_mismatch",
        "expected": 1
    },
    {
        "name": "Header mismatch",
        "html": "<header>Welcome to another company</header>",
        "feature": "is_header_mismatch",
        "expected": 1
    },
    {
        "name": "Copyright mismatch",
        "html": "<footer>Copyright © Another Company</footer>",
        "feature": "is_copyright_mismatch",
        "expected": 1
    },
    {
        "name": "Social mismatch",
        "html": '<a href="https://facebook.com/othercompany">Facebook</a>',
        "feature": "is_socials_mismatch",
        "expected": 1
    },
    {
        "name": "Favicon mismatch",
        "html": '<link rel="icon" href="https://other.com/favicon.ico">',
        "feature": "is_favicon_mismatch",
        "expected": 1
    },
    {
        "name": "Missing title",
        "html": "<html><head></head><body></body></html>",
        "feature": "has_missing_title",
        "expected": 1
    },
    {
        "name": "Title mismatch",
        "html": "<title>Welcome to Other Company</title>",
        "feature": "title_mismatch_with_domain",
        "expected": 1
    },
    {
        "name": "Inline event handler",
        "html": '<button onclick="alert(1)">Click</button>',
        "feature": "has_inline_event_handlers",
        "expected": 1
    },
    {
        "name": "External form action",
        "html": '<form action="https://other.com/login"></form>',
        "feature": "form_action_external",
        "expected": 1
    },
    {
        "name": "Matching branding",
        "html": """
            <title>Example Login</title>
            <header>Example</header>
            <footer>© Example 2026</footer>
            <link rel="icon" href="/favicon.ico">
            <form action="/login"></form>
        """,
        "feature": "is_copyright_mismatch",
        "expected": 0
    }
]

for test in branding_tests:
    result = extract_html_features(
        test["html"],
        "https://example.com"
    )

    actual = result[test["feature"]]

    print(
        f"{test['name']}: "
        f"{'PASS' if actual == test['expected'] else 'FAIL'}"
    )

    if actual != test["expected"]:
        print(f"Expected: {test['expected']}, Actual: {actual}")