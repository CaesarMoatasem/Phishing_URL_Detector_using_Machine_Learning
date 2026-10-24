"""
feature_extraction.py

Extracts lexical (string-based) features from a URL.
These features can all be computed instantly, with no network
calls, WHOIS lookups, or third-party APIs — which makes them
ideal for a live, real-time URL checker.

If you trained your main model on the full 30-column Kaggle/UCI
dataset (which includes features like web_traffic, Page_Rank,
age_of_domain that DO require external lookups), you have two
options for the live app:
  1) Retrain a second, lighter model using only the lexical
     subset of those columns (recommended, and what this file
     supports out of the box), or
  2) Wire in real WHOIS / traffic API calls here to compute the
     full feature set (slower, needs API keys).
"""

import math
import re
from collections import Counter
from urllib.parse import urlparse

KNOWN_SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd",
    "buff.ly", "adf.ly", "shorte.st", "rebrand.ly", "cutt.ly"
}

IP_PATTERN = re.compile(
    r"^(?:(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|[01]?\d?\d)$"
)

# TLDs frequently abused for phishing/malware because they're cheap or
# unmoderated (source: recurring entries in Spamhaus/Netcraft abuse reports).
SUSPICIOUS_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "work", "click", "link",
    "country", "kim", "review", "loan", "download", "racing", "science",
    "party", "gdn", "men", "win", "stream", "bid",
}

# Words attackers stuff into a URL to fake a trust/urgency signal
# (login pages, account alerts, verification flows).
SENSITIVE_KEYWORDS = {
    "secure", "account", "update", "login", "verify", "signin", "sign-in",
    "banking", "confirm", "password", "billing", "webscr", "suspend",
    "unlock", "recover", "wallet", "invoice",
}

SPECIAL_CHARS = set("%=&_~$!*'();:@?#[]")

# Order matters: this list defines the feature vector column order
# used by both train_model.py and app.py. Keep them in sync.
FEATURE_NAMES = [
    "url_length",
    "num_dots",
    "num_hyphens",
    "num_at",
    "num_digits",
    "num_subdomains",
    "has_ip_address",
    "has_https",
    "has_https_token_in_domain",
    "has_shortener",
    "has_double_slash_redirect",
    "has_prefix_suffix",
    "url_entropy",
    "digit_ratio",
    "special_char_count",
    "path_depth",
    "num_query_params",
    "domain_length",
    "hyphen_ratio_domain",
    "has_suspicious_tld",
    "has_sensitive_keyword",
]


def _shannon_entropy(s: str) -> float:
    """Higher entropy ~= more random-looking string (common in
    algorithmically generated phishing domains)."""
    if not s:
        return 0.0
    counts = Counter(s)
    length = len(s)
    return -sum((c / length) * math.log2(c / length) for c in counts.values())


def extract_features(url: str) -> list:
    """Return a list of numeric features for the given URL,
    in the same order as FEATURE_NAMES."""

    url = url.strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        # No scheme provided (e.g. "paypal-secure.com") — assume http
        # so urlparse can split host/path correctly.
        url = "http://" + url

    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        path = parsed.path or ""
        query = parsed.query
        params = parsed.params
        scheme = parsed.scheme
    except ValueError:
        # urlparse rejects a few malformed real-world URLs (e.g. a stray
        # "[" in the host, which it tries to validate as an IPv6 literal).
        # Fall back to a plain manual split so we never crash on bad input.
        scheme, _, rest = url.partition("://")
        domain, _, path = rest.partition("/")
        domain = domain.lower()
        path = "/" + path
        query = ""
        params = ""

    # strip port if present, e.g. example.com:8080
    domain_no_port = domain.split(":")[0]
    # Strip a leading "www." for domain-shape features so a URL isn't
    # judged differently just because someone typed (or omitted) "www." —
    # e.g. "www.google.com" and "google.com" should look equally legitimate.
    domain_stripped = re.sub(r"^www\.", "", domain_no_port)
    domain_labels = [d for d in domain_stripped.split(".") if d]
    tld = domain_labels[-1] if domain_labels else ""

    path_segments = [seg for seg in path.split("/") if seg]
    query_params = [p for p in query.split("&") if p]

    # Content with scheme + leading "www." removed. Whether someone types
    # "google.com", "www.google.com", or "https://www.google.com" is a
    # typing/style choice, not a legitimacy signal — so length/entropy/
    # digit-based features are computed on this normalized form instead of
    # the raw url, otherwise those boilerplate characters swamp the real
    # signal (this mattered in practice: a training set mixing bare
    # domains with fully-qualified https://www. URLs taught a bogus
    # "longer url == phishing" rule from the scheme/www prefix alone).
    content = domain_stripped + path + ("?" + query if query else "")
    content_lower = content.lower()

    features = {
        "url_length": len(content),
        "num_dots": content.count("."),
        "num_hyphens": content.count("-"),
        "num_at": content.count("@"),
        "num_digits": sum(c.isdigit() for c in content),
        "num_subdomains": max(domain_stripped.count(".") - 1, 0),
        "has_ip_address": 1 if IP_PATTERN.match(domain_no_port) else 0,
        "has_https": 1 if scheme == "https" else 0,
        # "https" appearing somewhere it shouldn't (a fake trust signal)
        "has_https_token_in_domain": 1 if "https" in domain_stripped else 0,
        "has_shortener": 1 if domain_stripped in KNOWN_SHORTENERS else 0,
        "has_double_slash_redirect": 1 if "//" in (path + params) else 0,
        "has_prefix_suffix": 1 if "-" in domain_stripped else 0,
        "url_entropy": _shannon_entropy(content_lower),
        "digit_ratio": sum(c.isdigit() for c in content) / len(content) if content else 0.0,
        "special_char_count": sum(1 for c in content if c in SPECIAL_CHARS),
        "path_depth": len(path_segments),
        "num_query_params": len(query_params),
        "domain_length": len(domain_stripped),
        "hyphen_ratio_domain": (
            domain_stripped.count("-") / len(domain_stripped) if domain_stripped else 0.0
        ),
        "has_suspicious_tld": 1 if tld in SUSPICIOUS_TLDS else 0,
        "has_sensitive_keyword": 1 if any(k in content_lower for k in SENSITIVE_KEYWORDS) else 0,
    }

    return [features[name] for name in FEATURE_NAMES]


def extract_features_dict(url: str) -> dict:
    """Same as extract_features but returns a labeled dict —
    handy for showing 'red flags' in the UI."""
    values = extract_features(url)
    return dict(zip(FEATURE_NAMES, values))


if __name__ == "__main__":
    # quick manual test
    test_urls = [
        "https://www.google.com",
        "http://paypal-secure-login.verify-account.com/update",
        "http://192.168.1.1/login",
        "https://bit.ly/3xY9z",
    ]
    for u in test_urls:
        print(u, "->", extract_features_dict(u))
