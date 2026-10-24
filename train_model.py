"""
train_model.py

Trains a RandomForest classifier on lexical URL features and
saves it to model.pkl for app.py to load.

Dataset
-------
Training data is built from two sources:

1. data/raw_urls.csv - ~420k labeled URLs ("good"/"bad") pulled from
   https://github.com/faizann24/Using-machine-learning-to-detect-malicious-URLs
   (a public malicious-URL dataset commonly used for exactly this kind
   of lexical classifier). Run download_dataset.py to (re-)fetch it.
   NOTE: these URLs have no scheme (e.g. "example.com", not
   "https://example.com"), so has_https / has_https_token_in_domain
   carry no signal for this portion of the data - that's a known
   limitation of the source, not a bug here.
2. CURATED_URLS below - a small hand-picked set of real, scheme-ful
   examples (kept from the original starter dataset) that still gives
   the model a few genuine https-vs-http examples to learn from.

If data/raw_urls.csv is missing, training falls back to CURATED_URLS
only (small, but keeps the pipeline runnable out of the box).
"""

import os
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

from feature_extraction import extract_features, FEATURE_NAMES

BASE_DIR = os.path.dirname(__file__)
RAW_DATASET_PATH = os.path.join(BASE_DIR, "data", "raw_urls.csv")
TOP_DOMAINS_PATH = os.path.join(BASE_DIR, "data", "top_domains.csv")

# ---- Hand-verified real examples (label: 1 = legitimate, 0 = phishing) ----
# Small on its own, but scheme-ful, so it's the only source of genuine
# http-vs-https signal alongside the large schemeless dataset above.
CURATED_URLS = [
    ("https://www.google.com", 1),
    ("https://www.github.com", 1),
    ("https://en.wikipedia.org/wiki/Phishing", 1),
    ("https://www.python.org/downloads", 1),
    ("https://www.amazon.com/gp/css/order-history", 1),
    ("https://www.microsoft.com/en-us", 1),
    ("https://mail.google.com/mail/u/0/", 1),
    ("https://www.linkedin.com/in/someone", 1),
    ("https://www.nytimes.com/section/technology", 1),
    ("https://www.bbc.com/news", 1),
    ("https://www.dropbox.com/login", 1),
    ("https://www.reddit.com/r/MachineLearning", 1),
    ("https://stackoverflow.com/questions/tagged/python", 1),
    ("https://www.apple.com/shop/buy-iphone", 1),
    ("https://www.paypal.com/signin", 1),
    ("https://www.chase.com/personal/credit-cards", 1),
    ("https://www.netflix.com/browse", 1),
    ("https://www.coursera.org/learn/machine-learning", 1),
    ("https://www.spotify.com/premium", 1),
    ("https://www.imdb.com/chart/top", 1),
    ("https://accounts.google.com/signin", 1),
    ("https://www.ebay.com/sch/i.html?_nkw=laptop", 1),
    ("https://www.twitch.tv/directory", 1),
    ("https://www.wordpress.com/start", 1),
    ("https://www.adobe.com/products/photoshop.html", 1),
    ("https://www.tranco-list.eu", 1),
    ("https://www.cloudflare.com/plans", 1),
    ("https://www.researchgate.net/search", 1),
    ("https://scholar.google.com/", 1),
    ("https://www.udemy.com/courses", 1),
    ("https://www.noon.com/", 1),
    ("http://paypal-secure-login.verify-account.com/update", 0),
    ("http://192.168.1.104/login/index.php", 0),
    ("http://apple-id-locked.support-verify.com", 0),
    ("http://amaz0n-account-update.com/signin", 0),
    ("http://secure-chase-bank.com.verify-login.net/", 0),
    ("http://bit.ly/3xY9zLk", 0),
    ("http://microsoft-support-alert.win/login", 0),
    ("http://facebook-security-check.net/confirm", 0),
    ("http://www.paypa1.com/webapps/signin", 0),
    ("http://update-your-info-now.com/bank/login.php", 0),
    ("http://netflix-billing-update.info/account", 0),
    ("http://ebay-resolution-center.com-secure.info/", 0),
    ("http://192.168.0.15/wp-admin/verify.php", 0),
    ("http://appleid.apple.com.verify-account-now.com/", 0),
    ("http://your-bank-secure-login.com/update-details", 0),
    ("http://tinyurl.com/2p8x9k3z", 0),
    ("http://dropbox-file-shared-view.com/document", 0),
    ("http://signin-ebay.com.item-verify.net/", 0),
    ("http://192.168.1.1/webscr?cmd=login", 0),
    ("http://linkedin-invite-accept.com/connect", 0),
    ("http://wellsfargo-secure-alert.com/verify", 0),
    ("http://irs-tax-refund-claim.net/form", 0),
    ("http://instagram-copyright-appeal.com/form", 0),
    ("http://outlook-mail-storage-full.com/upgrade", 0),
    ("http://coinbase-wallet-verify.com/login", 0),
    ("http://usps-package-redelivery.info/track", 0),
    ("http://bankofamerica.com.session-verify.net/", 0),
    ("http://192.168.1.50/onedrive/share.php", 0),
    ("http://steamcommunity-trade-offer.com/gift", 0),
    ("http://att-billing-suspended.com/pay-now", 0),
    ("http://verify-yahoo-account-now.com/login", 0),
    ("http://xn--pypal-4ve.com/signin", 0),
    ("http://office365-license-expired.com/renew", 0),
    ("http://google-docs-shared-file.com-view.net/", 0),
    ("http://hsbc-online-banking.com.alert-verify.net/", 0),
    ("http://targeted-refund-form.com/claim-now", 0),
    ("http://binance-security-team.com/unlock", 0),
    ("http://update-icloud-payment.com/billing", 0),
    ("http://192.168.2.20/paypal/webapps/login", 0),
    ("http://free-gift-card-claim-now.com/amazon", 0),
    ("http://zoom-meeting-invite-secure.com/join", 0),
]


def load_dataset() -> pd.DataFrame:
    """Return a DataFrame with columns [url, label] (label: 1=legit, 0=phishing)."""
    curated_df = pd.DataFrame(CURATED_URLS, columns=["url", "label"])

    if not os.path.exists(RAW_DATASET_PATH):
        print(f"NOTE: {RAW_DATASET_PATH} not found - training on {len(curated_df)} "
              "curated URLs only. Run download_dataset.py to fetch the full dataset.")
        return curated_df

    raw_df = pd.read_csv(RAW_DATASET_PATH)
    raw_df = raw_df[raw_df["label"].isin(["good", "bad"])].copy()
    raw_df["label"] = (raw_df["label"] == "good").astype(int)
    raw_df = raw_df[["url", "label"]]

    frames = [raw_df]

    if os.path.exists(TOP_DOMAINS_PATH):
        # Real top-ranked domains (bare, no path) — supplements raw_df,
        # which (being a general web crawl) under-represents what a
        # well-known site's plain homepage URL looks like.
        top_df = pd.read_csv(TOP_DOMAINS_PATH)
        top_df = top_df.rename(columns={"Domain": "url"})[["url"]]
        top_df["label"] = 1
        frames.append(top_df)
    else:
        print(f"NOTE: {TOP_DOMAINS_PATH} not found - skipping top-domains "
              "supplement. Run download_dataset.py to fetch it.")

    frames.append(curated_df)

    # Priority when the same URL appears more than once: curated > top
    # domains > raw crawl (last duplicate wins), reflecting how much we
    # trust each source's label.
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset="url", keep="last")
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    return combined


def main():
    df = load_dataset()
    print(f"Total examples: {len(df)}  "
          f"(legitimate={int((df['label'] == 1).sum())}, "
          f"phishing={int((df['label'] == 0).sum())})")

    X = [extract_features(str(u)) for u in df["url"]]
    y = df["label"].tolist()

    features_df = pd.DataFrame(X, columns=FEATURE_NAMES)

    X_train, X_test, y_train, y_test = train_test_split(
        features_df, y,
        test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\nHold-out accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=["Phishing", "Legitimate"]))
    print("Confusion matrix ([[TN, FP], [FN, TP]] for Phishing=0, Legitimate=1):")
    print(confusion_matrix(y_test, y_pred))

    print("\nRunning 3-fold cross-validation (this takes a bit longer)...")
    # The raw dataset isn't shuffled (similar URLs cluster in blocks), so an
    # un-shuffled StratifiedKFold can give folds with very different
    # difficulty. Shuffle here to get a representative estimate.
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, features_df, y, cv=cv, n_jobs=-1)
    print(f"Cross-val accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    importances = sorted(zip(FEATURE_NAMES, model.feature_importances_), key=lambda x: -x[1])
    print("\nFeature importances:")
    for name, importance in importances:
        print(f"  {name:28s} {importance:.4f}")

    with open(os.path.join(BASE_DIR, "model.pkl"), "wb") as f:
        pickle.dump(model, f)
    print("\nSaved trained model to model.pkl")


if __name__ == "__main__":
    main()
