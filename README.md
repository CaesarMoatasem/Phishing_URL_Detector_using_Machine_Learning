# URL Threat Scanner

A small Flask web app that checks whether a URL looks like phishing,
using a machine learning model trained on lexical (string-based)
URL features.

## How it works

1. `feature_extraction.py` — pulls ~20 features straight out of the
   URL string itself (length, entropy, digit ratio, path depth, IP
   address usage, "@" symbol, hyphens, subdomain count, HTTPS usage,
   known shorteners, sensitive keywords, suspicious TLDs, etc.). None
   of these need a network call, so predictions are instant. Scheme
   (`http://`/`https://`) and a leading `www.` are normalized out of
   the length/entropy-based features, so the verdict doesn't depend
   on how someone happens to type the URL.
2. `download_dataset.py` — fetches the two public datasets used for
   training (see below) into `data/`.
3. `train_model.py` — builds the training set from those datasets
   plus a small hand-curated list, trains a RandomForest, and saves
   it as `model.pkl`. Prints hold-out accuracy, a classification
   report, a confusion matrix, 3-fold cross-validation, and feature
   importances.
4. `app.py` — Flask app: paste a URL, it extracts features, loads
   the model, and shows a verdict with a confidence score and the
   specific red flags that were detected.

## Dataset

Training data comes from three sources, combined in `train_model.py`:

- **`data/raw_urls.csv`** (~410k rows) — a public benign/malicious
  URL dataset ([faizann24/Using-machine-learning-to-detect-malicious-URLs](https://github.com/faizann24/Using-machine-learning-to-detect-malicious-URLs)),
  labeled `good`/`bad`. This is a general web crawl, so `good` URLs
  skew toward deep content pages rather than bare homepages.
- **`data/top_domains.csv`** (~52k rows) — the top-ranked domains
  from the [Majestic Million](https://majestic.com/reports/majestic-million),
  added as legitimate examples. This fixes a real gap: bare, everyday
  domains like `google.com` are underrepresented in a crawl dataset,
  so without this the model had too few examples of what a real
  top-tier site's homepage actually looks like.
- **`CURATED_URLS`** in `train_model.py` — a small hand-picked,
  scheme-ful list (~100 URLs) kept from the original starter set, so
  the model still sees a few genuine http-vs-https examples.

## Setup

```bash
pip install -r requirements.txt
python download_dataset.py  # fetches data/raw_urls.csv + data/top_domains.csv (~25MB)
python train_model.py       # trains model.pkl on the combined dataset
python app.py                # starts the server
```

Then open **http://127.0.0.1:5000** in your browser.

If `data/raw_urls.csv` / `data/top_domains.csv` are missing,
`train_model.py` falls back to `CURATED_URLS` only (small, but keeps
the pipeline runnable without a download).

Current results (your numbers may vary slightly run-to-run):
hold-out accuracy ~92%, 3-fold cross-validation ~92% (± 0.1%).

## Project structure

```
phishing_url_detector/
├── app.py                 # Flask app
├── feature_extraction.py  # live lexical feature extractor
├── download_dataset.py    # fetches the training datasets into data/
├── train_model.py         # builds the dataset, trains, saves model.pkl
├── data/
│   ├── raw_urls.csv       # downloaded (good/bad URLs)
│   └── top_domains.csv    # downloaded (Majestic Million top domains)
├── model.pkl              # trained model (generated)
├── requirements.txt
├── templates/
│   └── index.html
└── static/
    └── style.css
```

