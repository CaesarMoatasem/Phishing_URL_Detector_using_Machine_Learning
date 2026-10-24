"""

1. data/raw_urls.csv - ~420k URLs labeled "good"/"bad" from
   https://github.com/faizann24/Using-machine-learning-to-detect-malicious-URLs
   
2. data/top_domains.csv - the top ~52k domains from the Majestic Million
   (https://majestic.com/reports/majestic-million)

"""

import os
import urllib.request

RAW_URLS_URL = (    
    "https://raw.githubusercontent.com/faizann24/"
    "Using-machine-learning-to-detect-malicious-URLs/master/data/data.csv"
)
MAJESTIC_URL = "https://downloads.majestic.com/majestic_million.csv"
# The full Majestic Million is ~80MB; we only need the top (most popular,
# most trustworthy) rows, so fetch a byte-range instead of the whole file.
MAJESTIC_RANGE_BYTES = 4_000_000

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_URLS_PATH = os.path.join(DATA_DIR, "raw_urls.csv")
TOP_DOMAINS_PATH = os.path.join(DATA_DIR, "top_domains.csv")


def download_raw_urls():
    print(f"Downloading {RAW_URLS_URL} ...")
    urllib.request.urlretrieve(RAW_URLS_URL, RAW_URLS_PATH)
    size_mb = os.path.getsize(RAW_URLS_PATH) / (1024 * 1024)
    print(f"Saved to {RAW_URLS_PATH} ({size_mb:.1f} MB)")


def download_top_domains():
    print(f"Downloading top domains from {MAJESTIC_URL} "
          f"(first {MAJESTIC_RANGE_BYTES:,} bytes) ...")
    req = urllib.request.Request(
        MAJESTIC_URL, headers={"Range": f"bytes=0-{MAJESTIC_RANGE_BYTES}"}
    )
    with urllib.request.urlopen(req) as resp:
        raw = resp.read().decode("utf-8", errors="ignore")
    # The byte range can cut off mid-row; drop the last (possibly partial) line.
    lines = raw.splitlines()[:-1]
    with open(TOP_DOMAINS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved {len(lines) - 1} domains to {TOP_DOMAINS_PATH}")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    download_raw_urls()
    download_top_domains()


if __name__ == "__main__":
    main()
