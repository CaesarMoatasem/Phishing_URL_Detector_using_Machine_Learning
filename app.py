"""
The Live Website: http://127.0.0.1:5000
"""
import pickle
import os
import pandas as pd
from flask import Flask, render_template, request

from feature_extraction import extract_features, extract_features_dict, FEATURE_NAMES

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Human-readable explanations shown when a feature looks suspicious
RED_FLAG_LABELS = {
    "has_ip_address": "Uses a raw IP address instead of a domain name",
    "has_at": "Contains an '@' symbol (can hide the real destination)",
    "has_prefix_suffix": "Domain contains a hyphen (common in fake lookalike domains)",
    "has_https_token_in_domain": "The word 'https' appears in the domain itself (fake trust signal)",
    "has_shortener": "Uses a known URL-shortening service",
    "has_double_slash_redirect": "Contains a suspicious '//' redirect pattern in the path",
    "num_subdomains": "Unusually high number of subdomains",
    "url_length": "Unusually long URL",
}


def get_red_flags(feat_dict: dict) -> list:
    flags = []
    if feat_dict.get("has_ip_address"):
        flags.append(RED_FLAG_LABELS["has_ip_address"])
    if feat_dict.get("num_at", 0) > 0:
        flags.append(RED_FLAG_LABELS["has_at"])
    if feat_dict.get("has_prefix_suffix"):
        flags.append(RED_FLAG_LABELS["has_prefix_suffix"])
    if feat_dict.get("has_https_token_in_domain"):
        flags.append(RED_FLAG_LABELS["has_https_token_in_domain"])
    if feat_dict.get("has_shortener"):
        flags.append(RED_FLAG_LABELS["has_shortener"])
    if feat_dict.get("has_double_slash_redirect"):
        flags.append(RED_FLAG_LABELS["has_double_slash_redirect"])
    if feat_dict.get("num_subdomains", 0) >= 3:
        flags.append(RED_FLAG_LABELS["num_subdomains"])
    if feat_dict.get("url_length", 0) > 75:
        flags.append(RED_FLAG_LABELS["url_length"])
    return flags


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        url = request.form.get("url", "").strip()
        if url:
            features = extract_features(url)
            feat_dict = extract_features_dict(url)
            features_df = pd.DataFrame([features], columns=FEATURE_NAMES)

            prediction = model.predict(features_df)[0]
            proba = model.predict_proba(features_df)[0]
            confidence = round(max(proba) * 100, 1)

            is_legit = prediction == 1
            result = {
                "url": url,
                "verdict": "Legitimate" if is_legit else "Phishing",
                "is_legit": is_legit,
                "confidence": confidence,
                "red_flags": get_red_flags(feat_dict),
            }

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)