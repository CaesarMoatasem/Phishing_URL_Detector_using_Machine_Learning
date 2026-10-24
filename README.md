# Phishing_URL_Detector_using_Machine_Learning
URL Threat Scanner is a Flask web app that detects phishing URLs with a Random Forest model trained on ~460k URLs. It extracts ~20 features from the URL text (length, entropy, IP use, suspicious TLDs, shorteners, keywords) with no network calls and returns an instant verdict, confidence score and red flags.
