"""
compare_models.py

Compares several classifiers (RandomForest, XGBoost, LightGBM, SVM) on the
same lexical URL features and the same train/test split, so their metrics
are directly comparable.

This script does NOT touch model.pkl or app.py - it's purely for evaluation/
reporting. Run it with:  python compare_models.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from feature_extraction import FEATURE_NAMES, extract_features
from train_model import load_dataset

RANDOM_STATE = 42


def build_models(y) -> dict:
    """Build the model dict. XGBoost has no class_weight param like the
    others - it uses scale_pos_weight instead, computed here from the
    actual label counts (count(phishing) / count(legitimate)) so it
    down-weights the majority "legitimate" class the same way the other
    models' class_weight="balanced" does. Without this, XGBoost defaults
    to treating both classes equally and badly under-catches phishing
    (60% recall vs. ~85% for the balanced models)."""
    neg_pos_ratio = y.count(0) / y.count(1)

    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            eval_metric="logloss",
            scale_pos_weight=neg_pos_ratio,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=-1,
        ),
        # SVM needs scaled features to train in reasonable time/accuracy, unlike
        # the tree models above - handled via SCALED_MODELS below.
        "SVM": SVC(
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=RANDOM_STATE,
        ),
    }


# Models that need standardized features (mean 0, std 1) to perform well.
SCALED_MODELS = {"SVM"}

# Kernel SVM training time scales roughly quadratically with sample count,
# so fitting it on the full ~470k-row dataset is impractical (would take
# hours). Train it on a stratified subsample instead - a standard, documented
# practice for kernel SVM on large datasets. It's still evaluated on the
# full held-out test set, so the reported metrics stay comparable.
SVM_MAX_TRAIN_SAMPLES = 15000


def main():
    df = load_dataset()
    print(f"Total examples: {len(df)}  "
          f"(legitimate={int((df['label'] == 1).sum())}, "
          f"phishing={int((df['label'] == 0).sum())})")

    X = [extract_features(str(u)) for u in df["url"]]
    y = df["label"].tolist()
    features_df = pd.DataFrame(X, columns=FEATURE_NAMES)

    X_train, X_test, y_train, y_test = train_test_split(
        features_df, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    models = build_models(y)

    results = []
    roc_curves = {}
    for name, model in models.items():
        print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")

        if name in SCALED_MODELS:
            train_X, test_X, cv_X = X_train_scaled, X_test_scaled, scaler.transform(features_df)
        else:
            train_X, test_X, cv_X = X_train, X_test, features_df

        train_y, cv_y = y_train, y

        if name == "SVM" and len(train_X) > SVM_MAX_TRAIN_SAMPLES:
            train_X, _, train_y, _ = train_test_split(
                train_X, y_train,
                train_size=SVM_MAX_TRAIN_SAMPLES,
                random_state=RANDOM_STATE,
                stratify=y_train,
            )
            cv_X, _, cv_y, _ = train_test_split(
                cv_X, y,
                train_size=SVM_MAX_TRAIN_SAMPLES,
                random_state=RANDOM_STATE,
                stratify=y,
            )
            print(f"(SVM trained/cross-validated on a stratified subsample of "
                  f"{SVM_MAX_TRAIN_SAMPLES} rows - full data is impractical for "
                  f"kernel SVM. Still evaluated on the full {len(X_test)}-row test set.)")

        model.fit(train_X, train_y)
        y_pred = model.predict(test_X)
        # Probability that the URL is legitimate (class 1) - used for both
        # ROC-AUC and the ROC curve plot. classes_ order is looked up
        # rather than assumed, since it's not guaranteed to be [0, 1].
        pos_idx = list(model.classes_).index(1)
        y_score = model.predict_proba(test_X)[:, pos_idx]

        acc = accuracy_score(y_test, y_pred)
        # Phishing (class 0) precision/recall/F1 - the class that actually
        # matters for a detector, since missing phishing (a false negative
        # here) is worse than a false alarm on a legitimate site.
        prec = precision_score(y_test, y_pred, pos_label=0)
        rec = recall_score(y_test, y_pred, pos_label=0)
        f1 = f1_score(y_test, y_pred, pos_label=0)
        # ROC-AUC is threshold-independent and symmetric between classes
        # for binary classification, so it doesn't matter which class is
        # treated as "positive" here - it summarizes overall separability.
        auc = roc_auc_score(y_test, y_score)

        print(classification_report(y_test, y_pred, target_names=["Phishing", "Legitimate"]))
        print("Confusion matrix ([[TN, FP], [FN, TP]] for Phishing=0, Legitimate=1):")
        print(confusion_matrix(y_test, y_pred))
        print(f"ROC-AUC: {auc:.4f}")

        cv_scores = cross_val_score(model, cv_X, cv_y, cv=cv, n_jobs=-1)
        print(f"3-fold CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=1)
        roc_curves[name] = (fpr, tpr, auc)

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Phish Precision": prec,
            "Phish Recall": rec,
            "Phish F1": f1,
            "ROC-AUC": auc,
            "CV Accuracy": cv_scores.mean(),
        })

    summary = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)
    print(f"\n{'=' * 60}\nSummary (sorted by ROC-AUC)\n{'=' * 60}")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    plt.figure(figsize=(7, 6))
    for name, (fpr, tpr, auc) in roc_curves.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random guess (AUC = 0.5000)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Phishing URL Detection Models")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = "roc_curves.png"
    plt.savefig(roc_path, dpi=150)
    print(f"\nSaved ROC curve plot to {roc_path}")


if __name__ == "__main__":
    main()
