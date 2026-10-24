# Model Comparison: RandomForest vs. XGBoost vs. LightGBM vs. SVM

## Purpose

The production app used a single RandomForest classifier. This comparison
evaluates three additional algorithms — XGBoost, LightGBM, and SVM — on the
same data and features, to check whether a different model would perform
better before committing to one.

## Methodology

- **Script:** `compare_models.py` (does not modify `model.pkl` or `app.py`)
- **Data:** `data/raw_urls.csv` + `data/top_domains.csv` + curated examples,
  combined via the same `load_dataset()` used by `train_model.py` —
  463,893 URLs total (397,407 legitimate / 66,486 phishing)
- **Features:** the same 21 lexical features from `feature_extraction.py`
  used by the live app (URL length, entropy, subdomain count, IP-address
  usage, suspicious TLD, etc.) — no external lookups (WHOIS, traffic APIs)
- **Split:** single 80/20 stratified train/test split (`random_state=42`),
  identical across all four models, plus 3-fold stratified cross-validation
- **Class imbalance:** legitimate URLs outnumber phishing URLs ~6:1, so
  every model was configured for balanced class weighting
  (`class_weight="balanced"` for RandomForest/LightGBM/SVM;
  `scale_pos_weight` for XGBoost, computed to have the equivalent effect)
- **SVM caveat:** kernel SVM training time scales roughly quadratically
  with sample count, so training on the full 370k-row training split was
  impractical (would take hours). SVM was trained/cross-validated on a
  15,000-row stratified subsample instead, then evaluated on the full
  92,779-row held-out test set. This is a standard, documented practice
  for applying kernel SVM to large datasets, but it means SVM's numbers
  aren't fully apples-to-apples with the tree-based models.

## Results

| Model | Accuracy | ROC-AUC | Phishing Precision | Phishing Recall | Phishing F1 | CV Accuracy |
|---|---|---|---|---|---|---|
| **RandomForest** | **91.9%** | **0.9603** | 67% | 85% | 0.75 | 91.7% |
| LightGBM | 88.2% | 0.9498 | 56% | 87% | 0.68 | 88.1% |
| XGBoost | 87.8% | 0.9471 | 55% | 87% | 0.67 | 87.6% |
| SVM (15k subsample) | 83.7% | 0.8986 | 46% | 83% | 0.59 | 82.5% |

(Precision/recall/F1 above are for the **Phishing** class — the one that
actually matters for a detector, since missing a phishing site is worse
than a false alarm on a legitimate one. ROC-AUC is threshold-independent —
it measures how well each model ranks phishing URLs above legitimate ones
across *all* possible decision thresholds, not just the default 0.5 cutoff
used for the precision/recall/F1 columns. It's not skewed by the ~6:1
class imbalance the way plain accuracy can be.)

### ROC curves

![ROC curves comparing RandomForest, XGBoost, LightGBM, and SVM](roc_curves.png)

RandomForest's curve sits above the others across virtually the entire
range, confirming its higher AUC isn't an artifact of one threshold choice
— it separates phishing from legitimate URLs better everywhere. XGBoost
and LightGBM are close behind and nearly overlap each other. SVM is
clearly weakest, though recall it was trained on a 15k-row subsample
rather than the full data (see caveat above), so some of that gap may
narrow with more SVM training data.

### Note on an initial run

XGBoost was first tested with its default settings, before class
weighting was added. It looked highly accurate (93.1%) but that number
was misleading: it only caught 60% of phishing URLs, worse than every
other model. It just avoided flagging legitimate sites as phishing very
aggressively, which inflates plain accuracy on this ~6:1 imbalanced
dataset without actually being a better phishing detector. Once XGBoost
was given the same balanced class weighting as the other models, its
phishing recall rose to 87% — in line with LightGBM — but accuracy and
precision dropped, matching the table above. This is why accuracy alone
was not used as the deciding metric.

## Hyperparameter check: RandomForest tree depth

`train_model.py` doesn't cap `max_depth`, so each of RandomForest's 300
trees grows until leaves have fewer than `min_samples_leaf=2` samples. On
the trained production model this produces deep trees (depth 37-52,
averaging 43, ~9.9M nodes total across all 300 trees). To check whether
this was excessive (risking overfitting or wasted size/training time), a
sweep over `max_depth` was run with everything else held constant:

| max_depth | Accuracy | Phishing F1 | Total nodes (all trees) | Train time |
|---|---|---|---|---|
| **None (uncapped, current)** | **91.89%** | **0.7495** | 9.9M | 38.6s |
| 30 | 91.71% | 0.7472 | 8.8M | 35.9s |
| 20 | 89.11% | 0.6963 | 3.5M | 33.8s |
| 15 | 86.20% | 0.6436 | 1.2M | 30.5s |
| 10 | 81.90% | 0.5760 | 0.24M | 25.5s |

Capping depth at 30 recovers almost all of the tree size/training-time
saving available (~11% fewer nodes) while losing negligible accuracy, but
going any shallower costs real accuracy — depth 10 drops accuracy from
92% to 82%. Since train/CV accuracy already track closely at the uncapped
depth (no sign of overfitting), there's no meaningful benefit to capping
it, so **`max_depth` is being left uncapped**.

## Conclusion

**RandomForest remains the best-performing model** for this task once all
four algorithms are compared under the same balanced-weighting setup: it
has the highest accuracy, the highest ROC-AUC (0.9603, vs. 0.895-0.950 for
the others), the highest cross-validated accuracy, and the best
precision/recall balance on the phishing class (fewest false alarms on
legitimate sites, while still catching the large majority of phishing
attempts). Because ROC-AUC is threshold-independent, this result isn't
just an artifact of the 6:1 class imbalance or of the default 0.5
decision threshold — RandomForest genuinely separates the two classes
better than the alternatives. XGBoost, LightGBM, and SVM all catch
phishing at a similar or slightly better rate but with noticeably more
false positives.

No change was made to the production model (`model.pkl`) or `app.py` as a
result of this comparison — the existing RandomForest choice is retained.

## How to reproduce

```
python compare_models.py
```

Full per-model classification reports and confusion matrices are printed
to the console (see `compare_output.log` for the last run's full output).
