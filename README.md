# retail-churn-pytorch

PyTorch MLP and TabNet churn classifiers trained on a synthetic retail dataset
of 200,000 customers across six behavioral segments using RFM features.

This repo covers model training, evaluation, and explainability.
The BQML baseline and SQL-native scoring pipeline live in the companion repo:
https://github.com/gbhorne/retail-churn-bqml

This repo serves as the flexible deployment track for comparison against the
SQL-native BQML baseline in the companion repo.

---

## Churn label definition

| Property | Value |
|----------|-------|
| Observation window | Full synthetic purchase history per customer |
| Prediction horizon | N/A (synthetic labels assigned at generation time) |
| Churn rule | Probabilistic per segment (see table below) |
| Scoring timestamp | End of observation window |

In this synthetic dataset, churn represents a probabilistic likelihood of
inactivity based on behavioral segment rather than a fixed time-based definition.
Because churn labels are assigned probabilistically and not derived from recency,
no feature directly determines the target, avoiding feature leakage.

| Segment | Churn probability |
|---------|------------------|
| Champions | 5% |
| Loyal | 10% |
| Potential | 40% |
| Recent | 25% |
| At-Risk | 65% |
| Hibernating | 85% |

In a production deployment, churn would be defined as no purchase within
a fixed horizon (e.g. 90 days) after a feature cutoff date. Features must
be computed strictly before that cutoff to prevent leakage.

---

## Experiment design

All three models (BQML, MLP, TabNet) are trained on the same underlying
dataset with the same feature definitions to ensure a fair comparison.
Because customer histories are synthetically generated without a temporal
prediction task, a random stratified split is appropriate for this experiment.

| Property | Value |
|----------|-------|
| Dataset | 200,000 synthetic retail customers |
| Features | 9 numeric + log-scaled variants + interaction terms + categoricals |
| MLP categorical handling | One-hot encoding (no ordinal assumption) |
| TabNet categorical handling | Integer IDs via cat_idxs and cat_dims |
| Train split | 72.25% (stratified) |
| Validation split | 12.75% (stratified) |
| Test split | 15% (stratified) |
| Split method | Random stratified (no temporal ordering in synthetic data) |
| Random seed | 42 (fixed for numpy, torch, and Python random) |
| Class balance | 53% churned, 47% not churned |
| Class imbalance handling | WeightedRandomSampler for MLP, native for TabNet |

Note: in a production system with real transaction data, a time-based split
must be used to prevent future data leaking into training features.

---

## Architecture

![Architecture](docs/charts/architecture.svg)

---

## Results (reference run, CPU, Windows)

Results are saved to mlp_results.json and tabnet_results.json after each
training run. The numbers below are from the reference runs documented here.
Results will vary across hardware, random seeds, and library versions.
PR-AUC for BQML is not included in this reference run but can be derived
from ML.ROC_CURVE in the companion repo.

| Model | ROC-AUC | PR-AUC | F1 (churned) | Training time |
|-------|---------|--------|--------------|---------------|
| BQML logistic regression | 0.826 | See companion repo | 0.725 | 86s |
| PyTorch MLP | 0.8335 | 0.8101 | 0.790 | 115.7s |
| TabNet | 0.8344 | 0.8153 | 0.790 | 825.7s |

In this experiment, neither the MLP nor TabNet materially outperformed the
simpler BQML baseline on this feature set. The differences are in
explainability, portability, and deployment flexibility.

---

## Cost comparison

| Model | Infrastructure | Estimated training cost | Scoring mechanism |
|-------|---------------|------------------------|-------------------|
| BQML logistic regression | None (serverless) | Cents per TB processed | ML.PREDICT in SQL |
| PyTorch MLP | CPU VM or local machine | VM cost per training hour | REST API endpoint |
| TabNet | CPU VM (GPU recommended) | Higher VM cost, longer run | REST API endpoint |

BQML has the lowest total cost of ownership for batch scoring workloads.
PyTorch models are more appropriate when a REST API serving layer is required
or when the model needs to be embedded in an application.

---

## Decision threshold guide

The default classification threshold of 0.5 is not always the right business
decision. Use the precision-recall chart to select a threshold based on
campaign economics.

| Churn probability | Risk tier | Recommended action |
|-------------------|-----------|--------------------|
| >= 0.80 | High | Immediate retention intervention, personalized outreach |
| 0.60 to 0.79 | Medium-high | Proactive nurture campaign, targeted offer |
| 0.40 to 0.59 | Medium | Engagement campaign, product recommendation |
| < 0.40 | Low | No action or low-cost passive re-engagement |

Breakeven threshold formula: campaign cost / expected recovered lifetime value.
Example: $5 campaign cost, $80 recovered LTV means act on any customer above 0.063.

---

## Key design decisions

- MLP uses one-hot encoding for categoricals. Label encoding with scaling
  imposes a false ordinal relationship on nominal features.
- TabNet receives categorical columns as integer IDs via cat_idxs and cat_dims,
  not as scaled floats. This is how TabNet is designed to be used.
- WeightedRandomSampler handles class imbalance during MLP training.
  pos_weight is not used alongside oversampling to avoid double-weighting
  and probability calibration distortion.
- The scaler, feature column order, and label mappings are saved alongside
  model weights so inference artifacts are fully self-contained.
- SHAP uses training data as the background distribution, not test data.
- Global seeds are set for numpy, torch, and random for reproducibility.
- The segment feature is derived from RFM scores and may carry partial
  signal overlap with the churn label. It is retained as a real-world
  proxy for customer lifecycle stage but should be monitored for redundancy
  in production feature selection.

---

## Serving the MLP model

The inference pipeline loads model weights, scaler, and feature metadata
together. All three artifacts must be present for reliable inference.

```
# Score a single customer
python predict.py --user_id 1

# Score all customers in a batch
python predict.py --batch data/features.csv
```

The predict.py script applies the same preprocessing pipeline used during
training and outputs churn probability, risk tier, and recommended action
per customer.

---

## Charts

![Model comparison](docs/charts/model_comparison.png)

![AUC vs. training time](docs/charts/auc_vs_training_time.png)

![Segment distribution](docs/charts/segment_distribution.png)

![Churn by segment](docs/charts/churn_by_segment.png)

![Precision vs. recall threshold](docs/charts/precision_recall_threshold.png)

![SHAP summary](docs/charts/shap_summary.png)

![SHAP feature importance](docs/charts/shap_bar.png)

---

## When to use each model

| Consideration | BQML | MLP | TabNet |
|--------------|------|-----|--------|
| No Python required | Yes | No | No |
| Trains in under 2 minutes | Yes | Yes | No |
| Portable model artifact | No (stays in BigQuery) | Yes (.pth) | Yes (.zip) |
| Feature importance | No | Via SHAP | Built-in masks |
| Handles categoricals natively | No | Via one-hot | Yes (cat_idxs) |
| Deploy to Vertex AI endpoint | No | Yes | Yes |
| Lowest infrastructure cost | Yes | No | No |
| Best for | SQL pipelines | REST APIs | Explainability audits |

---

## Project structure

| Path | Purpose |
|------|---------|
| data/export_from_bq.py | Pulls rfm_scores from BigQuery to CSV |
| src/features.py | Feature engineering with separate MLP and TabNet paths |
| src/dataset.py | PyTorch Dataset and WeightedRandomSampler |
| src/mlp.py | 3-layer MLP architecture |
| src/tabnet.py | TabNet wrapper with cat_idxs and cat_dims |
| src/train.py | MLP training loop with early stopping |
| src/evaluate.py | Metrics and SHAP summary |
| train_mlp.py | MLP training entrypoint, saves weights + scaler + metrics |
| train_tabnet.py | TabNet training entrypoint, saves weights + metadata + metrics |
| predict.py | Inference pipeline: score single customer or batch CSV |
| run_shap.py | SHAP explainability using training data as background |
| build_docs.py | Generates charts and README (no auto-push) |
| mlp_results.json | Saved MLP experiment results (generated at training time) |
| tabnet_results.json | Saved TabNet experiment results (generated at training time) |

---

## Synthetic data disclaimer

All customer data is synthetically generated. No real customer records,
PII, or proprietary retail data was used. The synthetic dataset is stored
in BigQuery and generated by the companion retail-churn-bqml repo.
The generator script is generate_synthetic_data.py in that repo.

---

## License

MIT