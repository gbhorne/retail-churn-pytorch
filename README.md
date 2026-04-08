# retail-churn-pytorch

Retail customer churn prediction using PyTorch MLP and TabNet trained on 200,000
synthetic retail customers across six behavioral segments with RFM features.
Part of a two-repo comparison against a BigQuery ML logistic regression baseline.

Companion repo: https://github.com/gbhorne/retail-churn-bqml

---

## Architecture

![Architecture](docs/charts/architecture.svg)

---

## Results

| Model | ROC-AUC | PR-AUC | F1 (churned) | Training time |
|-------|---------|--------|--------------|---------------|
| BQML Logistic Regression | 0.826 | N/A | 0.725 | 86s |
| PyTorch MLP | 0.834 | 0.809 | 0.790 | 42s |
| TabNet | 0.834 | 0.816 | 0.790 | 1281s |

All three models land within 0.008 AUC of each other. Neural network complexity
does not meaningfully improve accuracy on clean tabular RFM data. The real
differences are explainability, portability, and infrastructure cost.

---

## Charts

![Model comparison](docs/charts/model_comparison.png)

![AUC vs training time](docs/charts/auc_vs_training_time.png)

![Segment distribution](docs/charts/segment_distribution.png)

![Churn by segment](docs/charts/churn_by_segment.png)

![Precision recall threshold](docs/charts/precision_recall_threshold.png)

---

## GCP setup

- Project: customer-churn-492703
- Dataset: customer_intelligence
- Table: rfm_scores (200,000 rows)
- Scored output: bqml_churn_scores

---

## Quick start

### 1. Clone and install

```
git clone https://github.com/gbhorne/retail-churn-pytorch.git
cd retail-churn-pytorch
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Authenticate to GCP

```
gcloud auth application-default login
gcloud auth login
gcloud config set project customer-churn-492703
gcloud auth application-default set-quota-project customer-churn-492703
```

### 3. Export data from BigQuery

```
python data/export_from_bq.py
```

### 4. Train MLP

```
python train_mlp.py
```

### 5. Train TabNet

```
python train_tabnet.py
```

### 6. Generate all docs

```
python build_docs.py
```

---

## Project structure

| Path | Purpose |
|------|---------|
| data/export_from_bq.py | Pulls rfm_scores from BigQuery to CSV |
| src/features.py | Feature engineering shared by both models |
| src/dataset.py | PyTorch Dataset and WeightedRandomSampler |
| src/mlp.py | 3-layer MLP architecture |
| src/tabnet.py | TabNet wrapper and evaluation |
| src/train.py | MLP training loop with early stopping |
| src/evaluate.py | Metrics and SHAP summary plot |
| train_mlp.py | End-to-end MLP training entrypoint |
| train_tabnet.py | End-to-end TabNet training entrypoint |
| build_docs.py | Generates all charts, SVG diagram, and README |

---

## Exploitation guide

### Understanding the output

The scored output table bqml_churn_scores contains one row per customer with:

- churn_probability: float between 0 and 1. Model confidence the customer
  will not purchase again within the scoring window.
- churn_risk: High (>=0.7), Medium (>=0.4), Low (<0.4). Bucketed for
  campaign targeting.
- rfm_segment: Champions, Loyal, Potential, Recent, At-Risk, Hibernating.
  Derived from RFM scores independently of the churn model.

### Choosing a decision threshold

The default threshold of 0.5 is not always the right business decision.

If your win-back campaign costs $5 and recovers $80 in lifetime value, the
breakeven threshold is 5/80 = 0.063. Flag any customer above 0.063, not 0.5.
This dramatically increases recall at the cost of precision.

If your campaign is expensive (direct mail, sales call), raise the threshold
to 0.7 or higher to maximize precision.

Use the precision-recall chart to pick the threshold that matches your
campaign economics.

### Segment action playbook

| Segment | Churn risk | Recommended action | Channel |
|---------|-----------|-------------------|---------|
| Champions | High | Loyalty reward, early access | Email |
| Champions | Medium | Points bonus, VIP reminder | Email |
| Loyal | High | 10-15% discount on next order | Email + SMS |
| Loyal | Medium | Free shipping on next order | Email |
| At-Risk | High | Win-back offer 15-20% discount | Email + SMS |
| At-Risk | Medium | Re-engagement, product recommendation | Email |
| Potential | High | Onboarding nudge, social proof | Email |
| Recent | High | Second purchase incentive | Email |
| Hibernating | High | Aggressive win-back or suppress | Email |
| Hibernating | Low | Suppress from active campaigns | None |

### Priority scoring query

Rank customers by expected recovery value before running any campaign:

```sql
SELECT
  user_id,
  rfm_segment,
  churn_risk,
  churn_probability,
  monetary,
  ROUND(monetary * churn_probability, 2) AS expected_loss,
  CASE
    WHEN rfm_segment = 'Champions'   THEN 1
    WHEN rfm_segment = 'Loyal'       THEN 2
    WHEN rfm_segment = 'At-Risk'     THEN 3
    WHEN rfm_segment = 'Potential'   THEN 4
    WHEN rfm_segment = 'Recent'      THEN 5
    WHEN rfm_segment = 'Hibernating' THEN 6
  END AS segment_priority
FROM `customer-churn-492703.customer_intelligence.bqml_churn_scores`
WHERE churn_risk = 'High'
ORDER BY segment_priority ASC, expected_loss DESC
LIMIT 1000
```

### High-risk Champions campaign list

```sql
SELECT
  user_id,
  monetary,
  frequency,
  ROUND(churn_probability * 100, 1) AS churn_pct,
  rfm_segment
FROM `customer-churn-492703.customer_intelligence.bqml_churn_scores`
WHERE rfm_segment = 'Champions'
  AND churn_risk  = 'High'
ORDER BY churn_probability DESC
```

### Win-back list

```sql
SELECT
  user_id,
  rfm_segment,
  recency_days,
  monetary,
  ROUND(churn_probability * 100, 1) AS churn_pct
FROM `customer-churn-492703.customer_intelligence.bqml_churn_scores`
WHERE rfm_segment IN ('At-Risk', 'Hibernating')
  AND churn_risk = 'High'
ORDER BY monetary DESC
```

### Suppression list

```sql
SELECT
  user_id,
  rfm_segment,
  monetary,
  churn_probability
FROM `customer-churn-492703.customer_intelligence.bqml_churn_scores`
WHERE rfm_segment = 'Hibernating'
  AND churn_risk  = 'Low'
  AND monetary    < 50
ORDER BY churn_probability ASC
```

### Model drift monitoring

Retrain when any of these conditions are met:

- Monthly ROC-AUC on new scoring data drops more than 0.02 below baseline
- Churn rate in the live scored table shifts more than 5 percentage points
  from the training churn rate of 52.9%
- Business rules around the churn definition change

---

## When to use each model

| Consideration | BQML | MLP | TabNet |
|--------------|------|-----|--------|
| No Python required | Yes | No | No |
| Trains in under 2 minutes | Yes | Yes | No |
| Portable model artifact | No | Yes (.pth) | Yes (.zip) |
| Feature importance | No | Via SHAP | Built-in masks |
| Deploy to Vertex AI endpoint | No | Yes | Yes |
| Best for batch scoring in SQL | Yes | No | No |
| Best for real-time API scoring | No | Yes | Yes |
| Recommended for | SQL pipelines | REST APIs | Explainability audits |

---

## Synthetic data disclaimer

All customer data in this project is synthetically generated using numpy random
distributions. No real customer records, PII, or proprietary retail data was
used. The data was designed to produce realistic RFM behavioral patterns for
the purpose of demonstrating ML pipeline construction.

---

## License

MIT