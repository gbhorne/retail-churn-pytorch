# build_docs.py
# Generates all charts, SVG architecture diagram, and README.md
# then commits and pushes to GitHub.
# Run: python build_docs.py

import os
import subprocess
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("docs/charts", exist_ok=True)

# ── shared style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0F1117",
    "axes.facecolor":   "#0F1117",
    "axes.edgecolor":   "#2E3250",
    "axes.labelcolor":  "#C8CADB",
    "xtick.color":      "#C8CADB",
    "ytick.color":      "#C8CADB",
    "text.color":       "#C8CADB",
    "grid.color":       "#2E3250",
    "grid.linestyle":   "--",
    "grid.linewidth":   0.5,
    "font.family":      "DejaVu Sans",
    "font.size":        11,
})

COLORS   = {"bqml": "#1D9E75", "mlp": "#534AB7", "tab": "#D85A30"}
MODELS   = ["BQML LogReg", "PyTorch MLP", "TabNet"]
ROC_AUC  = [0.826, 0.834, 0.834]
PR_AUC   = [0.0,   0.809, 0.816]
F1_CHURN = [0.725, 0.790, 0.790]
TRAIN_S  = [86,    42,    1281]
BAR_COLS = [COLORS["bqml"], COLORS["mlp"], COLORS["tab"]]

# ── chart 1: model comparison ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.patch.set_facecolor("#0F1117")
fig.suptitle("Model comparison — retail churn prediction",
             color="#FFFFFF", fontsize=14, fontweight="bold", y=1.02)

for ax, vals, title, ylabel, ylim, labels in [
    (axes[0], ROC_AUC,  "ROC-AUC",            "AUC",    (0.78, 0.86), [f"{v:.3f}" for v in ROC_AUC]),
    (axes[1], PR_AUC,   "PR-AUC",             "PR-AUC", (0.0,  0.90), ["N/A", "0.809", "0.816"]),
    (axes[2], F1_CHURN, "F1 score (churned)", "F1",     (0.60, 0.85), [f"{v:.3f}" for v in F1_CHURN]),
]:
    bars = ax.bar(MODELS, vals, color=BAR_COLS, width=0.5, zorder=3)
    ax.set_ylim(ylim)
    ax.set_title(title, color="#FFFFFF", fontsize=12)
    ax.set_ylabel(ylabel, color="#C8CADB")
    ax.grid(axis="y", zorder=0)
    for bar, label in zip(bars, labels):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (ylim[1] - ylim[0]) * 0.01,
                label, ha="center", va="bottom", color="#FFFFFF", fontsize=10)

plt.tight_layout()
plt.savefig("docs/charts/model_comparison.png", dpi=150,
            bbox_inches="tight", facecolor="#0F1117")
plt.close()
print("Saved: docs/charts/model_comparison.png")

# ── chart 2: AUC vs training time ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor("#0F1117")
for model, auc, t, color in zip(MODELS, ROC_AUC, TRAIN_S, BAR_COLS):
    ax.scatter(t, auc, color=color, s=200, zorder=5)
    ax.annotate(model, (t, auc), textcoords="offset points",
                xytext=(10, 4), color=color, fontsize=11, fontweight="bold")
ax.set_xlabel("Training time (seconds)")
ax.set_ylabel("ROC-AUC")
ax.set_title("AUC vs training time", color="#FFFFFF",
             fontsize=13, fontweight="bold")
ax.set_xlim(-100, 1500)
ax.set_ylim(0.81, 0.845)
ax.grid(zorder=0)
plt.tight_layout()
plt.savefig("docs/charts/auc_vs_training_time.png", dpi=150,
            bbox_inches="tight", facecolor="#0F1117")
plt.close()
print("Saved: docs/charts/auc_vs_training_time.png")

# ── chart 3: churn rate by segment ────────────────────────────────────────────
segments  = ["Champions", "Loyal", "Potential", "Recent", "At-Risk", "Hibernating"]
high_risk = [5.2,  10.2, 39.6, 25.5, 64.8, 85.1]
low_risk  = [94.8, 89.8, 60.4, 74.5, 35.2, 14.9]
x, w = np.arange(len(segments)), 0.35
fig, ax = plt.subplots(figsize=(11, 5))
fig.patch.set_facecolor("#0F1117")
ax.bar(x - w/2, high_risk, w, label="Churned",     color="#D85A30", zorder=3)
ax.bar(x + w/2, low_risk,  w, label="Not churned", color="#1D9E75", zorder=3)
ax.set_xticks(x)
ax.set_xticklabels(segments)
ax.set_ylabel("Percentage of segment (%)")
ax.set_title("Churn rate by customer segment", color="#FFFFFF",
             fontsize=13, fontweight="bold")
ax.legend(facecolor="#1A1D2E", edgecolor="#2E3250", labelcolor="#C8CADB")
ax.grid(axis="y", zorder=0)
plt.tight_layout()
plt.savefig("docs/charts/churn_by_segment.png", dpi=150,
            bbox_inches="tight", facecolor="#0F1117")
plt.close()
print("Saved: docs/charts/churn_by_segment.png")

# ── chart 4: precision-recall at thresholds ───────────────────────────────────
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
mlp_prec   = [0.68, 0.72, 0.78, 0.83, 0.88, 0.93]
mlp_rec    = [0.93, 0.88, 0.81, 0.72, 0.59, 0.41]
tab_prec   = [0.67, 0.71, 0.77, 0.83, 0.89, 0.94]
tab_rec    = [0.94, 0.89, 0.81, 0.71, 0.58, 0.39]
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor("#0F1117")
ax.plot(thresholds, mlp_prec, "o-",  color=COLORS["mlp"], label="MLP precision",   linewidth=2)
ax.plot(thresholds, mlp_rec,  "s--", color=COLORS["mlp"], label="MLP recall",      linewidth=2)
ax.plot(thresholds, tab_prec, "o-",  color=COLORS["tab"], label="TabNet precision", linewidth=2)
ax.plot(thresholds, tab_rec,  "s--", color=COLORS["tab"], label="TabNet recall",    linewidth=2)
ax.axvline(0.5, color="#FFFFFF", linewidth=0.8, linestyle=":", alpha=0.5)
ax.set_xlabel("Decision threshold")
ax.set_ylabel("Score")
ax.set_title("Precision vs recall at different thresholds",
             color="#FFFFFF", fontsize=13, fontweight="bold")
ax.legend(facecolor="#1A1D2E", edgecolor="#2E3250", labelcolor="#C8CADB", fontsize=9)
ax.grid(zorder=0)
ax.set_ylim(0.3, 1.0)
plt.tight_layout()
plt.savefig("docs/charts/precision_recall_threshold.png", dpi=150,
            bbox_inches="tight", facecolor="#0F1117")
plt.close()
print("Saved: docs/charts/precision_recall_threshold.png")

# ── chart 5: segment distribution pie ─────────────────────────────────────────
seg_names  = ["Hibernating", "At-Risk", "Potential", "Loyal", "Recent", "Champions"]
seg_counts = [70000, 40000, 30000, 24000, 20000, 16000]
seg_colors = ["#534AB7", "#D85A30", "#BA7517", "#1D9E75", "#185FA5", "#0F6E56"]
fig, ax = plt.subplots(figsize=(8, 8))
fig.patch.set_facecolor("#0F1117")
wedges, texts, autotexts = ax.pie(
    seg_counts, labels=seg_names, colors=seg_colors,
    autopct="%1.1f%%", startangle=140,
    textprops={"color": "#C8CADB", "fontsize": 11},
    wedgeprops={"edgecolor": "#0F1117", "linewidth": 2}
)
for at in autotexts:
    at.set_color("#FFFFFF")
    at.set_fontsize(10)
ax.set_title("Customer segment distribution (200,000 customers)",
             color="#FFFFFF", fontsize=13, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig("docs/charts/segment_distribution.png", dpi=150,
            bbox_inches="tight", facecolor="#0F1117")
plt.close()
print("Saved: docs/charts/segment_distribution.png")

# ── SVG architecture diagram ───────────────────────────────────────────────────
svg_lines = [
    '<svg width="900" height="520" viewBox="0 0 900 520" xmlns="http://www.w3.org/2000/svg">',
    '  <defs>',
    '    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">',
    '      <path d="M2 1L8 5L2 9" fill="none" stroke="#6B7280" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>',
    '    </marker>',
    '  </defs>',
    '  <rect width="900" height="520" fill="#0F1117" rx="12"/>',
    '  <text x="450" y="38" text-anchor="middle" font-family="DejaVu Sans" font-size="16" font-weight="bold" fill="#FFFFFF">Retail churn prediction pipeline</text>',
    '  <rect x="30" y="60" width="840" height="100" rx="10" fill="#1A1D2E" stroke="#2E3250" stroke-width="1"/>',
    '  <text x="50" y="85" font-family="DejaVu Sans" font-size="11" fill="#6B7280" font-weight="bold">DATA LAYER</text>',
    '  <rect x="60" y="95" width="180" height="50" rx="8" fill="#0F6E56" stroke="#1D9E75" stroke-width="1"/>',
    '  <text x="150" y="118" text-anchor="middle" font-family="DejaVu Sans" font-size="12" font-weight="bold" fill="#9FE1CB">TheLook ecommerce</text>',
    '  <text x="150" y="134" text-anchor="middle" font-family="DejaVu Sans" font-size="10" fill="#5DCAA5">bigquery-public-data</text>',
    '  <rect x="280" y="95" width="180" height="50" rx="8" fill="#0F6E56" stroke="#1D9E75" stroke-width="1"/>',
    '  <text x="370" y="118" text-anchor="middle" font-family="DejaVu Sans" font-size="12" font-weight="bold" fill="#9FE1CB">Synthetic generator</text>',
    '  <text x="370" y="134" text-anchor="middle" font-family="DejaVu Sans" font-size="10" fill="#5DCAA5">generate_synthetic_data.py</text>',
    '  <rect x="500" y="95" width="180" height="50" rx="8" fill="#085041" stroke="#1D9E75" stroke-width="1"/>',
    '  <text x="590" y="118" text-anchor="middle" font-family="DejaVu Sans" font-size="12" font-weight="bold" fill="#9FE1CB">rfm_scores</text>',
    '  <text x="590" y="134" text-anchor="middle" font-family="DejaVu Sans" font-size="10" fill="#5DCAA5">200,000 customers</text>',
    '  <line x1="240" y1="120" x2="278" y2="120" stroke="#6B7280" stroke-width="1" marker-end="url(#arrow)"/>',
    '  <line x1="460" y1="120" x2="498" y2="120" stroke="#6B7280" stroke-width="1" marker-end="url(#arrow)"/>',
    '  <rect x="30" y="190" width="840" height="110" rx="10" fill="#1A1D2E" stroke="#2E3250" stroke-width="1"/>',
    '  <text x="50" y="215" font-family="DejaVu Sans" font-size="11" fill="#6B7280" font-weight="bold">FEATURE ENGINEERING</text>',
    '  <rect x="60" y="225" width="220" height="60" rx="8" fill="#26215C" stroke="#534AB7" stroke-width="1"/>',
    '  <text x="170" y="251" text-anchor="middle" font-family="DejaVu Sans" font-size="12" font-weight="bold" fill="#CECBF6">RFM scoring</text>',
    '  <text x="170" y="267" text-anchor="middle" font-family="DejaVu Sans" font-size="10" fill="#AFA9EC">recency, frequency, monetary</text>',
    '  <rect x="320" y="225" width="220" height="60" rx="8" fill="#26215C" stroke="#534AB7" stroke-width="1"/>',
    '  <text x="430" y="251" text-anchor="middle" font-family="DejaVu Sans" font-size="12" font-weight="bold" fill="#CECBF6">Log scaling</text>',
    '  <text x="430" y="267" text-anchor="middle" font-family="DejaVu Sans" font-size="10" fill="#AFA9EC">skew compression</text>',
    '  <rect x="580" y="225" width="220" height="60" rx="8" fill="#26215C" stroke="#534AB7" stroke-width="1"/>',
    '  <text x="690" y="251" text-anchor="middle" font-family="DejaVu Sans" font-size="12" font-weight="bold" fill="#CECBF6">Interactions</text>',
    '  <text x="690" y="267" text-anchor="middle" font-family="DejaVu Sans" font-size="10" fill="#AFA9EC">recency x frequency</text>',
    '  <line x1="280" y1="255" x2="318" y2="255" stroke="#6B7280" stroke-width="1" marker-end="url(#arrow)"/>',
    '  <line x1="540" y1="255" x2="578" y2="255" stroke="#6B7280" stroke-width="1" marker-end="url(#arrow)"/>',
    '  <rect x="30" y="330" width="260" height="120" rx="10" fill="#1A1D2E" stroke="#1D9E75" stroke-width="1.5"/>',
    '  <text x="160" y="355" text-anchor="middle" font-family="DejaVu Sans" font-size="11" fill="#1D9E75" font-weight="bold">BQML LOGISTIC REG</text>',
    '  <text x="160" y="374" text-anchor="middle" font-family="DejaVu Sans" font-size="10" fill="#5DCAA5">CREATE MODEL in SQL</text>',
    '  <text x="160" y="392" text-anchor="middle" font-family="DejaVu Sans" font-size="10" fill="#5DCAA5">ROC-AUC: 0.826</text>',
    '  <text x="160" y="410" text-anchor="middle" font-family="DejaVu Sans" font-size="10" fill="#5DCAA5">Training: 86 seconds</text>',
    '  <text x="160" y="428" text-anchor="middle" font-family="DejaVu Sans" font-size="10" fill="#5DCAA5">No Python required</text>',
    '  <rect x="320" y="330" width="260" height="120" rx="10" fill="#1A1D2E" stroke="#534AB7" stroke-width="1.5"/>',
    '  <text x="450" y="355" text-anchor="middle" font-family="DejaVu Sans" font-size="11" fill="#7F77DD" font-weight="bold">PYTORCH MLP</text>',
    '  <text x="450" y="374" text-anchor="middle" font-family="DejaVu Sans" font-size="10" fill="#AFA9EC">3-layer feedforward network</text>',
    '  <text x="450" y="392" text-anchor="middle" font-family="DejaVu Sans" font-size="10" fill="#AFA9EC">ROC-AUC: 0.834</text>',
    '  <text x="450" y="410" text-anchor="middle" font-family="DejaVu Sans" font-size="10" fill="#AFA9EC">Training: 42 seconds</text>',
    '  <text x="450" y="428" text-anchor="middle" font-family="DejaVu Sans" font-size="10" fill="#AFA9EC">SHAP explainability</text>',
    '  <rect x="610" y="330" width="260" height="120" rx="10" fill="#1A1D2E" stroke="#D85A30" stroke-width="1.5"/>',
    '  <text x="740" y="355" text-anchor="middle" font-family="DejaVu Sans" font-size="11" fill="#D85A30" font-weight="bold">TABNET</text>',
    '  <text x="740" y="374" text-anchor="middle" font-family="DejaVu Sans" font-size="10" fill="#F0997B">Attention-based tabular model</text>',
    '  <text x="740" y="392" text-anchor="middle" font-family="DejaVu Sans" font-size="10" fill="#F0997B">ROC-AUC: 0.834</text>',
    '  <text x="740" y="410" text-anchor="middle" font-family="DejaVu Sans" font-size="10" fill="#F0997B">Training: 1281 seconds</text>',
    '  <text x="740" y="428" text-anchor="middle" font-family="DejaVu Sans" font-size="10" fill="#F0997B">Built-in feature masks</text>',
    '  <line x1="590" y1="300" x2="160" y2="330" stroke="#6B7280" stroke-width="1" marker-end="url(#arrow)"/>',
    '  <line x1="690" y1="300" x2="450" y2="330" stroke="#6B7280" stroke-width="1" marker-end="url(#arrow)"/>',
    '  <line x1="790" y1="300" x2="740" y2="330" stroke="#6B7280" stroke-width="1" marker-end="url(#arrow)"/>',
    '  <rect x="30" y="470" width="840" height="36" rx="8" fill="#0F2318" stroke="#1D9E75" stroke-width="1"/>',
    '  <text x="450" y="493" text-anchor="middle" font-family="DejaVu Sans" font-size="11" fill="#1D9E75" font-weight="bold">OUTPUT: bqml_churn_scores -- churn_probability, churn_risk, rfm_segment per customer</text>',
    '  <line x1="160" y1="450" x2="160" y2="468" stroke="#6B7280" stroke-width="1" marker-end="url(#arrow)"/>',
    '  <line x1="450" y1="450" x2="450" y2="468" stroke="#6B7280" stroke-width="1" marker-end="url(#arrow)"/>',
    '  <line x1="740" y1="450" x2="740" y2="468" stroke="#6B7280" stroke-width="1" marker-end="url(#arrow)"/>',
    '</svg>',
]

with open("docs/charts/architecture.svg", "w", encoding="utf-8") as f:
    f.write("\n".join(svg_lines))
print("Saved: docs/charts/architecture.svg")

# ── README ────────────────────────────────────────────────────────────────────
readme_lines = [
    "# retail-churn-pytorch",
    "",
    "Retail customer churn prediction using PyTorch MLP and TabNet trained on 200,000",
    "synthetic retail customers across six behavioral segments with RFM features.",
    "Part of a two-repo comparison against a BigQuery ML logistic regression baseline.",
    "",
    "Companion repo: https://github.com/gbhorne/retail-churn-bqml",
    "",
    "---",
    "",
    "## Architecture",
    "",
    "![Architecture](docs/charts/architecture.svg)",
    "",
    "---",
    "",
    "## Results",
    "",
    "| Model | ROC-AUC | PR-AUC | F1 (churned) | Training time |",
    "|-------|---------|--------|--------------|---------------|",
    "| BQML Logistic Regression | 0.826 | N/A | 0.725 | 86s |",
    "| PyTorch MLP | 0.834 | 0.809 | 0.790 | 42s |",
    "| TabNet | 0.834 | 0.816 | 0.790 | 1281s |",
    "",
    "All three models land within 0.008 AUC of each other. Neural network complexity",
    "does not meaningfully improve accuracy on clean tabular RFM data. The real",
    "differences are explainability, portability, and infrastructure cost.",
    "",
    "---",
    "",
    "## Charts",
    "",
    "![Model comparison](docs/charts/model_comparison.png)",
    "",
    "![AUC vs training time](docs/charts/auc_vs_training_time.png)",
    "",
    "![Segment distribution](docs/charts/segment_distribution.png)",
    "",
    "![Churn by segment](docs/charts/churn_by_segment.png)",
    "",
    "![Precision recall threshold](docs/charts/precision_recall_threshold.png)",
    "",
    "---",
    "",
    "## GCP setup",
    "",
    "- Project: customer-churn-492703",
    "- Dataset: customer_intelligence",
    "- Table: rfm_scores (200,000 rows)",
    "- Scored output: bqml_churn_scores",
    "",
    "---",
    "",
    "## Quick start",
    "",
    "### 1. Clone and install",
    "",
    "```",
    "git clone https://github.com/gbhorne/retail-churn-pytorch.git",
    "cd retail-churn-pytorch",
    "python -m venv .venv",
    ".venv\\Scripts\\Activate.ps1",
    "pip install -r requirements.txt",
    "```",
    "",
    "### 2. Authenticate to GCP",
    "",
    "```",
    "gcloud auth application-default login",
    "gcloud auth login",
    "gcloud config set project customer-churn-492703",
    "gcloud auth application-default set-quota-project customer-churn-492703",
    "```",
    "",
    "### 3. Export data from BigQuery",
    "",
    "```",
    "python data/export_from_bq.py",
    "```",
    "",
    "### 4. Train MLP",
    "",
    "```",
    "python train_mlp.py",
    "```",
    "",
    "### 5. Train TabNet",
    "",
    "```",
    "python train_tabnet.py",
    "```",
    "",
    "### 6. Generate all docs",
    "",
    "```",
    "python build_docs.py",
    "```",
    "",
    "---",
    "",
    "## Project structure",
    "",
    "| Path | Purpose |",
    "|------|---------|",
    "| data/export_from_bq.py | Pulls rfm_scores from BigQuery to CSV |",
    "| src/features.py | Feature engineering shared by both models |",
    "| src/dataset.py | PyTorch Dataset and WeightedRandomSampler |",
    "| src/mlp.py | 3-layer MLP architecture |",
    "| src/tabnet.py | TabNet wrapper and evaluation |",
    "| src/train.py | MLP training loop with early stopping |",
    "| src/evaluate.py | Metrics and SHAP summary plot |",
    "| train_mlp.py | End-to-end MLP training entrypoint |",
    "| train_tabnet.py | End-to-end TabNet training entrypoint |",
    "| build_docs.py | Generates all charts, SVG diagram, and README |",
    "",
    "---",
    "",
    "## Exploitation guide",
    "",
    "### Understanding the output",
    "",
    "The scored output table bqml_churn_scores contains one row per customer with:",
    "",
    "- churn_probability: float between 0 and 1. Model confidence the customer",
    "  will not purchase again within the scoring window.",
    "- churn_risk: High (>=0.7), Medium (>=0.4), Low (<0.4). Bucketed for",
    "  campaign targeting.",
    "- rfm_segment: Champions, Loyal, Potential, Recent, At-Risk, Hibernating.",
    "  Derived from RFM scores independently of the churn model.",
    "",
    "### Choosing a decision threshold",
    "",
    "The default threshold of 0.5 is not always the right business decision.",
    "",
    "If your win-back campaign costs $5 and recovers $80 in lifetime value, the",
    "breakeven threshold is 5/80 = 0.063. Flag any customer above 0.063, not 0.5.",
    "This dramatically increases recall at the cost of precision.",
    "",
    "If your campaign is expensive (direct mail, sales call), raise the threshold",
    "to 0.7 or higher to maximize precision.",
    "",
    "Use the precision-recall chart to pick the threshold that matches your",
    "campaign economics.",
    "",
    "### Segment action playbook",
    "",
    "| Segment | Churn risk | Recommended action | Channel |",
    "|---------|-----------|-------------------|---------|",
    "| Champions | High | Loyalty reward, early access | Email |",
    "| Champions | Medium | Points bonus, VIP reminder | Email |",
    "| Loyal | High | 10-15% discount on next order | Email + SMS |",
    "| Loyal | Medium | Free shipping on next order | Email |",
    "| At-Risk | High | Win-back offer 15-20% discount | Email + SMS |",
    "| At-Risk | Medium | Re-engagement, product recommendation | Email |",
    "| Potential | High | Onboarding nudge, social proof | Email |",
    "| Recent | High | Second purchase incentive | Email |",
    "| Hibernating | High | Aggressive win-back or suppress | Email |",
    "| Hibernating | Low | Suppress from active campaigns | None |",
    "",
    "### Priority scoring query",
    "",
    "Rank customers by expected recovery value before running any campaign:",
    "",
    "```sql",
    "SELECT",
    "  user_id,",
    "  rfm_segment,",
    "  churn_risk,",
    "  churn_probability,",
    "  monetary,",
    "  ROUND(monetary * churn_probability, 2) AS expected_loss,",
    "  CASE",
    "    WHEN rfm_segment = 'Champions'   THEN 1",
    "    WHEN rfm_segment = 'Loyal'       THEN 2",
    "    WHEN rfm_segment = 'At-Risk'     THEN 3",
    "    WHEN rfm_segment = 'Potential'   THEN 4",
    "    WHEN rfm_segment = 'Recent'      THEN 5",
    "    WHEN rfm_segment = 'Hibernating' THEN 6",
    "  END AS segment_priority",
    "FROM `customer-churn-492703.customer_intelligence.bqml_churn_scores`",
    "WHERE churn_risk = 'High'",
    "ORDER BY segment_priority ASC, expected_loss DESC",
    "LIMIT 1000",
    "```",
    "",
    "### High-risk Champions campaign list",
    "",
    "```sql",
    "SELECT",
    "  user_id,",
    "  monetary,",
    "  frequency,",
    "  ROUND(churn_probability * 100, 1) AS churn_pct,",
    "  rfm_segment",
    "FROM `customer-churn-492703.customer_intelligence.bqml_churn_scores`",
    "WHERE rfm_segment = 'Champions'",
    "  AND churn_risk  = 'High'",
    "ORDER BY churn_probability DESC",
    "```",
    "",
    "### Win-back list",
    "",
    "```sql",
    "SELECT",
    "  user_id,",
    "  rfm_segment,",
    "  recency_days,",
    "  monetary,",
    "  ROUND(churn_probability * 100, 1) AS churn_pct",
    "FROM `customer-churn-492703.customer_intelligence.bqml_churn_scores`",
    "WHERE rfm_segment IN ('At-Risk', 'Hibernating')",
    "  AND churn_risk = 'High'",
    "ORDER BY monetary DESC",
    "```",
    "",
    "### Suppression list",
    "",
    "```sql",
    "SELECT",
    "  user_id,",
    "  rfm_segment,",
    "  monetary,",
    "  churn_probability",
    "FROM `customer-churn-492703.customer_intelligence.bqml_churn_scores`",
    "WHERE rfm_segment = 'Hibernating'",
    "  AND churn_risk  = 'Low'",
    "  AND monetary    < 50",
    "ORDER BY churn_probability ASC",
    "```",
    "",
    "### Model drift monitoring",
    "",
    "Retrain when any of these conditions are met:",
    "",
    "- Monthly ROC-AUC on new scoring data drops more than 0.02 below baseline",
    "- Churn rate in the live scored table shifts more than 5 percentage points",
    "  from the training churn rate of 52.9%",
    "- Business rules around the churn definition change",
    "",
    "---",
    "",
    "## When to use each model",
    "",
    "| Consideration | BQML | MLP | TabNet |",
    "|--------------|------|-----|--------|",
    "| No Python required | Yes | No | No |",
    "| Trains in under 2 minutes | Yes | Yes | No |",
    "| Portable model artifact | No | Yes (.pth) | Yes (.zip) |",
    "| Feature importance | No | Via SHAP | Built-in masks |",
    "| Deploy to Vertex AI endpoint | No | Yes | Yes |",
    "| Best for batch scoring in SQL | Yes | No | No |",
    "| Best for real-time API scoring | No | Yes | Yes |",
    "| Recommended for | SQL pipelines | REST APIs | Explainability audits |",
    "",
    "---",
    "",
    "## Synthetic data disclaimer",
    "",
    "All customer data in this project is synthetically generated using numpy random",
    "distributions. No real customer records, PII, or proprietary retail data was",
    "used. The data was designed to produce realistic RFM behavioral patterns for",
    "the purpose of demonstrating ML pipeline construction.",
    "",
    "---",
    "",
    "## License",
    "",
    "MIT",
]

with open("README.md", "w", encoding="utf-8") as f:
    f.write("\n".join(readme_lines))
print("README.md generated.")

# ── git commit and push ───────────────────────────────────────────────────────
commands = [
    ["git", "add", "."],
    ["git", "commit", "-m",
     "docs: full README, exploitation guide, 5 charts, SVG architecture diagram"],
    ["git", "push"],
]

for cmd in commands:
    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)

print("\nDone. All assets generated and pushed.")