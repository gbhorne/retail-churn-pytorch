# build_docs.py
# Generates charts and README only. Does NOT auto-commit or push.
# After running, review changes then: git add . && git commit && git push
import os, numpy as np
import matplotlib.pyplot as plt

os.makedirs('docs/charts', exist_ok=True)

plt.rcParams.update({
    'figure.facecolor': '#0F1117', 'axes.facecolor': '#0F1117',
    'axes.edgecolor': '#2E3250', 'axes.labelcolor': '#C8CADB',
    'xtick.color': '#C8CADB', 'ytick.color': '#C8CADB',
    'text.color': '#C8CADB', 'grid.color': '#2E3250',
    'grid.linestyle': '--', 'grid.linewidth': 0.5,
    'font.family': 'DejaVu Sans', 'font.size': 12,
})

COLORS   = {'bqml': '#1D9E75', 'mlp': '#534AB7', 'tab': '#D85A30'}
MODELS   = ['BQML LogReg', 'PyTorch MLP', 'TabNet']
ROC_AUC  = [0.826, 0.834, 0.834]
PR_AUC   = [0.0,   0.809, 0.816]
F1_CHURN = [0.725, 0.790, 0.790]
TRAIN_S  = [86, 42, 1281]
BAR_COLS = [COLORS['bqml'], COLORS['mlp'], COLORS['tab']]

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.patch.set_facecolor('#0F1117')
fig.suptitle('Model comparison: retail churn prediction (reference run, CPU)',
             color='#FFFFFF', fontsize=13, fontweight='bold', y=1.02)
for ax, vals, title, ylabel, ylim, labels in [
    (axes[0], ROC_AUC,  'ROC-AUC',            'AUC',    (0.78, 0.86), [f'{v:.3f}' for v in ROC_AUC]),
    (axes[1], PR_AUC,   'PR-AUC',             'PR-AUC', (0.0,  0.90), ['N/A', '0.809', '0.816']),
    (axes[2], F1_CHURN, 'F1 score (churned)', 'F1',     (0.60, 0.85), [f'{v:.3f}' for v in F1_CHURN]),
]:
    bars = ax.bar(MODELS, vals, color=BAR_COLS, width=0.5, zorder=3)
    ax.set_ylim(ylim); ax.set_title(title, color='#FFFFFF', fontsize=13, fontweight='bold')
    ax.set_ylabel(ylabel, color='#C8CADB', fontsize=11); ax.grid(axis='y', zorder=0)
    for bar, label in zip(bars, labels):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+(ylim[1]-ylim[0])*0.01,
                label, ha='center', va='bottom', color='#FFFFFF', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('docs/charts/model_comparison.png', dpi=200, bbox_inches='tight', facecolor='#0F1117')
plt.close(); print('Saved: docs/charts/model_comparison.png')

fig, ax = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor('#0F1117')
for model, auc, t, color in zip(MODELS, ROC_AUC, TRAIN_S, BAR_COLS):
    ax.scatter(t, auc, color=color, s=250, zorder=5)
    ax.annotate(model, (t, auc), textcoords='offset points', xytext=(12,5), color=color, fontsize=12, fontweight='bold')
ax.set_xlabel('Training time (seconds, reference run on CPU)')
ax.set_ylabel('ROC-AUC')
ax.set_title('AUC vs. training time (reference run)', color='#FFFFFF', fontsize=14, fontweight='bold')
ax.set_xlim(-100,1500); ax.set_ylim(0.81,0.845); ax.grid(zorder=0)
plt.tight_layout()
plt.savefig('docs/charts/auc_vs_training_time.png', dpi=200, bbox_inches='tight', facecolor='#0F1117')
plt.close(); print('Saved: docs/charts/auc_vs_training_time.png')

segments  = ['Champions','Loyal','Potential','Recent','At-Risk','Hibernating']
high_risk = [5.2, 10.2, 39.6, 25.5, 64.8, 85.1]
low_risk  = [94.8, 89.8, 60.4, 74.5, 35.2, 14.9]
x, w = np.arange(len(segments)), 0.35
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor('#0F1117')
ax.bar(x-w/2, high_risk, w, label='Churned',     color='#D85A30', zorder=3)
ax.bar(x+w/2, low_risk,  w, label='Not churned', color='#1D9E75', zorder=3)
ax.set_xticks(x); ax.set_xticklabels(segments, fontsize=11)
ax.set_ylabel('Percentage of segment (%)'); ax.grid(axis='y', zorder=0)
ax.set_title('Churn rate by customer segment', color='#FFFFFF', fontsize=14, fontweight='bold')
ax.legend(facecolor='#1A1D2E', edgecolor='#2E3250', labelcolor='#C8CADB', fontsize=11)
plt.tight_layout()
plt.savefig('docs/charts/churn_by_segment.png', dpi=200, bbox_inches='tight', facecolor='#0F1117')
plt.close(); print('Saved: docs/charts/churn_by_segment.png')

thresholds = [0.3,0.4,0.5,0.6,0.7,0.8]
mlp_prec=[0.68,0.72,0.78,0.83,0.88,0.93]; mlp_rec=[0.93,0.88,0.81,0.72,0.59,0.41]
tab_prec=[0.67,0.71,0.77,0.83,0.89,0.94]; tab_rec=[0.94,0.89,0.81,0.71,0.58,0.39]
fig, ax = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor('#0F1117')
ax.plot(thresholds,mlp_prec,'o-', color=COLORS['mlp'],label='MLP precision',   linewidth=2.5,markersize=7)
ax.plot(thresholds,mlp_rec, 's--',color=COLORS['mlp'],label='MLP recall',      linewidth=2.5,markersize=7)
ax.plot(thresholds,tab_prec,'o-', color=COLORS['tab'],label='TabNet precision', linewidth=2.5,markersize=7)
ax.plot(thresholds,tab_rec, 's--',color=COLORS['tab'],label='TabNet recall',    linewidth=2.5,markersize=7)
ax.axvline(0.5, color='#FFFFFF', linewidth=1.0, linestyle=':', alpha=0.6, label='Default threshold (0.5)')
ax.set_xlabel('Decision threshold'); ax.set_ylabel('Score')
ax.set_title('Precision vs. recall at different thresholds (illustrative)', color='#FFFFFF', fontsize=13, fontweight='bold')
ax.legend(facecolor='#1A1D2E', edgecolor='#2E3250', labelcolor='#C8CADB', fontsize=10)
ax.grid(zorder=0); ax.set_ylim(0.3,1.0)
plt.tight_layout()
plt.savefig('docs/charts/precision_recall_threshold.png', dpi=200, bbox_inches='tight', facecolor='#0F1117')
plt.close(); print('Saved: docs/charts/precision_recall_threshold.png')

seg_names  = ['Hibernating','At-Risk','Potential','Loyal','Recent','Champions']
seg_counts = [70000,40000,30000,24000,20000,16000]
seg_colors = ['#534AB7','#D85A30','#BA7517','#1D9E75','#185FA5','#0F6E56']
fig, ax = plt.subplots(figsize=(9, 9))
fig.patch.set_facecolor('#0F1117')
wedges,texts,autotexts = ax.pie(seg_counts,labels=seg_names,colors=seg_colors,
    autopct='%1.1f%%',startangle=140,
    textprops={'color':'#C8CADB','fontsize':12},
    wedgeprops={'edgecolor':'#0F1117','linewidth':2})
for at in autotexts: at.set_color('#FFFFFF'); at.set_fontsize(11)
ax.set_title('Customer segment distribution (200,000 synthetic customers)',
             color='#FFFFFF', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('docs/charts/segment_distribution.png', dpi=200, bbox_inches='tight', facecolor='#0F1117')
plt.close(); print('Saved: docs/charts/segment_distribution.png')

svg = [
    '<svg width="960" height="540" viewBox="0 0 960 540" xmlns="http://www.w3.org/2000/svg">',
    '  <defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">',
    '    <path d="M2 1L8 5L2 9" fill="none" stroke="#4B5563" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></marker></defs>',
    '  <rect width="960" height="540" fill="#0F1117" rx="14"/>',
    '  <text x="480" y="36" text-anchor="middle" font-family="Arial,Helvetica,sans-serif" font-size="17" font-weight="bold" fill="#FFFFFF">Retail churn prediction pipeline</text>',
    '  <rect x="28" y="58" width="904" height="104" rx="10" fill="#1A1D2E" stroke="#2E3250" stroke-width="1"/>',
    '  <text x="48" y="80" font-family="Arial,Helvetica,sans-serif" font-size="11" fill="#6B7280" font-weight="bold" letter-spacing="1">DATA LAYER</text>',
    '  <rect x="58" y="90" width="190" height="56" rx="8" fill="#085041" stroke="#1D9E75" stroke-width="1"/>',
    '  <text x="153" y="115" text-anchor="middle" font-family="Arial,Helvetica,sans-serif" font-size="12" font-weight="bold" fill="#9FE1CB">Synthetic generator</text>',
    '  <text x="153" y="133" text-anchor="middle" font-family="Arial,Helvetica,sans-serif" font-size="10" fill="#5DCAA5">retail-churn-bqml repo</text>',
    '  <rect x="288" y="90" width="190" height="56" rx="8" fill="#085041" stroke="#1D9E75" stroke-width="1"/>',
    '  <text x="383" y="115" text-anchor="middle" font-family="Arial,Helvetica,sans-serif" font-size="12" font-weight="bold" fill="#9FE1CB">BigQuery rfm_scores</text>',
    '  <text x="383" y="133" text-anchor="middle" font-family="Arial,Helvetica,sans-serif" font-size="10" fill="#5DCAA5">200,000 synthetic customers</text>',
    '  <rect x="518" y="90" width="190" height="56" rx="8" fill="#085041" stroke="#1D9E75" stroke-width="1"/>',
    '  <text x="613" y="115" text-anchor="middle" font-family="Arial,Helvetica,sans-serif" font-size="12" font-weight="bold" fill="#9FE1CB">export_from_bq.py</text>',
    '  <text x="613" y="133" text-anchor="middle" font-family="Arial,Helvetica,sans-serif" font-size="10" fill="#5DCAA5">data/features.csv</text>',
    '  <line x1="248" y1="118" x2="286" y2="118" stroke="#4B5563" stroke-width="1.5" marker-end="url(#arrow)"/>',
    '  <line x1="478" y1="118" x2="516" y2="118" stroke="#4B5563" stroke-width="1.5" marker-end="url(#arrow)"/>',
    '  <rect x="28" y="194" width="904" height="114" rx="10" fill="#1A1D2E" stroke="#2E3250" stroke-width="1"/>',
    '  <text x="48" y="216" font-family="Arial,Helvetica,sans-serif" font-size="11" fill="#6B7280" font-weight="bold" letter-spacing="1">FEATURE ENGINEERING</text>',
    '  <rect x="58" y="226" width="234" height="64" rx="8" fill="#26215C" stroke="#534AB7" stroke-width="1"/>',
    '  <text x="175" y="252" text-anchor="middle" font-family="Arial,Helvetica,sans-serif" font-size="12" font-weight="bold" fill="#CECBF6">RFM scoring</text>',
    '  <text x="175" y="270" text-anchor="middle" font-family="Arial,Helvetica,sans-serif" font-size="10" fill="#AFA9EC">Recency, frequency, monetary</text>',
    '  <rect x="330" y="226" width="234" height="64" rx="8" fill="#26215C" stroke="#534AB7" stroke-width="1"/>',
    '  <text x="447" y="252" text-anchor="middle" font-family="Arial,Helvetica,sans-serif" font-size="12" font-weight="bold" fill="#CECBF6">MLP path</text>',
    '  <text x="447" y="270" text-anchor="middle" font-family="Arial,Helvetica,sans-serif" font-size="10" fill="#AFA9EC">One-hot encode + StandardScaler</text>',
    '  <rect x="602" y="226" width="234" height="64" rx="8" fill="#26215C" stroke="#534AB7" stroke-width="1"/>',
    '  <text x="719" y="252" text-anchor="middle" font-family="Arial,Helvetica,sans-serif" font-size="12" font-weight="bold" fill="#CECBF6">TabNet path</text>',
    '  <text x="719" y="270" text-anchor="middle" font-family="Arial,Helvetica,sans-serif" font-size="10" fill="#AFA9EC">Scale numerics + integer cat IDs</text>',
    '  <line x1="292" y1="258" x2="328" y2="258" stroke="#4B5563" stroke-width="1.5" marker-end="url(#arrow)"/>',
    '  <line x1="564" y1="258" x2="600" y2="258" stroke="#4B5563" stroke-width="1.5" marker-end="url(#arrow)"/>',
    '  <rect x="28" y="342" width="278" height="128" rx="10" fill="#1A1D2E" stroke="#1D9E75" stroke-width="1.5"/>',
    '  <text x="167" y="368" text-anchor="middle" font-family="Arial,Helvetica,sans-serif" font-size="12" fill="#1D9E75" font-weight="bold">BQML logistic regression</text>',
    '  <text x="167" y="390" text-anchor="middle" font-family="Arial,Helvetica,sans-serif" font-size="11" fill="#5DCAA5">SQL-native, no Python</text>',
    '  <text x="167" y="410" text-anchor="middle" font-family="Arial,Helvetica,sans-serif" font-size="11" fill="#5DCAA5">ROC-AUC: 0.826 (reference run)</text>',
    '  <text x="167" y="430" text-anchor="middle" font-family="Arial,Helvetica,sans-serif" font-size="11" fill="#5DCAA5">See: retail-churn-bqml repo</text>',
    '  <rect x="342" y="342" width="278" height="128" rx="10" fill="#1A1D2E" stroke="#534AB7" stroke-width="1.5"/>',
    '  <text x="481" y="368" text-anchor="middle" font-family="Arial,Helvetica,sans-serif" font-size="12" fill="#7F77DD" font-weight="bold">PyTorch MLP</text>',
    '  <text x="481" y="390" text-anchor="middle" font-family="Arial,Helvetica,sans-serif" font-size="11" fill="#AFA9EC">3-layer feedforward, one-hot cats</text>',
    '  <text x="481" y="410" text-anchor="middle" font-family="Arial,Helvetica,sans-serif" font-size="11" fill="#AFA9EC">ROC-AUC: 0.834 (reference run)</text>',
    '  <text x="481" y="430" text-anchor="middle" font-family="Arial,Helvetica,sans-serif" font-size="11" fill="#AFA9EC">SHAP explainability</text>',
    '  <rect x="656" y="342" width="278" height="128" rx="10" fill="#1A1D2E" stroke="#D85A30" stroke-width="1.5"/>',
    '  <text x="795" y="368" text-anchor="middle" font-family="Arial,Helvetica,sans-serif" font-size="12" fill="#D85A30" font-weight="bold">TabNet</text>',
    '  <text x="795" y="390" text-anchor="middle" font-family="Arial,Helvetica,sans-serif" font-size="11" fill="#F0997B">Attention-based, native cat support</text>',
    '  <text x="795" y="410" text-anchor="middle" font-family="Arial,Helvetica,sans-serif" font-size="11" fill="#F0997B">ROC-AUC: 0.834 (reference run)</text>',
    '  <text x="795" y="430" text-anchor="middle" font-family="Arial,Helvetica,sans-serif" font-size="11" fill="#F0997B">Built-in feature masks</text>',
    '  <line x1="613" y1="308" x2="167" y2="340" stroke="#4B5563" stroke-width="1.5" marker-end="url(#arrow)"/>',
    '  <line x1="719" y1="308" x2="481" y2="340" stroke="#4B5563" stroke-width="1.5" marker-end="url(#arrow)"/>',
    '  <line x1="825" y1="308" x2="795" y2="340" stroke="#4B5563" stroke-width="1.5" marker-end="url(#arrow)"/>',
    '  <rect x="28" y="490" width="904" height="38" rx="8" fill="#0F2318" stroke="#1D9E75" stroke-width="1"/>',
    '  <text x="480" y="514" text-anchor="middle" font-family="Arial,Helvetica,sans-serif" font-size="12" fill="#1D9E75" font-weight="bold">Trained artifacts: mlp_churn.pth + scaler, tabnet_churn.zip + metadata</text>',
    '  <line x1="167" y1="470" x2="167" y2="488" stroke="#4B5563" stroke-width="1.5" marker-end="url(#arrow)"/>',
    '  <line x1="481" y1="470" x2="481" y2="488" stroke="#4B5563" stroke-width="1.5" marker-end="url(#arrow)"/>',
    '  <line x1="795" y1="470" x2="795" y2="488" stroke="#4B5563" stroke-width="1.5" marker-end="url(#arrow)"/>',
    '</svg>',
]
with open('docs/charts/architecture.svg', 'w', encoding='utf-8') as f: f.write('\n'.join(svg))
print('Saved: docs/charts/architecture.svg')

readme = [
    '# retail-churn-pytorch',
    '',
    'PyTorch MLP and TabNet churn classifiers trained on a synthetic retail dataset',
    'of 200,000 customers across six behavioral segments using RFM features.',
    '',
    'This repo handles model training, evaluation, and explainability only.',
    'The BQML baseline and SQL-native scoring pipeline live in the companion repo:',
    'https://github.com/gbhorne/retail-churn-bqml',
    '',
    '---',
    '',
    '## Churn label definition',
    '',
    'A customer is labeled as churned (1) if their churn probability exceeds the',
    'segment-level threshold assigned during synthetic data generation.',
    'Churn labels are assigned probabilistically per segment, not derived',
    'deterministically from recency alone. This avoids the leakage pattern where',
    'the label is a direct function of a training feature.',
    '',
    '| Segment | Assigned churn probability |',
    '|---------|--------------------------|',
    '| Champions | 5% |',
    '| Loyal | 10% |',
    '| Potential | 40% |',
    '| Recent | 25% |',
    '| At-Risk | 65% |',
    '| Hibernating | 85% |',
    '',
    '---',
    '',
    '## Architecture',
    '',
    '![Architecture](docs/charts/architecture.svg)',
    '',
    '---',
    '',
    '## Results (reference run, CPU, Windows)',
    '',
    'These numbers are from a single reference run on a Windows CPU machine.',
    'Results may vary across hardware, random seeds, and library versions.',
    '',
    '| Model | ROC-AUC | PR-AUC | F1 (churned) | Training time |',
    '|-------|---------|--------|--------------|---------------|',
    '| BQML logistic regression | 0.826 | N/A | 0.725 | 86s |',
    '| PyTorch MLP | 0.834 | 0.809 | 0.790 | 42s |',
    '| TabNet | 0.834 | 0.816 | 0.790 | 1,281s |',
    '',
    'In this experiment, neither the MLP nor TabNet materially outperformed the',
    'simpler BQML baseline on this feature set. The differences are in',
    'explainability, portability, and deployment flexibility.',
    '',
    '---',
    '',
    '## Key design decisions',
    '',
    '- MLP uses one-hot encoding for categoricals. Label encoding with scaling',
    '  imposes a false ordinal relationship on nominal features.',
    '- TabNet receives categorical columns as integer IDs via cat_idxs and cat_dims,',
    '  not as scaled floats. This is how TabNet is intended to be used.',
    '- WeightedRandomSampler handles class imbalance during MLP training.',
    '  pos_weight is not used alongside oversampling to avoid double-weighting.',
    '- The scaler, feature column order, and label mappings are saved alongside',
    '  model weights so inference artifacts are fully self-contained.',
    '- SHAP uses training data as the background distribution, not test data.',
    '- Global seeds are set for numpy, torch, and random for reproducibility.',
    '',
    '---',
    '',
    '## Charts',
    '',
    '![Model comparison](docs/charts/model_comparison.png)',
    '',
    '![AUC vs. training time](docs/charts/auc_vs_training_time.png)',
    '',
    '![Segment distribution](docs/charts/segment_distribution.png)',
    '',
    '![Churn by segment](docs/charts/churn_by_segment.png)',
    '',
    '![Precision vs. recall threshold](docs/charts/precision_recall_threshold.png)',
    '',
    '![SHAP summary](docs/charts/shap_summary.png)',
    '',
    '![SHAP feature importance](docs/charts/shap_bar.png)',
    '',
    '---',
    '',
    '## When to use each model',
    '',
    '| Consideration | BQML | MLP | TabNet |',
    '|--------------|------|-----|--------|',
    '| No Python required | Yes | No | No |',
    '| Trains in under 2 minutes | Yes | Yes | No |',
    '| Portable model artifact | No | Yes (.pth) | Yes (.zip) |',
    '| Feature importance | No | Via SHAP | Built-in masks |',
    '| Handles categoricals natively | No | Via one-hot | Yes (cat_idxs) |',
    '| Deploy to Vertex AI endpoint | No | Yes | Yes |',
    '| Best for | SQL pipelines | REST APIs | Explainability audits |',
    '',
    '---',
    '',
    '## Project structure',
    '',
    '| Path | Purpose |',
    '|------|---------|',
    '| data/export_from_bq.py | Pulls rfm_scores from BigQuery to CSV |',
    '| src/features.py | Feature engineering with separate MLP and TabNet paths |',
    '| src/dataset.py | PyTorch Dataset and WeightedRandomSampler |',
    '| src/mlp.py | 3-layer MLP architecture |',
    '| src/tabnet.py | TabNet wrapper with cat_idxs and cat_dims |',
    '| src/train.py | MLP training loop with early stopping |',
    '| src/evaluate.py | Metrics and SHAP summary plot |',
    '| train_mlp.py | MLP training entrypoint, saves weights + scaler + feature list |',
    '| train_tabnet.py | TabNet training entrypoint, saves weights + metadata |',
    '| run_shap.py | SHAP explainability using training data as background |',
    '| build_docs.py | Generates charts and README (no auto-push) |',
    '',
    '---',
    '',
    '## Synthetic data disclaimer',
    '',
    'All customer data is synthetically generated. No real customer records,',
    'PII, or proprietary retail data was used. The synthetic dataset is stored',
    'in BigQuery and generated by the companion retail-churn-bqml repo.',
    'The generator script is generate_synthetic_data.py in that repo.',
    '',
    '---',
    '',
    '## License',
    '',
    'MIT',
]
with open('README.md', 'w', encoding='utf-8') as f: f.write('\n'.join(readme))
print('README.md generated.')
print('\nReview changes then run:')
print('  git add .')
print('  git commit -m "fix: categorical handling, TabNet cat_idxs, remove pos_weight, save artifacts"')
print('  git push')