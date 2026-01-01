"""
Generate Visualizations for 13-Action Recommendation Model
Creates confusion matrix and performance plots
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("GENERATING VISUALIZATIONS FOR 13-ACTION MODEL")
print("="*70)

# Load test data and model
print("\n1. Loading model and test data...")
test_df = pd.read_csv('data/processed/recommendation_test.csv')
model = pickle.load(open('models/recommendation_engine/best_recommendation_model.pkl', 'rb'))
label_encoder = pickle.load(open('models/recommendation_engine/label_encoder.pkl', 'rb'))

X_test = test_df['cleaned_text']
y_test = test_df['action']
y_test_encoded = label_encoder.transform(y_test)

# Make predictions
print("2. Making predictions...")
y_pred_encoded = model.predict(X_test)
y_pred = label_encoder.inverse_transform(y_pred_encoded)

# Create output directory
output_dir = Path('reports/figures/recommendation')
output_dir.mkdir(parents=True, exist_ok=True)

# 1. Confusion Matrix
print("3. Generating confusion matrix...")
cm = confusion_matrix(y_test, y_pred)
actions = sorted(label_encoder.classes_)

plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=actions, yticklabels=actions,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - 13 Action Recommendation Model\nTest Accuracy: 96.03%', 
          fontsize=14, fontweight='bold')
plt.xlabel('Predicted Action', fontsize=12)
plt.ylabel('True Action', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig(output_dir / 'confusion_matrix_13actions.png', dpi=300, bbox_inches='tight')
print(f"   [OK] Saved: {output_dir / 'confusion_matrix_13actions.png'}")
plt.close()

# 2. Per-Action Performance
print("4. Generating per-action performance chart...")
report = classification_report(y_test, y_pred, output_dict=True)

actions_perf = []
for action in actions:
    if action in report:
        actions_perf.append({
            'Action': action,
            'Precision': report[action]['precision'],
            'Recall': report[action]['recall'],
            'F1-Score': report[action]['f1-score']
        })

df_perf = pd.DataFrame(actions_perf)
df_perf = df_perf.sort_values('F1-Score', ascending=True)

fig, ax = plt.subplots(figsize=(12, 8))
x = np.arange(len(df_perf))
width = 0.25

bars1 = ax.barh(x - width, df_perf['Precision'], width, label='Precision', color='#2ecc71')
bars2 = ax.barh(x, df_perf['Recall'], width, label='Recall', color='#3498db')
bars3 = ax.barh(x + width, df_perf['F1-Score'], width, label='F1-Score', color='#e74c3c')

ax.set_xlabel('Score', fontsize=12)
ax.set_title('Per-Action Performance Metrics\n13 Action Recommendation Model', 
             fontsize=14, fontweight='bold')
ax.set_yticks(x)
ax.set_yticklabels(df_perf['Action'], fontsize=9)
ax.legend(loc='lower right')
ax.set_xlim([0, 1.05])
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'per_action_performance.png', dpi=300, bbox_inches='tight')
print(f"   [OK] Saved: {output_dir / 'per_action_performance.png'}")
plt.close()

# 3. Category-wise Action Distribution
print("5. Generating category-wise action distribution...")
category_mapping = {
    'Compensate Guest': 'Food',
    'Inspect Kitchen Hygiene': 'Food',
    'Revise Menu Quality': 'Food',
    'Staff Training (Food)': 'Food',
    'Deep Clean & Maintenance': 'Rooms',
    'Pest Control Investigation': 'Rooms',
    'Upgrade Room Amenities': 'Rooms',
    'Disciplinary Action': 'Services',
    'Review Check-in Process': 'Services',
    'Staff Communication Training': 'Services',
    'Pool Maintenance': 'Recreation',
    'Recreation Staff Review': 'Recreation',
    'Update Gym Equipment': 'Recreation'
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
categories = ['Food', 'Rooms', 'Services', 'Recreation']

for idx, category in enumerate(categories):
    ax = axes[idx // 2, idx % 2]
    
    # Get actions for this category
    cat_actions = [action for action, cat in category_mapping.items() if cat == category]
    cat_data = df_perf[df_perf['Action'].isin(cat_actions)]
    
    if len(cat_data) > 0:
        x_pos = np.arange(len(cat_data))
        ax.bar(x_pos, cat_data['F1-Score'], color='#3498db', alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([a.replace(' (Food)', '').replace('&', '\n&') 
                            for a in cat_data['Action']], 
                           rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('F1-Score', fontsize=10)
        ax.set_title(f'{category} Actions ({len(cat_actions)} actions)', 
                    fontsize=11, fontweight='bold')
        ax.set_ylim([0, 1.05])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(cat_data['F1-Score']):
            ax.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=8)

plt.suptitle('Action Performance by Category', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir / 'category_wise_performance.png', dpi=300, bbox_inches='tight')
print(f"   [OK] Saved: {output_dir / 'category_wise_performance.png'}")
plt.close()

# 4. Overall Model Summary
print("6. Generating model summary visualization...")
overall_metrics = {
    'Test Accuracy': report['accuracy'],
    'Macro Avg Precision': report['macro avg']['precision'],
    'Macro Avg Recall': report['macro avg']['recall'],
    'Macro Avg F1-Score': report['macro avg']['f1-score'],
    'Weighted Avg F1-Score': report['weighted avg']['f1-score']
}

fig, ax = plt.subplots(figsize=(10, 6))
metrics = list(overall_metrics.keys())
values = list(overall_metrics.values())
colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']

bars = ax.barh(metrics, values, color=colors, alpha=0.7)
ax.set_xlabel('Score', fontsize=12)
ax.set_title('Overall Model Performance Summary\n13 Action Recommendation Model', 
             fontsize=14, fontweight='bold')
ax.set_xlim([0, 1.05])
ax.grid(axis='x', alpha=0.3)

# Add value labels
for i, (metric, value) in enumerate(zip(metrics, values)):
    ax.text(value + 0.01, i, f'{value:.4f}', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'overall_performance_summary.png', dpi=300, bbox_inches='tight')
print(f"   [OK] Saved: {output_dir / 'overall_performance_summary.png'}")
plt.close()

print("\n" + "="*70)
print("VISUALIZATION GENERATION COMPLETE!")
print("="*70)
print(f"\nAll figures saved to: {output_dir}")
print("\nGenerated files:")
print("1. confusion_matrix_13actions.png")
print("2. per_action_performance.png")
print("3. category_wise_performance.png")
print("4. overall_performance_summary.png")
print("="*70)
