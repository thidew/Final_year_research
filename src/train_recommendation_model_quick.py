"""
Quick Recommendation Model Training - Simplified for 13 Actions
Uses pre-optimized hyperparameters for faster training
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

print("="*60)
print("QUICK RECOMMENDATION ENGINE TRAINING (13 Actions)")
print("="*60)

# Load data
print("\nLoading preprocessed data...")
train = pd.read_csv("data/processed/recommendation_train.csv")
val = pd.read_csv("data/processed/recommendation_val.csv")
test = pd.read_csv("data/processed/recommendation_test.csv")

print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
print(f"Unique actions: {train['action'].nunique()}")

# Prepare features
X_train = train['cleaned_text']
y_train = train['action']
X_val = val['cleaned_text']
y_val = val['action']
X_test = test['cleaned_text']
y_test = test['action']

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.transform(y_val)
y_test_encoded = label_encoder.transform(y_test)

print("\nAction classes:")
for i, action in enumerate(label_encoder.classes_):
    print(f"{i+1}. {action}")

# Create pipeline with optimized parameters
print("\nTraining Random Forest model...")
pipeline = ImbPipeline([
    ('tfidf', TfidfVectorizer(
        stop_words='english',
        max_features=5000,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.9
    )),
    ('smote', SMOTE(random_state=42, k_neighbors=5)),
    ('clf', RandomForestClassifier(
        n_estimators=300,
        max_depth=30,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    ))
])

# Train
pipeline.fit(X_train, y_train_encoded)

# Validate
y_val_pred = pipeline.predict(X_val)
val_acc = accuracy_score(y_val_encoded, y_val_pred)
val_f1 = f1_score(y_val_encoded, y_val_pred, average='weighted')

print(f"\nValidation Accuracy: {val_acc:.4f}")
print(f"Validation F1-Score: {val_f1:.4f}")

# Test
y_test_pred = pipeline.predict(X_test)
test_acc = accuracy_score(y_test_encoded, y_test_pred)
test_f1 = f1_score(y_test_encoded, y_test_pred, average='weighted')

print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test F1-Score: {test_f1:.4f}")

# Decode predictions for report
y_test_pred_decoded = label_encoder.inverse_transform(y_test_pred)

print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
print(classification_report(y_test, y_test_pred_decoded))

# Save model
model_dir = Path("models/recommendation_engine")
model_dir.mkdir(parents=True, exist_ok=True)

model_path = model_dir / "best_recommendation_model.pkl"
with open(model_path, 'wb') as f:
    pickle.dump(pipeline, f)

with open(model_dir / "label_encoder.pkl", 'wb') as f:
    pickle.dump(label_encoder, f)

print(f"\n[OK] Saved model to {model_path}")

# Save metadata
metadata = {
    'model_name': 'Random Forest',
    'num_actions': len(label_encoder.classes_),
    'actions': list(label_encoder.classes_),
    'test_accuracy': float(test_acc),
    'test_f1_score': float(test_f1),
    'val_accuracy': float(val_acc),
    'val_f1_score': float(val_f1),
    'trained_date': datetime.now().isoformat()
}

metadata_path = model_dir / "model_metadata.json"
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"[OK] Saved metadata to {metadata_path}")

# Save results
results_df = pd.DataFrame([{
    'model': 'Random Forest',
    'val_accuracy': val_acc,
    'val_f1': val_f1,
    'test_accuracy': test_acc,
    'test_f1': test_f1
}])

results_path = model_dir / "training_results.csv"
results_df.to_csv(results_path, index=False)
print(f"[OK] Saved results to {results_path}")

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"Model: Random Forest")
print(f"Actions: {len(label_encoder.classes_)}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test F1-Score: {test_f1:.4f}")
print("="*60)
