"""
Advanced Category Classification Model Training
Maximum accuracy using ensemble methods
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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

import sys
sys.path.append(str(Path(__file__).parent))
from utils import print_model_metrics
from sklearn.model_selection import GridSearchCV
from plotting_utils import (
    plot_confusion_matrix_custom,
    plot_multiclass_roc,
    plot_precision_recall_curves,
    plot_learning_curves,
    plot_class_prediction_error
)


class CategoryModelTrainer:
    """Train category classification models"""
    
    def __init__(self, data_dir: str = "data/processed"):
        self.data_dir = Path(data_dir)
        self.models = {}
        self.results = []
        
    def load_data(self):
        """Load preprocessed data"""
        print("Loading preprocessed category data...")
        
        train = pd.read_csv(self.data_dir / "synthetic_category_train.csv")
        val = pd.read_csv(self.data_dir / "synthetic_category_val.csv")
        test = pd.read_csv(self.data_dir / "synthetic_category_test.csv")
        
        print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        print("\nCategory distribution in training:")
        print(train['category'].value_counts())
        
        return train, val, test
    
    def train_models(self, X_train, y_train, X_val, y_val):
        """Train multiple classification models"""
        print("\n" + "="*60)
        print("TRAINING CATEGORY CLASSIFICATION MODELS")
        print("="*60)
        
        
        # Define grid search parameters
        grids = {
            'Logistic Regression': {
                'model': LogisticRegression(random_state=42, max_iter=2000),
                'params': {
                    'clf__C': [0.1, 1.0, 5.0, 10.0],
                    'clf__solver': ['liblinear', 'lbfgs'],
                    'tfidf__ngram_range': [(1,1), (1,2)],
                    'tfidf__max_features': [2000, 5000]
                }
            },
            'Linear SVM': {
                'model': LinearSVC(random_state=42, dual='auto'),
                'params': {
                    'clf__C': [0.1, 1.0, 5.0, 10.0],
                    'clf__loss': ['hinge', 'squared_hinge'],
                    'tfidf__ngram_range': [(1,1), (1,2)]
                }
            },
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'clf__n_estimators': [100, 300],
                    'clf__max_depth': [None, 20, 50],
                    'tfidf__max_features': [2000, 5000]
                }
            }
        }
        
        for name, config in grids.items():
            print(f"\n{'-'*60}")
            print(f"Tuning {name}...")
            print(f"{'-'*60}")
            
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words='english')),
                ('clf', config['model'])
            ])
            
            # Grid Search with Cross Validation
            gs = GridSearchCV(pipeline, config['params'], cv=3, scoring='f1_weighted', n_jobs=-1, verbose=1)
            gs.fit(X_train, y_train)
            
            print(f"Best Params: {gs.best_params_}")
            best_model = gs.best_estimator_
            
            y_val_pred = best_model.predict(X_val)
            val_acc = accuracy_score(y_val, y_val_pred)
            val_f1 = f1_score(y_val, y_val_pred, average='weighted')
            
            print(f"Validation Accuracy: {val_acc:.4f}")
            print(f"Validation F1-Score: {val_f1:.4f}")
            
            self.models[name] = best_model
            self.results.append({
                'model': name,
                'val_accuracy': val_acc,
                'val_f1': val_f1
            })
    
    def create_ensemble(self, X_train, y_train, X_val, y_val):
        """Create voting ensemble"""
        print("\n" + "="*60)
        print("CREATING ENSEMBLE MODEL")
        print("="*60)
        
        # Top 3 models
        results_sorted = sorted(self.results, key=lambda x: x['val_f1'], reverse=True)
        top_model_names = [r['model'] for r in results_sorted[:3]]
        
        print(f"Top models for ensemble: {top_model_names}")
        
        estimators = [(name, self.models[name]) for name in top_model_names]
        
        ensemble = VotingClassifier(
            estimators=estimators,
            voting='hard',
            weights=[3, 2, 1]
        )
        
        print("\nTraining ensemble...")
        ensemble.fit(X_train, y_train)
        
        y_val_pred = ensemble.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred, average='weighted')
        
        print(f"Ensemble Validation Accuracy: {val_acc:.4f}")
        print(f"Ensemble Validation F1-Score: {val_f1:.4f}")
        
        self.models['Ensemble'] = ensemble
        self.results.append({
            'model': 'Ensemble',
            'val_accuracy': val_acc,
            'val_f1': val_f1
        })
    
    def evaluate_best_model(self, test_df):
        """Evaluate best model"""
        print("\n" + "="*60)
        print("FINAL EVALUATION ON TEST SET")
        print("="*60)
        
        best_result = max(self.results, key=lambda x: x['val_f1'])
        best_model_name = best_result['model']
        
        print(f"\nBest Model: {best_model_name}")
        print(f"Validation F1: {best_result['val_f1']:.4f}")
        
        best_model = self.models[best_model_name]
        
        X_test = test_df['cleaned_text']
        y_test = test_df['category']
        
        y_pred = best_model.predict(X_test)
        
        print_model_metrics(y_test, y_pred, f"{best_model_name} (Test Set)")
        
        test_acc = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred, average='weighted')
        
        # --- GENERATE PLOTS ---
        plot_dir = Path("reports/figures/category")
        classes = sorted(y_test.unique())
        
        plot_confusion_matrix_custom(y_test, y_pred, classes, "Category Model", plot_dir)
        plot_multiclass_roc(best_model, X_test, y_test, classes, "Category Model", plot_dir)
        plot_precision_recall_curves(best_model, X_test, y_test, classes, "Category Model", plot_dir)
        plot_class_prediction_error(y_test, y_pred, classes, "Category Model", plot_dir)
        
        # Note: Learning curve takes time, running on subset
        # plot_learning_curves(best_model, X_test, y_test, "Category Model", plot_dir)
        
        return best_model_name, best_model, test_acc, test_f1
    
    def save_model(self, model_name, model, test_acc, test_f1):
        """Save model"""
        model_dir = Path("models/category_classifier")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / "best_category_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"\n✓ Saved model to {model_path}")
        
        metadata = {
            'model_name': model_name,
            'test_accuracy': float(test_acc),
            'test_f1_score': float(test_f1),
            'trained_date': datetime.now().isoformat(),
            'all_results': self.results
        }
        
        metadata_path = model_dir / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Saved metadata to {metadata_path}")
        
        results_df = pd.DataFrame(self.results)
        results_df = results_df.sort_values('val_f1', ascending=False)
        results_path = model_dir / "training_results.csv"
        results_df.to_csv(results_path, index=False)
        
        print(f"✓ Saved results to {results_path}")
        
        print("\n" + "="*60)
        print("MODEL COMPARISON SUMMARY")
        print("="*60)
        print(results_df.to_string(index=False))
        print("="*60)


def main():
    """Main training pipeline"""
    print("="*60)
    print("ADVANCED CATEGORY CLASSIFICATION MODEL TRAINING")
    print("="*60)
    
    trainer = CategoryModelTrainer()
    
    train_df, val_df, test_df = trainer.load_data()
    
    X_train = train_df['cleaned_text']
    y_train = train_df['category']
    X_val = val_df['cleaned_text']
    y_val = val_df['category']
    
    # Train models
    trainer.train_models(X_train, y_train, X_val, y_val)
    
    # Create ensemble
    trainer.create_ensemble(X_train, y_train, X_val, y_val)
    
    # Evaluate
    best_name, best_model, test_acc, test_f1 = trainer.evaluate_best_model(test_df)
    
    # Save
    trainer.save_model(best_name, best_model, test_acc, test_f1)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Best Model: {best_name}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1-Score: {test_f1:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
