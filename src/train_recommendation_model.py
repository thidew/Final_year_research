"""
Advanced Recommendation Engine Training
Intelligent action mapping for negative feedback
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
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import StackingClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

import sys
sys.path.append(str(Path(__file__).parent))
from utils import print_model_metrics, TextPreprocessor
from sklearn.model_selection import GridSearchCV
from plotting_utils import (
    plot_confusion_matrix_custom,
    plot_multiclass_roc,
    plot_precision_recall_curves,
    plot_learning_curves,
    plot_class_prediction_error
)


class RecommendationModelTrainer:
    """Train recommendation generation models"""
    
    def __init__(self, data_dir: str = "data/processed"):
        self.data_dir = Path(data_dir)
        self.models = {}
        self.results = []
        
    def load_data(self):
        """Load preprocessed data"""
        print("Loading preprocessed recommendation data...")
        
        train = pd.read_csv(self.data_dir / "recommendation_train.csv")
        val = pd.read_csv(self.data_dir / "recommendation_val.csv")
        test = pd.read_csv(self.data_dir / "recommendation_test.csv")
        
        # Determine number of classes
        n_classes = train['action'].nunique()
        print(f"Detected {n_classes} unique actions.")
        
        print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        print("\nAction distribution in training:")
        print(train['action'].value_counts())
        
        return train, val, test
    
    def train_models(self, train_df, val_df):
        """Train recommendation models with DEEP hyperparameter tuning using RandomizedSearchCV"""
        print("\n" + "="*60)
        print("DEEP HYPERPARAMETER TUNING - RANDOMIZED SEARCH")
        print("="*60)
        
        # Prepare features
        X_train = train_df['cleaned_text']
        y_train = train_df['action']
        X_val = val_df['cleaned_text']
        y_val = val_df['action']
        
        from sklearn.model_selection import RandomizedSearchCV
        import xgboost as xgb
        import lightgbm as lgb
        import numpy as np
        
        # Encode labels for XGBoost/LightGBM compatibility
        self.label_encoder = LabelEncoder()
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_val_encoded = self.label_encoder.transform(y_val)
        
        # Simplified parameter space for RandomizedSearchCV (using lists instead of scipy distributions)
        param_distributions = {
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42, class_weight='balanced'),
                'params': {
                    'clf__n_estimators': [100, 200, 300, 500, 700, 1000],
                    'clf__max_depth': [None, 10, 20, 30, 50, 100],
                    'clf__min_samples_split': [2, 5, 10, 15],
                    'clf__min_samples_leaf': [1, 2, 4, 8],
                    'clf__max_features': ['sqrt', 'log2', None],
                    'clf__bootstrap': [True, False],
                    'tfidf__max_features': [1000, 2000, 3000, 5000, 7000],
                    'tfidf__ngram_range': [(1,1), (1,2), (1,3), (2,3)],
                    'tfidf__min_df': [1, 2, 3],
                    'tfidf__max_df': [0.7, 0.8, 0.9, 0.95],
                    'tfidf__sublinear_tf': [True, False],
                    'smote__k_neighbors': [3, 5, 7, 9]
                },
                'n_iter': 20
            },
            'XGBoost': {
                'model': xgb.XGBClassifier(random_state=42, eval_metric='mlogloss'),
                'params': {
                    'clf__n_estimators': [100, 200, 300, 500, 700],
                    'clf__max_depth': [3, 5, 7, 10, 15],
                    'clf__learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.3],
                    'clf__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                    'clf__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                    'clf__min_child_weight': [1, 3, 5, 7],
                    'clf__gamma': [0, 0.1, 0.2, 0.3, 0.5],
                    'clf__reg_alpha': [0, 0.1, 0.5, 1.0],
                    'clf__reg_lambda': [0, 0.1, 0.5, 1.0],
                    'tfidf__max_features': [1000, 2000, 3000, 5000],
                    'tfidf__ngram_range': [(1,1), (1,2), (1,3)],
                    'smote__k_neighbors': [3, 5, 7, 9]
                },
                'n_iter': 20
            },
            'LightGBM': {
                'model': lgb.LGBMClassifier(random_state=42, verbose=-1),
                'params': {
                    'clf__n_estimators': [100, 200, 300, 500],
                    'clf__max_depth': [3, 5, 7, 10, 15],
                    'clf__learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                    'clf__num_leaves': [20, 31, 50, 100, 150],
                    'clf__min_child_samples': [5, 10, 20, 30, 50],
                    'clf__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                    'clf__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                    'clf__reg_alpha': [0, 0.1, 0.5, 1.0],
                    'clf__reg_lambda': [0, 0.1, 0.5, 1.0],
                    'tfidf__max_features': [1000, 2000, 3000, 5000],
                    'tfidf__ngram_range': [(1,1), (1,2), (1,3)],
                    'smote__k_neighbors': [3, 5, 7, 9]
                },
                'n_iter': 20
            }
        }
        
        best_overall_score = 0
        best_overall_model = None
        best_overall_name = None
        
        for name, config in param_distributions.items():
            print(f"\n{'-'*60}")
            print(f"DEEP TUNING: {name} ({config['n_iter']} iterations)")
            print(f"{'-'*60}")
            
            # ImbPipeline with SMOTE
            pipeline = ImbPipeline([
                ('tfidf', TfidfVectorizer(stop_words='english')),
                ('smote', SMOTE(random_state=42)),
                ('clf', config['model'])
            ])
            
            # RandomizedSearchCV - explores MUCH larger space
            rs = RandomizedSearchCV(
                pipeline, 
                config['params'], 
                n_iter=config['n_iter'],
                cv=3, 
                scoring='f1_weighted', 
                n_jobs=-1, 
                verbose=2,
                random_state=42
            )
            
            rs.fit(X_train, y_train_encoded)
            
            print(f"\nBest Params Found:")
            for param, value in rs.best_params_.items():
                print(f"  {param}: {value}")
            
            best_model = rs.best_estimator_
            
            y_val_pred = best_model.predict(X_val)
            val_acc = accuracy_score(y_val_encoded, y_val_pred)
            val_f1 = f1_score(y_val_encoded, y_val_pred, average='weighted')
            
            print(f"\n✓ Validation Accuracy: {val_acc:.4f}")
            print(f"✓ Validation F1-Score: {val_f1:.4f}")
            
            self.models[name] = best_model
            self.results.append({
                'model': name,
                'val_accuracy': val_acc,
                'val_f1': val_f1
            })
            
            # Track best overall
            if val_f1 > best_overall_score:
                best_overall_score = val_f1
                best_overall_model = best_model
                best_overall_name = name
        
        # --- ADVANCED STACKING with best models ---
        print(f"\n{'-'*60}")
        print(f"Training Meta-Stacking Ensemble...")
        print(f"{'-'*60}")
        
        # Use top 3 models
        sorted_results = sorted(self.results, key=lambda x: x['val_f1'], reverse=True)
        top_models = [(r['model'], self.models[r['model']]) for r in sorted_results[:3]]
        
        print(f"Ensemble using: {[m[0] for m in top_models]}")
        
        stacking_clf = StackingClassifier(
            estimators=top_models,
            final_estimator=xgb.XGBClassifier(n_estimators=100, random_state=42),
            cv=5
        )
        
        stacking_clf.fit(X_train, y_train_encoded)
        
        y_val_pred = stacking_clf.predict(X_val)
        val_acc = accuracy_score(y_val_encoded, y_val_pred)
        val_f1 = f1_score(y_val_encoded, y_val_pred, average='weighted')
        
        print(f"\n✓ Meta-Ensemble Accuracy: {val_acc:.4f}")
        print(f"✓ Meta-Ensemble F1-Score: {val_f1:.4f}")
        
        self.models['Meta-Stacking Ensemble'] = stacking_clf
        self.results.append({
            'model': 'Meta-Stacking Ensemble',
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
        y_test = test_df['action']
        y_test_encoded = self.label_encoder.transform(y_test)
        
        y_pred_encoded = best_model.predict(X_test)
        
        # Decode for readable report
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        
        print_model_metrics(y_test, y_pred, f"{best_model_name} (Test Set)")
        
        test_acc = accuracy_score(y_test_encoded, y_pred_encoded)
        test_f1 = f1_score(y_test_encoded, y_pred_encoded, average='weighted')
        
        # --- GENERATE PLOTS ---
        plot_dir = Path("reports/figures/recommendation")
        plot_dir.mkdir(parents=True, exist_ok=True)
        classes = sorted(y_test.unique())
        
        try:
            plot_confusion_matrix_custom(y_test, y_pred, classes, "Recommendation Model", plot_dir)
        except Exception as e:
            print(f"Could not plot confusion matrix: {e}")
            
        try:
            plot_multiclass_roc(best_model, X_test, y_test_encoded, classes, "Recommendation Model", plot_dir)
            plot_precision_recall_curves(best_model, X_test, y_test_encoded, classes, "Recommendation Model", plot_dir)
        except Exception as e:
            print(f"Could not plot probability curves: {e}")
            
        try:
            plot_class_prediction_error(y_test, y_pred, classes, "Recommendation Model", plot_dir)
        except Exception as e:
             print(f"Could not plot prediction error: {e}")
        
        return best_model_name, best_model, test_acc, test_f1
    
    def save_model(self, model_name, model, test_acc, test_f1):
        """Save model"""
        model_dir = Path("models/recommendation_engine")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / "best_recommendation_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
            
        with open(model_dir / "label_encoder.pkl", 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
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
    print("ADVANCED RECOMMENDATION ENGINE TRAINING")
    print("="*60)
    
    trainer = RecommendationModelTrainer()
    
    train_df, val_df, test_df = trainer.load_data()
    
    # Train models
    trainer.train_models(train_df, val_df)
    
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
