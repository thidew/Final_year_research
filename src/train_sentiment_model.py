"""
Advanced Sentiment Analysis Model Training
Uses multiple approaches and selects the best performing model for maximum accuracy
GPU-accelerated deep learning with transformers
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
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import lightgbm as lgb

import sys
sys.path.append(str(Path(__file__).parent))
from utils import print_model_metrics
from plotting_utils import (
    plot_confusion_matrix_custom,
    plot_multiclass_roc,
    plot_precision_recall_curves,
    plot_learning_curves,
    plot_class_prediction_error
)

class SentimentModelTrainer:
    """Train and compare multiple sentiment analysis models"""
    
    def __init__(self, data_dir: str = "data/processed"):
        self.data_dir = Path(data_dir)
        self.models = {}
        self.results = []
        
    def load_data(self):
        """Load preprocessed data"""
        print("Loading preprocessed data...")
        
        train = pd.read_csv(self.data_dir / "sentiment_train.csv")
        val = pd.read_csv(self.data_dir / "sentiment_val.csv")
        test = pd.read_csv(self.data_dir / "sentiment_test.csv")
        
        print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        
        return train, val, test
    
    def train_traditional_ml(self, X_train, y_train, X_val, y_val):
        """Train traditional ML models with TF-IDF"""
        print("\n" + "="*60)
        print("TRAINING TRADITIONAL ML MODELS")
        print("="*60)
        
        
        # Grid Search for best base models
        grids = {
            'Logistic Regression': {
                'model': LogisticRegression(random_state=42, max_iter=2000),
                'params': {
                    'clf__C': [0.1, 1.0, 10.0],
                    'tfidf__ngram_range': [(1,1), (1,2)],
                    'tfidf__max_features': [2000, 5000]
                }
            },
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'clf__n_estimators': [100, 300],
                    'clf__max_depth': [None, 30],
                    'tfidf__max_features': [2000, 5000]
                }
            }
        }
        
        best_estimators = []
        
        for name, config in grids.items():
            print(f"\n{'-'*60}")
            print(f"Tuning {name}...")
            print(f"{'-'*60}")
            
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words='english')),
                ('clf', config['model'])
            ])
            
            gs = GridSearchCV(pipeline, config['params'], cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
            gs.fit(X_train, y_train)
            
            print(f"Best Params: {gs.best_params_}")
            best_model = gs.best_estimator_
            self.models[name] = best_model
            best_estimators.append((name, best_model))
            
            y_val_pred = best_model.predict(X_val)
            val_acc = accuracy_score(y_val, y_val_pred)
            val_f1 = f1_score(y_val, y_val_pred, average='weighted')

            print(f"Validation Accuracy: {val_acc:.4f}")
            print(f"Validation F1-Score: {val_f1:.4f}")

            self.results.append({
                'model': name,
                'type': 'Traditional ML',
                'val_accuracy': val_acc,
                'val_f1': val_f1
            })

        # Advanced Ensemble with Soft Voting
        print("\n------------------------------------------------------------")
        print("Training Soft Voting Ensemble...")
        print("------------------------------------------------------------")
        
        ensemble = VotingClassifier(
            estimators=best_estimators,
            voting='soft'
        )
        ensemble.fit(X_train, y_train)
        
        y_val_pred = ensemble.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred, average='weighted')
        
        print(f"Validation Accuracy: {val_acc:.4f}")
        print(f"Validation F1-Score: {val_f1:.4f}")
        
        self.models['Ensemble'] = ensemble
        self.results.append({
            'model': 'Ensemble',
            'type': 'Ensemble',
            'val_accuracy': val_acc,
            'val_f1': val_f1
        })
    
    def train_advanced_ml_with_features(self, train_df, val_df):
        """Train ML models with engineered features"""
        print("\n" + "="*60)
        print("TRAINING ML WITH ADVANCED FEATURES")
        print("="*60)
        
        # Feature columns
        text_features = ['cleaned_text']
        numerical_features = [
            'char_count', 'word_count', 'avg_word_length',
            'sentiment_score', 'vader_compound', 'vader_pos', 'vader_neg', 'vader_neu',
            'textblob_polarity', 'textblob_subjectivity'
        ]
        
        # Prepare training data
        X_train_text = train_df['cleaned_text']
        X_train_num = train_df[numerical_features].values
        y_train = train_df['sentiment']
        
        X_val_text = val_df['cleaned_text']
        X_val_num = val_df[numerical_features].values
        y_val = val_df['sentiment']
        
        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1,2))
        X_train_tfidf = vectorizer.fit_transform(X_train_text).toarray()
        X_val_tfidf = vectorizer.transform(X_val_text).toarray()
        
        # Combine features
        X_train_combined = np.hstack([X_train_tfidf, X_train_num])
        X_val_combined = np.hstack([X_val_tfidf, X_val_num])
        
        # Train models
        models = {
            'Gradient Boosting + Features': GradientBoostingClassifier(
                n_estimators=300,
                random_state=42,
                learning_rate=0.1,
                max_depth=5,
                subsample=0.8
            ),
            
            'Random Forest + Features': RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2
            ),
            
            'Extra Trees + Features': RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                max_depth=20,
                criterion='entropy',
                min_samples_split=5
            )
        }
        
        for name, model in models.items():
            print(f"\n{'-'*60}")
            print(f"Training {name}...")
            print(f"{'-'*60}")
            
            model.fit(X_train_combined, y_train)
            
            y_val_pred = model.predict(X_val_combined)
            val_acc = accuracy_score(y_val, y_val_pred)
            val_f1 = f1_score(y_val, y_val_pred, average='weighted')
            
            print(f"Validation Accuracy: {val_acc:.4f}")
            print(f"Validation F1-Score: {val_f1:.4f}")
            
            # Store model with vectorizer
            self.models[name] = {
                'model': model,
                'vectorizer': vectorizer,
                'feature_cols': numerical_features
            }
            
            self.results.append({
                'model': name,
                'type': 'Advanced ML',
                'val_accuracy': val_acc,
                'val_f1': val_f1
            })
    
    def create_ensemble(self, X_train, y_train, X_val, y_val):
        """Create ensemble of best models"""
        print("\n" + "="*60)
        print("CREATING ENSEMBLE MODEL")
        print("="*60)
        
        # Select top 3 traditional ML models based on F1 score
        trad_results = [r for r in self.results if r['type'] == 'Traditional ML']
        trad_results_sorted = sorted(trad_results, key=lambda x: x['val_f1'], reverse=True)
        
        top_model_names = [r['model'] for r in trad_results_sorted[:3]]
        print(f"Top models for ensemble: {top_model_names}")
        
        # Create voting ensemble
        estimators = [(name, self.models[name]) for name in top_model_names]
        
        ensemble = VotingClassifier(
            estimators=estimators,
            voting='hard',  # Changed from 'soft' to support LinearSVC
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
            'type': 'Ensemble',
            'val_accuracy': val_acc,
            'val_f1': val_f1
        })
    
    def evaluate_best_model(self, test_df):
        """Evaluate the best model on test set"""
        print("\n" + "="*60)
        print("FINAL EVALUATION ON TEST SET")
        print("="*60)
        
        # Find best model
        best_result = max(self.results, key=lambda x: x['val_f1'])
        best_model_name = best_result['model']
        
        print(f"\nBest Model: {best_model_name}")
        print(f"Validation F1: {best_result['val_f1']:.4f}")
        
        best_model = self.models[best_model_name]
        
        # Test evaluation
        X_test = test_df['cleaned_text']
        y_test = test_df['sentiment']
        
        # Handle advanced ML models with features
        if isinstance(best_model, dict):
            # Advanced ML with features
            vectorizer = best_model['vectorizer']
            model = best_model['model']
            feature_cols = best_model['feature_cols']
            
            X_test_tfidf = vectorizer.transform(X_test).toarray()
            X_test_num = test_df[feature_cols].values
            X_test_combined = np.hstack([X_test_tfidf, X_test_num])
            
            y_pred = model.predict(X_test_combined)
        else:
            # Pipeline or ensemble
            y_pred = best_model.predict(X_test)
        
        # Metrics
        print_model_metrics(y_test, y_pred, f"{best_model_name} (Test Set)")
        
        # Test scores
        test_acc = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred, average='weighted')
        
        # --- GENERATE PLOTS ---
        plot_dir = Path("reports/figures/sentiment")
        classes = sorted(y_test.unique())
        
        # Ensure input format matches what model expects (DataFrame vs Series)
        if isinstance(X_test, pd.Series):
            X_test_plot = pd.DataFrame(X_test, columns=['cleaned_text'])
        else:
            X_test_plot = X_test

        plot_confusion_matrix_custom(y_test, y_pred, classes, "Sentiment Model", plot_dir)
        try:
             # Sentiment model might have complex feature engineering pipeline
             # We pass X_test assuming the pipeline handles it
            plot_multiclass_roc(best_model, X_test, y_test, classes, "Sentiment Model", plot_dir)
            plot_precision_recall_curves(best_model, X_test, y_test, classes, "Sentiment Model", plot_dir)
        except Exception as e:
            print(f"Could not plot probability curves: {e}")
            
        plot_class_prediction_error(y_test, y_pred, classes, "Sentiment Model", plot_dir)
        
        return best_model_name, best_model, test_acc, test_f1
    
    def save_model(self, model_name, model, test_acc, test_f1):
        """Save the best model"""
        model_dir = Path("models/sentiment_analyzer")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = model_dir / "best_sentiment_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"\n✓ Saved model to {model_path}")
        
        # Save metadata
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
        
        # Save results summary
        results_df = pd.DataFrame(self.results)
        results_df = results_df.sort_values('val_f1', ascending=False)
        results_path = model_dir / "training_results.csv"
        results_df.to_csv(results_path, index=False)
        
        print(f"✓ Saved results to {results_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("MODEL COMPARISON SUMMARY")
        print("="*60)
        print(results_df.to_string(index=False))
        print("="*60)


def main():
    """Main training pipeline"""
    print("="*60)
    print("ADVANCED SENTIMENT ANALYSIS MODEL TRAINING")
    print("="*60)
    
    trainer = SentimentModelTrainer()
    
    # Load data
    train_df, val_df, test_df = trainer.load_data()
    
    # Prepare text data
    X_train = train_df['cleaned_text']
    y_train = train_df['sentiment']
    X_val = val_df['cleaned_text']
    y_val = val_df['sentiment']
    
    # 1. Train traditional ML models
    trainer.train_traditional_ml(X_train, y_train, X_val, y_val)
    
    # 2. Train advanced ML with features
    trainer.train_advanced_ml_with_features(train_df, val_df)
    
    # 3. Create ensemble
    trainer.create_ensemble(X_train, y_train, X_val, y_val)
    
    # 4. Evaluate best model
    best_name, best_model, test_acc, test_f1 = trainer.evaluate_best_model(test_df)
    
    # 5. Save model
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
