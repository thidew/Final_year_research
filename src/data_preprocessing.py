"""
Advanced Data Preprocessing Pipeline for Customer Feedback Analysis
Prepares data for maximum model accuracy with GPU-accelerated deep learning
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from sklearn.model_selection import train_test_split
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Add src to path
sys.path.append(str(Path(__file__).parent))
from utils import TextPreprocessor, balance_dataset, calculate_sentiment_score, get_category_keywords, augment_text, extract_negative_sentences

class DataPreprocessor:
    """Comprehensive data preprocessing for all three models"""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.preprocessor = TextPreprocessor()
        self.vader = SentimentIntensityAnalyzer()
        
    def load_datasets(self):
        """Load all available datasets"""
        print("Loading datasets...")
        
        datasets = {}
        
        # Balanced dataset for sentiment
        balanced_path = self.data_dir / "balanced_hotel_reviews.csv"
        if balanced_path.exists():
            datasets['balanced'] = pd.read_csv(balanced_path)
            print(f"✓ Loaded balanced dataset: {len(datasets['balanced'])} rows")
        
        # Negative dataset for categories and actions
        negative_path = self.data_dir / "negative data set.csv"
        if negative_path.exists():
            datasets['negative'] = pd.read_csv(negative_path)
            print(f"✓ Loaded negative dataset: {len(datasets['negative'])} rows")
        
        # Cleaned dataset
        cleaned_path = self.data_dir / "cleaned_hotel_reviews.csv"
        if cleaned_path.exists():
            datasets['cleaned'] = pd.read_csv(cleaned_path)
            print(f"✓ Loaded cleaned dataset: {len(datasets['cleaned'])} rows")

        # NEW: Synthetic Action Dataset
        syn_action_path = self.data_dir / "synthetic_recommendation_actions.csv"
        if syn_action_path.exists():
            datasets['synthetic_actions'] = pd.read_csv(syn_action_path)
            print(f"✓ Loaded synthetic action dataset: {len(datasets['synthetic_actions'])} rows")
        
        return datasets
    
    def prepare_sentiment_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for sentiment analysis model
        
        Expected columns: 'text', 'sentiment'
        """
        print("\n" + "="*60)
        print("PREPARING SENTIMENT ANALYSIS DATA")
        print("="*60)
        
        # Ensure required columns exist
        if 'text' not in df.columns or 'sentiment' not in df.columns:
            raise ValueError("Dataset must have 'text' and 'sentiment' columns")
        
        # Remove missing values
        df = df.dropna(subset=['text', 'sentiment'])
        print(f"Rows after removing NaN: {len(df)}")
        
        # Clean text
        print("Cleaning text...")
        df['cleaned_text'] = df['text'].apply(
            lambda x: self.preprocessor.clean_text(x, remove_stopwords=False, lemmatize=True)
        )
        
        # Remove very short reviews (less than 3 words)
        df = df[df['cleaned_text'].str.split().str.len() >= 3]
        print(f"Rows after removing short reviews: {len(df)}")
        
        # Add advanced features
        print("Extracting features...")
        df['char_count'] = df['text'].str.len()
        df['word_count'] = df['text'].str.split().str.len()
        df['avg_word_length'] = df['text'].apply(
            lambda x: np.mean([len(word) for word in x.split()]) if x.split() else 0
        )
        
        # Sentiment features
        print("Adding sentiment lexicon features...")
        df['sentiment_score'] = df['cleaned_text'].apply(calculate_sentiment_score)
        
        # VADER sentiment
        df['vader_compound'] = df['text'].apply(lambda x: self.vader.polarity_scores(x)['compound'])
        df['vader_pos'] = df['text'].apply(lambda x: self.vader.polarity_scores(x)['pos'])
        df['vader_neg'] = df['text'].apply(lambda x: self.vader.polarity_scores(x)['neg'])
        df['vader_neu'] = df['text'].apply(lambda x: self.vader.polarity_scores(x)['neu'])
        
        # TextBlob polarity and subjectivity
        print("Adding TextBlob features...")
        df['textblob_polarity'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
        df['textblob_subjectivity'] = df['text'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
        
        # Class distribution
        print("\nSentiment Distribution:")
        print(df['sentiment'].value_counts())
        print(f"\nClass percentages:")
        print(df['sentiment'].value_counts(normalize=True) * 100)
        
        return df
    
    def prepare_category_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for category classification model
        
        Expected: negative dataset with bad_* columns
        """
        print("\n" + "="*60)
        print("PREPARING CATEGORY CLASSIFICATION DATA")
        print("="*60)
        
        # Map bad_* columns to categories
        category_mapping = {
            'bad_food': 'Food',
            'bad_room': 'Rooms',
            'bad_service': 'Services',
            'bad_recreation': 'Recreation'
        }
        
        data = []
        
        # Extract reviews with category labels
        for idx, row in df.iterrows():
            review = row.get('Review_text', '')
            if not isinstance(review, str) or len(review) < 5:
                continue
            
            # Find which categories this review belongs to
            # FIX: Only assign category if specific keywords are present
            # This implements Weak Supervision to remove label noise
            cat_keywords = get_category_keywords()
            
            for col, cat_name in category_mapping.items():
                if col in df.columns and row[col] == 1:
                    # Weak supervision check
                    keywords = cat_keywords.get(cat_name, [])
                    if any(kw in str(review).lower() for kw in keywords):
                        data.append({
                            'text': review,
                            'category': cat_name
                        })
        
        if not data:
            raise ValueError("No labeled category data found")
        
        category_df = pd.DataFrame(data)
        print(f"Created {len(category_df)} category examples")
        
        # Clean text
        print("Cleaning text...")
        category_df['cleaned_text'] = category_df['text'].apply(
            lambda x: self.preprocessor.clean_text(x, remove_stopwords=False, lemmatize=True)
        )
        
        # Remove short reviews
        category_df = category_df[category_df['cleaned_text'].str.split().str.len() >= 3]
        
        # Add features
        category_df['char_count'] = category_df['text'].str.len()
        category_df['word_count'] = category_df['text'].str.split().str.len()
        
        # Category distribution
        print("\nCategory Distribution:")
        print(category_df['category'].value_counts())
        print(f"\nClass percentages:")
        print(category_df['category'].value_counts(normalize=True) * 100)
        
        # Balance if needed
        if category_df['category'].value_counts().max() / category_df['category'].value_counts().min() > 2:
            print("\nBalancing categories...")
            category_df = balance_dataset(category_df, 'category', strategy='oversample')
            print("After balancing:")
            print(category_df['category'].value_counts())
        
        return category_df
    
    def prepare_recommendation_data(self, df: pd.DataFrame, is_synthetic=False) -> pd.DataFrame:
        """
        Prepare data for action recommendation model.
        Supports both legacy 'negative' dataset AND new 'synthetic' dataset.
        """
        print("\n" + "="*60)
        print("PREPARING RECOMMENDATION DATA")
        print("="*60)
        
        action_df = None
        
        if is_synthetic:
            print("Processing SYNTHETIC ACTION DATA...")
            # Synthetic data already has 'text' and 'action' columns
            action_df = df.copy()
        
        else:
            # Legacy path for "negative data set.csv"
            print("Processing LEGACY negative data (mapping categories to actions)...")
            
            # Map categories to recommended actions
            categories_actions = {
                'bad_food': ('Food', 'Review Menu & Kitchen Quality'),
                'bad_room': ('Rooms', 'Inspect Room & Maintenance'),
                'bad_service': ('Services', 'Staff Training Required'),
                'bad_recreation': ('Recreation', 'Upgrade Pool & Activities')
            }
            
            data = []
            
            for idx, row in df.iterrows():
                review = row.get('Review_text', '')
                
                # CRITICAL FIX: Extract only negative parts of the review
                focused_review = extract_negative_sentences(review)
                
                for col, (cat_name, action) in categories_actions.items():
                    if col in df.columns and row[col] == 1:
                        # Weak supervision check
                        cat_keywords = get_category_keywords()
                        keywords = cat_keywords.get(cat_name, [])
                        if any(kw in str(review).lower() for kw in keywords):
                            data.append({
                                'text': focused_review,
                                'category': cat_name,
                                'action': action
                            })
            
            if data:
                action_df = pd.DataFrame(data)
        
        if action_df is None or len(action_df) == 0:
             raise ValueError("No labeled action data found")

        print(f"Created {len(action_df)} action examples")
        
        # Clean text
        print("Cleaning text...")
        action_df['cleaned_text'] = action_df['text'].apply(
            lambda x: self.preprocessor.clean_text(x, remove_stopwords=False, lemmatize=True)
        )
        
        # Remove short reviews
        action_df = action_df[action_df['cleaned_text'].str.split().str.len() >= 3]
        
        # Add features
        action_df['char_count'] = action_df['text'].str.len()
        action_df['word_count'] = action_df['text'].str.split().str.len()
        action_df['sentiment_score'] = action_df['cleaned_text'].apply(calculate_sentiment_score)
        
        # Distribution
        print("\nAction Distribution:")
        print(action_df['action'].value_counts())
        
        return action_df
    
    def create_train_test_splits(self, df: pd.DataFrame, target_col: str, 
                                 test_size: float = 0.2, val_size: float = 0.1):
        """
        Create train/validation/test splits
        
        Returns:
            Dictionary with train, val, test DataFrames
        """
        print(f"\nCreating train/val/test splits...")
        
        # First split: train+val vs test
        train_val, test = train_test_split(
            df, test_size=test_size, random_state=42, stratify=df[target_col]
        )
        
        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val, test_size=val_ratio, random_state=42, stratify=train_val[target_col]
        )
        
        print(f"Train: {len(train)} ({len(train)/len(df)*100:.1f}%)")
        print(f"Val:   {len(val)} ({len(val)/len(df)*100:.1f}%)")
        print(f"Test:  {len(test)} ({len(test)/len(df)*100:.1f}%)")
        
        return {
            'train': train,
            'val': val,
            'test': test
        }
    
    def augment_training_data(self, train_df: pd.DataFrame, text_col: str) -> pd.DataFrame:
        """Augment training data with synthetic examples"""
        print(f"Augmenting training data (Input: {len(train_df)} rows)...")
        
        augmented_rows = []
        for idx, row in train_df.iterrows():
            # Original
            augmented_rows.append(row.to_dict())
            
            # Augmented (10 copies for massive dataset expansion)
            aug_texts = augment_text(row[text_col], n_aug=10)
            for aug_text in aug_texts:
                new_row = row.to_dict()
                new_row[text_col] = aug_text
                # Also update cleaned text if it exists
                if 'cleaned_text' in new_row:
                    new_row['cleaned_text'] = self.preprocessor.clean_text(aug_text)
                augmented_rows.append(new_row)
                
        aug_df = pd.DataFrame(augmented_rows)
        print(f"Augmentation complete. New size: {len(aug_df)} rows")
        return aug_df
    
    def save_processed_data(self, df: pd.DataFrame, filename: str):
        """Save processed data"""
        output_path = Path("data/processed") / filename
        df.to_csv(output_path, index=False)
        print(f"\n✓ Saved to {output_path}")


def main():
    """Main preprocessing pipeline"""
    preprocessor = DataPreprocessor()
    
    # Load datasets
    datasets = preprocessor.load_datasets()
    
    # 1. Prepare Sentiment Data
    if 'balanced' in datasets:
        sentiment_df = preprocessor.prepare_sentiment_data(datasets['balanced'])
        preprocessor.save_processed_data(sentiment_df, 'sentiment_processed.csv')
        
        # Create splits
        sentiment_splits = preprocessor.create_train_test_splits(sentiment_df, 'sentiment')
        preprocessor.save_processed_data(sentiment_splits['train'], 'sentiment_train.csv')
        preprocessor.save_processed_data(sentiment_splits['val'], 'sentiment_val.csv')
        preprocessor.save_processed_data(sentiment_splits['test'], 'sentiment_test.csv')
    
    # 2. Prepare Category Data
    if 'negative' in datasets:
        category_df = preprocessor.prepare_category_data(datasets['negative'])
        preprocessor.save_processed_data(category_df, 'category_processed.csv')
        
        # Create splits
        category_splits = preprocessor.create_train_test_splits(category_df, 'category')
        
        # Augment only training data
        category_train_aug = preprocessor.augment_training_data(category_splits['train'], 'text')
        
        preprocessor.save_processed_data(category_train_aug, 'category_train.csv')
        preprocessor.save_processed_data(category_splits['val'], 'category_val.csv')
        preprocessor.save_processed_data(category_splits['test'], 'category_test.csv')
    
    # 3. Prepare Recommendation Data
    # PREFER SYNTHETIC DATA IF AVAILABLE
    if 'synthetic_actions' in datasets:
        print("\n[INFO] Using SYNTHETIC ACTION DATA for Recommendation Model")
        action_df = preprocessor.prepare_recommendation_data(datasets['synthetic_actions'], is_synthetic=True)
        # Proceed with saving
        preprocessor.save_processed_data(action_df, 'recommendation_processed.csv')
        
        # Create splits
        action_splits = preprocessor.create_train_test_splits(action_df, 'action')
        
        # Note: We might not need huge augmentation if we already generated 2500 samples
        # But let's do mild augmentation
        action_train_aug = preprocessor.augment_training_data(action_splits['train'], 'text')
        
        preprocessor.save_processed_data(action_train_aug, 'recommendation_train.csv')
        preprocessor.save_processed_data(action_splits['val'], 'recommendation_val.csv')
        preprocessor.save_processed_data(action_splits['test'], 'recommendation_test.csv')
        
    elif 'negative' in datasets:
        print("\n[INFO] Using LEGACY NEGATIVE DATA for Recommendation Model")
        action_df = preprocessor.prepare_recommendation_data(datasets['negative'])
        preprocessor.save_processed_data(action_df, 'recommendation_processed.csv')
        
        # Create splits
        action_splits = preprocessor.create_train_test_splits(action_df, 'action')
        
        # Augment only training data
        action_train_aug = preprocessor.augment_training_data(action_splits['train'], 'text')
        
        preprocessor.save_processed_data(action_train_aug, 'recommendation_train.csv')
        preprocessor.save_processed_data(action_splits['val'], 'recommendation_val.csv')
        preprocessor.save_processed_data(action_splits['test'], 'recommendation_test.csv')
    
    print("\n" + "="*60)
    print("DATA PREPROCESSING COMPLETE!")
    print("="*60)
    print("\nProcessed files saved to data/processed/")


if __name__ == "__main__":
    main()
