"""
Utility functions for the Advanced Customer Feedback Analysis System
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from textblob import TextBlob
import numpy as np

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4', quiet=True)

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

from nltk.sentiment.vader import SentimentIntensityAnalyzer

class TextPreprocessor:
    """Advanced text preprocessing for customer feedback analysis"""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        # Keep some negation words as they're important for sentiment
        self.stop_words -= {'not', 'no', 'nor', 'neither', 'never', 'none', 'nobody', 'nothing', 'nowhere'}
        
    def clean_text(self, text: str, remove_stopwords: bool = False, lemmatize: bool = True) -> str:
        """
        Clean and preprocess text
        
        Args:
            text: Input text string
            remove_stopwords: Whether to remove stop words
            lemmatize: Whether to lemmatize words
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Handle "room service" bigram explicitly to prevent confusion between 'Room' and 'Services'
        text = text.replace("room service", "room_service")
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters and digits but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords if requested
        if remove_stopwords:
            tokens = [word for word in tokens if word not in self.stop_words]
        
        # Lemmatize if requested
        if lemmatize:
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        
        # Remove extra whitespace
        cleaned_text = " ".join(tokens)
        cleaned_text = " ".join(cleaned_text.split())
        
        return cleaned_text
    
    def extract_features(self, text: str) -> Dict[str, Any]:
        """
        Extract additional features from text
        
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Basic length features
        features['char_count'] = len(text)
        features['word_count'] = len(text.split())
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        
        # Uppercase ratio
        features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        
        # Punctuation count
        features['punctuation_count'] = sum(1 for c in text if c in string.punctuation)
        
        # Exclamation and question marks
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        
        return features

    def extract_advanced_features(self, text: str) -> np.ndarray:
        """
        Extract advanced features for ML model (matches training logic)
        Returns numpy array of shape (1, n_features)
        """
        if not isinstance(text, str):
            return np.zeros((1, 10))
            
        features = []
        
        # 1. Basic Counts
        features.append(len(text)) # char_count
        features.append(len(text.split())) # word_count
        features.append(np.mean([len(word) for word in text.split()]) if text.split() else 0) # avg_word_length
        
        # 2. Sentiment Score (Simple)
        features.append(calculate_sentiment_score(text)) # sentiment_score
        
        # 3. VADER
        sia = SentimentIntensityAnalyzer()
        vader_scores = sia.polarity_scores(text)
        features.append(vader_scores['compound'])
        features.append(vader_scores['pos'])
        features.append(vader_scores['neg'])
        features.append(vader_scores['neu'])
        
        # 4. TextBlob
        blob = TextBlob(text)
        features.append(blob.sentiment.polarity)
        features.append(blob.sentiment.subjectivity)
        
        return np.array([features])


def load_sentiment_lexicons() -> Dict[str, List[str]]:
    """
    Load sentiment lexicons for feature engineering
    
    Returns:
        Dictionary with positive and negative word lists
    """
    # Common positive and negative words in hotel reviews
    positive_words = [
        'excellent', 'amazing', 'wonderful', 'fantastic', 'great', 'good', 'nice',
        'clean', 'comfortable', 'friendly', 'helpful', 'beautiful', 'lovely',
        'perfect', 'best', 'awesome', 'outstanding', 'superb', 'exceptional'
    ]
    
    negative_words = [
        'terrible', 'awful', 'horrible', 'bad', 'poor', 'worst', 'dirty',
        'rude', 'uncomfortable', 'noisy', 'smelly', 'broken', 'old', 'small',
        'disappointing', 'unclean', 'unfriendly', 'unhelpful', 'disgusting'
    ]
    
    return {
        'positive': positive_words,
        'negative': negative_words
    }


def extract_negative_sentences(text: str) -> str:
    """
    Extract only negative sentences from a review.
    Used to focus the Recommendation Model on the actual complaints.
    """
    if not isinstance(text, str):
        return ""
        
    sentences = nltk.sent_tokenize(text)
    sia = SentimentIntensityAnalyzer()
    
    neg_sentences = []
    for sent in sentences:
        # Check compound score
        score = sia.polarity_scores(sent)
        # If negative or neutral-leaning-negative (often complaints are stating facts "Room was small")
        if score['compound'] < 0.05: 
            neg_sentences.append(sent)
            
    # If we filtered everything out (e.g. subtly negative review), keep original
    if not neg_sentences:
        return text
        
    return " ".join(neg_sentences)


def augment_text(text: str, n_aug: int = 2) -> List[str]:
    """
    Augment text using synonym replacement to increase dataset size
    """
    from nltk.corpus import wordnet
    import random
    
    words = text.split()
    augmented_texts = []
    
    for _ in range(n_aug):
        new_words = words.copy()
        # Replace up to 30% of words
        n_replace = max(1, int(len(words) * 0.3))
        
        for _ in range(n_replace):
            idx = random.randint(0, len(new_words) - 1)
            word = new_words[idx]
            
            # Find synonyms
            synonyms = []
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    if lemma.name() != word:
                        synonyms.append(lemma.name().replace('_', ' '))
            
            if synonyms:
                new_words[idx] = random.choice(synonyms)
        
        augmented_texts.append(" ".join(new_words))
        
    return augmented_texts


def get_category_keywords() -> Dict[str, List[str]]:
    """
    Get keywords for each category for data filtering
    
    Returns:
        Dictionary of category keywords
    """
    return {
        'Food': ['food', 'breakfast', 'dinner', 'lunch', 'restaurant', 'meal', 'eat', 'drink', 'coffee', 'buffet', 'taste', 'delicious', 'menu', 'kitchen', 'chef', 'waiter', 'server', 'dining'],
        
        'Rooms': ['room', 'bed', 'bathroom', 'shower', 'clean', 'dirty', 'towel', 'pillows', 'ac', 'conditioning', 'air', 'toilet', 'sleep', 'noise', 'housekeeping', 'linen', 'sheet'],
        
        'Services': ['staff', 'service', 'room_service', 'reception', 'check', 'friendly', 'rude', 'helpful', 'manager', 'welcome', 'booking', 'reservation', 'desk', 'concierge'],
        
        'Recreation': ['pool', 'gym', 'beach', 'spa', 'activities', 'entertainment', 'music', 'view', 'location', 'sea', 'ocean', 'swim', 'fitness', 'massage']
    }


def calculate_sentiment_score(text: str) -> float:
    """
    Calculate a simple sentiment score based on positive/negative word counts
    
    Args:
        text: Input text
        
    Returns:
        Sentiment score (positive - negative word count)
    """
    lexicons = load_sentiment_lexicons()
    text_lower = text.lower()
    
    pos_count = sum(1 for word in lexicons['positive'] if word in text_lower)
    neg_count = sum(1 for word in lexicons['negative'] if word in text_lower)
    
    return pos_count - neg_count


def balance_dataset(df: pd.DataFrame, target_col: str, strategy: str = 'undersample') -> pd.DataFrame:
    """
    Balance dataset for imbalanced classes
    
    Args:
        df: Input dataframe
        target_col: Target column name
        strategy: 'undersample', 'oversample', or 'smote'
        
    Returns:
        Balanced dataframe
    """
    from sklearn.utils import resample
    
    # Get class counts
    class_counts = df[target_col].value_counts()
    
    if strategy == 'undersample':
        # Undersample majority classes to match minority
        min_count = class_counts.min()
        
        balanced_dfs = []
        for class_label in class_counts.index:
            class_df = df[df[target_col] == class_label]
            if len(class_df) > min_count:
                class_df = resample(class_df, n_samples=min_count, random_state=42)
            balanced_dfs.append(class_df)
        
        return pd.concat(balanced_dfs, ignore_index=True).sample(frac=1, random_state=42)
    
    elif strategy == 'oversample':
        # Oversample minority classes to match majority
        max_count = class_counts.max()
        
        balanced_dfs = []
        for class_label in class_counts.index:
            class_df = df[df[target_col] == class_label]
            if len(class_df) < max_count:
                class_df = resample(class_df, n_samples=max_count, replace=True, random_state=42)
            balanced_dfs.append(class_df)
        
        return pd.concat(balanced_dfs, ignore_index=True).sample(frac=1, random_state=42)
    
    return df


def print_model_metrics(y_true, y_pred, model_name: str = "Model"):
    """
    Print comprehensive model evaluation metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model for display
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import classification_report, confusion_matrix
    
    print(f"\n{'='*60}")
    print(f"{model_name} Performance Metrics")
    print(f"{'='*60}")
    
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"F1-Score:  {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    
    print(f"\n{'-'*60}")
    print("Classification Report:")
    print(f"{'-'*60}")
    print(classification_report(y_true, y_pred, zero_division=0))
    
    print(f"\n{'-'*60}")
    print("Confusion Matrix:")
    print(f"{'-'*60}")
    print(confusion_matrix(y_true, y_pred))
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = TextPreprocessor()
    
    test_text = "The room was TERRIBLE!!! The staff were very rude and unhelpful. Would NOT recommend."
    cleaned = preprocessor.clean_text(test_text)
    features = preprocessor.extract_features(test_text)
    score = calculate_sentiment_score(test_text)
    
    print("Original:", test_text)
    print("Cleaned:", cleaned)
    print("Features:", features)
    print("Sentiment Score:", score)
