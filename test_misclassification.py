"""
Test specific review to diagnose category misclassification issue
"""

import sys
sys.path.append('src')
import pickle
from utils import TextPreprocessor

print("="*70)
print("DIAGNOSING CATEGORY MISCLASSIFICATION ISSUE")
print("="*70)

# Load models
print("\nLoading models...")
sentiment_model = pickle.load(open('models/sentiment_analyzer/best_sentiment_model.pkl', 'rb'))
category_model = pickle.load(open('models/category_classifier/best_category_model.pkl', 'rb'))
recommendation_model = pickle.load(open('models/recommendation_engine/best_recommendation_model.pkl', 'rb'))
label_encoder = pickle.load(open('models/recommendation_engine/label_encoder.pkl', 'rb'))

preprocessor = TextPreprocessor()

# Problem review
review = "we eneter to the hotel and the welcome us warmly , but there food was not that much of testy , and there dilivery time is so late"

print(f"\nTest Review:")
print(f'"{review}"')
print("\n" + "="*70)

# Clean the review
cleaned = preprocessor.clean_text(review, remove_stopwords=False, lemmatize=True)
print(f"\nCleaned Review:")
print(f'"{cleaned}"')

# Step 1: Sentiment
try:
    sentiment = sentiment_model.predict([cleaned])[0]
    print(f"\nSentiment: {sentiment}")
except Exception as e:
    print(f"Sentiment error: {e}")
    sentiment = "negative"

# Step 2: Category
try:
    # Get prediction with probability
    category = category_model.predict([cleaned])[0]
    proba = category_model.predict_proba([cleaned])[0]
    
    # Get class names
    classes = category_model.classes_
    
    print(f"\nCategory Prediction: {category}")
    print("\nCategory Probabilities:")
    for cls, prob in zip(classes, proba):
        print(f"  {cls}: {prob:.4f} ({prob*100:.2f}%)")
    
    # Get top 2 predictions
    top_indices = proba.argsort()[-2:][::-1]
    print(f"\nTop 2 Predictions:")
    for idx in top_indices:
        print(f"  {classes[idx]}: {proba[idx]*100:.2f}%")
        
except Exception as e:
    print(f"Category error: {e}")
    import traceback
    traceback.print_exc()

# Step 3: Action
try:
    action_encoded = recommendation_model.predict([cleaned])[0]
    action = label_encoder.inverse_transform([action_encoded])[0]
    print(f"\nRecommended Action: {action}")
except Exception as e:
    print(f"Action error: {e}")

print("\n" + "="*70)
print("ANALYSIS")
print("="*70)
print("\nExpected:")
print("  Category: Food")
print("  Reason: 'food was not that much of testy', 'delivery time is so late'")
print("\nKeywords found in review:")
keywords = {
    'Food': ['food', 'testy', 'delivery', 'dilivery'],
    'Services': ['welcome', 'delivery', 'dilivery', 'time'],
    'Rooms': [],
    'Recreation': []
}

for cat, kws in keywords.items():
    found = [kw for kw in kws if kw in review.lower()]
    if found:
        print(f"  {cat}: {found}")

print("\n" + "="*70)
