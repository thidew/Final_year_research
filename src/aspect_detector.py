"""
Aspect-Based Category Detector
Hybrid approach combining keyword matching with sentiment analysis
"""

import re
from typing import Dict, List, Tuple, Optional
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
import nltk

# Ensure required NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)


class AspectBasedCategoryDetector:
    """
    Detects multiple aspects/categories in hotel reviews using keyword matching
    and sentiment analysis. Prioritizes negative aspects for action recommendations.
    """
    
    # Category-specific keywords (including common misspellings)
    ASPECT_KEYWORDS = {
        'Food': {
            'primary': [
                'food', 'meal', 'breakfast', 'lunch', 'dinner', 'brunch',
                'restaurant', 'kitchen', 'cook', 'chef', 'menu', 'dish',
                'cuisine', 'culinary', 'buffet', 'dining', 'cafe', 'cafeteria'
            ],
            'related': [
                'taste', 'tasty', 'testy', 'flavor', 'flavour', 'delicious',
                'disgusting', 'delivery', 'dilivery', 'serve', 'served',
                'serving', 'waiter', 'waitress', 'order', 'ordered',
                'steak', 'pasta', 'soup', 'salad', 'dessert', 'beverage',
                'drink', 'coffee', 'tea', 'wine', 'fresh', 'stale',
                'overcooked', 'undercooked', 'cold', 'hot', 'warm'
            ],
            'negative_indicators': [
                'tasteless', 'bland', 'terrible', 'awful', 'horrible',
                'poisoning', 'spoiled', 'rotten', 'dirty plates', 'hair in'
            ]
        },
        'Rooms': {
            'primary': [
                'room', 'rooms', 'bed', 'bedroom', 'bathroom', 'shower',
                'toilet', 'bath', 'ac', 'air conditioning', 'conditioning',
                'towel', 'towels', 'pillow', 'pillows', 'mattress', 'sheet',
                'sheets', 'blanket', 'furniture', 'closet', 'wardrobe'
            ],
            'related': [
                'clean', 'dirty', 'smell', 'smelly', 'noise', 'noisy',
                'loud', 'comfortable', 'uncomfortable', 'spacious', 'cramped',
                'small', 'tiny', 'view', 'window', 'balcony', 'amenities',
                'amenity', 'tv', 'television', 'wifi', 'internet'
            ],
            'negative_indicators': [
                'moldy', 'mold', 'bug', 'bugs', 'dirt', 'stain', 'stains',
                'cockroach', 'bedbugs', 'pest', 'broken', 'damaged'
            ]
        },
        'Services': {
            'primary': [
                'service', 'services', 'staff', 'employee', 'employees',
                'receptionist', 'concierge', 'check-in', 'checkin', 'checkout',
                'check-out', 'reception', 'front desk', 'desk', 'manager',
                'management', 'bellboy', 'porter', 'housekeeping'
            ],
            'related': [
                'welcome', 'welcomed', 'helpful', 'unhelpful', 'rude',
                'friendly', 'unfriendly', 'professional', 'unprofessional',
                'polite', 'impolite', 'courteous', 'attentive', 'inattentive',
                'response', 'responsive', 'accommodate', 'assistance'
            ],
            'negative_indicators': [
                'ignored', 'rude', 'slow', 'unprofessional', 'incompetent',
                'unhelpful', 'waiting', 'delay', 'late', 'poor service'
            ]
        },
        'Recreation': {
            'primary': [
                'pool', 'pools', 'swimming', 'gym', 'fitness', 'spa',
                'sauna', 'jacuzzi', 'hot tub', 'recreation', 'recreational',
                'activities', 'activity', 'entertainment', 'garden',
                'beach', 'sports', 'tennis', 'golf'
            ],
            'related': [
                'equipment', 'facility', 'facilities', 'amenity', 'amenities',
                'exercise', 'workout', 'relax', 'relaxation', 'leisure',
                'chlorine', 'water', 'clean', 'dirty', 'maintained',
                'maintenance', 'broken', 'working'
            ],
            'negative_indicators': [
                'broken equipment', 'dirty pool', 'closed', 'out of order',
                'not working', 'poor condition', 'unmaintained'
            ]
        }
    }
    
    def __init__(self):
        """Initialize the aspect detector with VADER sentiment analyzer"""
        self.vader = SentimentIntensityAnalyzer()
    
    def detect_aspects(self, review_text: str) -> Dict[str, float]:
        """
        Detect which aspects/categories are mentioned in the review.
        
        Args:
            review_text: The hotel review text
            
        Returns:
            Dictionary mapping category names to confidence scores
        """
        aspects = {}
        review_lower = review_text.lower()
        
        for category, keywords in self.ASPECT_KEYWORDS.items():
            score = 0
            
            # Primary keywords worth more (weight: 3)
            for kw in keywords['primary']:
                if kw in review_lower:
                    score += 3
            
            # Related keywords (weight: 1)
            for kw in keywords['related']:
                if kw in review_lower:
                    score += 1
            
            # Negative indicators boost score if present (weight: 2)
            for kw in keywords['negative_indicators']:
                if kw in review_lower:
                    score += 2
            
            if score > 0:
                aspects[category] = score
        
        return aspects
    
    def analyze_aspect_sentiment(self, review_text: str, aspect_keywords: Dict[str, List[str]]) -> str:
        """
        Analyze sentiment for a specific aspect by extracting relevant sentences.
        Enhanced to handle negations and mixed sentiments better.
        
        Args:
            review_text: The full review text
            aspect_keywords: Dictionary containing 'primary', 'related', 'negative_indicators' keywords
            
        Returns:
            'positive', 'negative', or 'neutral'
        """
        try:
            sentences = sent_tokenize(review_text)
        except:
            # Fallback if sent_tokenize fails
            sentences = [s.strip() for s in re.split(r'[.!?]+', review_text) if s.strip()]
        
        aspect_sentences = []
        all_keywords = (aspect_keywords['primary'] + 
                       aspect_keywords['related'] + 
                       aspect_keywords['negative_indicators'])
        
        # Extract sentences mentioning the aspect
        for sent in sentences:
            sent_lower = sent.lower()
            for kw in all_keywords:
                if kw in sent_lower:
                    aspect_sentences.append(sent)
                    break
        
        if not aspect_sentences:
            return 'neutral'
        
        # Enhanced sentiment analysis
        total_score = 0
        num_sentences = len(aspect_sentences)
        
        for sent in aspect_sentences:
            sent_lower = sent.lower()
            
            # Check for negative indicators first (these override VADER)
            has_negative_indicator = any(
                neg_kw in sent_lower 
                for neg_kw in aspect_keywords['negative_indicators']
            )
            
            # Check for explicit negation words near keywords
            negation_words = ['not', 'no', 'never', 'neither', 'nor', 'nothing', 
                            'nobody', 'none', 'nowhere', 'hardly', 'scarcely',
                            'barely', 'wasn\'t', 'weren\'t', 'isn\'t', 'aren\'t',
                            'won\'t', 'wouldn\'t', 'don\'t', 'doesn\'t', 'didn\'t']
            
            has_negation = any(neg in sent_lower for neg in negation_words)
            
            # Get VADER score
            vader_score = self.vader.polarity_scores(sent)['compound']
            
            # Adjust score based on context
            if has_negative_indicator:
                # Negative indicators strongly suggest negative sentiment
                total_score += -0.8
            elif has_negation and vader_score > 0:
                # Negation with positive words likely means negative
                # Example: "food was not tasty"
                total_score += -0.5
            else:
                # Use VADER score as-is
                total_score += vader_score
        
        # Average score across all aspect sentences
        avg_score = total_score / num_sentences if num_sentences > 0 else 0
        
        # Thresholds (more sensitive to negativity)
        if avg_score < -0.05:  # Lower threshold for negative
            return 'negative'
        elif avg_score > 0.15:  # Higher threshold for positive
            return 'positive'
        else:
            return 'neutral'
    
    def categorize_with_aspects(self, review_text: str, fallback_model=None) -> Tuple[str, Dict]:
        """
        Enhanced category detection with aspect-based analysis.
        
        Args:
            review_text: The review text to categorize
            fallback_model: Optional original category model for fallback
            
        Returns:
            Tuple of (primary_category, aspect_details_dict)
            aspect_details_dict contains detected aspects and their sentiments
        """
        # Step 1: Detect aspects
        aspects = self.detect_aspects(review_text)
        
        aspect_details = {}
        
        if not aspects:
            # No aspects detected, use fallback model if available
            if fallback_model:
                try:
                    category = fallback_model.predict([review_text])[0]
                    return category, {'method': 'fallback_model', 'aspects': {}}
                except:
                    return 'Services', {'method': 'default', 'aspects': {}}
            return 'Services', {'method': 'default', 'aspects': {}}
        
        # Step 2: Analyze sentiment for each detected aspect
        for category, score in aspects.items():
            keywords = self.ASPECT_KEYWORDS[category]
            sentiment = self.analyze_aspect_sentiment(review_text, keywords)
            
            aspect_details[category] = {
                'score': score,
                'sentiment': sentiment
            }
        
        # Step 3: Prioritize negative aspects
        negative_aspects = {
            cat: info for cat, info in aspect_details.items()
            if info['sentiment'] == 'negative'
        }
        
        if negative_aspects:
            # Return the negative aspect with highest score
            primary_category = max(negative_aspects.items(), 
                                 key=lambda x: x[1]['score'])[0]
            return primary_category, {
                'method': 'aspect_based',
                'aspects': aspect_details,
                'negative_aspects': list(negative_aspects.keys())
            }
        
        # Step 4: No negative aspects found
        # Check if fallback model should be used
        if fallback_model:
            try:
                category = fallback_model.predict([review_text])[0]
                return category, {
                    'method': 'fallback_no_negative',
                    'aspects': aspect_details
                }
            except:
                pass
        
        # Return aspect with highest score even if not negative
        primary_category = max(aspects.items(), key=lambda x: x[1])[0]
        return primary_category, {
            'method': 'aspect_based_highest',
            'aspects': aspect_details
        }


# Testing function
if __name__ == "__main__":
    detector = AspectBasedCategoryDetector()
    
    # Test cases
    test_reviews = [
        "We entered the hotel and they welcomed us warmly, but the food was not tasty and delivery time was late",
        "The room was clean but the pool was dirty and broken",
        "Excellent service from staff, very professional",
        "Terrible food, awful smell in the room"
    ]
    
    print("="*70)
    print("ASPECT-BASED CATEGORY DETECTOR - TEST")
    print("="*70)
    
    for i, review in enumerate(test_reviews, 1):
        print(f"\nTest {i}: \"{review}\"")
        category, details = detector.categorize_with_aspects(review)
        print(f"Primary Category: {category}")
        print(f"Method: {details['method']}")
        if 'aspects' in details and details['aspects']:
            print("Detected Aspects:")
            for asp, info in details['aspects'].items():
                print(f"  - {asp}: {info['sentiment']} (score: {info['score']})")
        print("-"*70)
