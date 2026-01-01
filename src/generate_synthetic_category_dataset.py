import pandas as pd
import random
from utils import TextPreprocessor

def generate_category_data(n_samples_per_class=1000):
    data = []
    preprocessor = TextPreprocessor()
    
    # --- 1. FOOD (Enhanced) ---
    foods = ['steak', 'pasta', 'soup', 'rice', 'chicken', 'fish', 'curry', 'burger', 'pizza', 'breakfast', 'buffet', 'coffee', 'tea', 'dessert', 'cake', 'salad', 'bread', 'wine', 'beer', 'culinary experience', 'protein', 'sauce', 'beverage selection', 'menu options', 'gastronomy', 'dining']
    food_adjs = ['cold', 'bland', 'salty', 'delicious', 'tasty', 'undercooked', 'overcooked', 'stale', 'tasteless', 'oily', 'greasy', 'raw', 'dry', 'fresh', 'hot', 'yummy', 'spicy', 'disappointing', 'meager', 'non-existent', 'exquisite', 'mediocre']
    
    for _ in range(n_samples_per_class):
        template = random.choice([
            f"The {random.choice(foods)} was {random.choice(food_adjs)}.",
            f"I ordered {random.choice(foods)} and it tasted {random.choice(food_adjs)}.",
            f"Great {random.choice(foods)} but the {random.choice(foods)} was {random.choice(food_adjs)}.",
            f"We ate at the restaurant and had {random.choice(foods)}.",
            f"Breakfast buffet had amazing {random.choice(foods)}.",
            f"Room service brought us cold {random.choice(foods)}.",
            f"Dining experience was bad because of the {random.choice(food_adjs)} {random.choice(foods)}.",
            f"The {random.choice(foods)} presentation was {random.choice(food_adjs)}."
        ])
        data.append({'text': template, 'category': 'Food'})

    # --- 2. ROOMS (Enhanced) ---
    room_items = ['bed', 'AC', 'toilet', 'shower', 'sink', 'TV', 'lights', 'door', 'window', 'carpet', 'sheets', 'wifi', 'fridge', 'balcony', 'view', 'furniture', 'pillow', 'acoustics', 'lavatory', 'suite aesthetic', 'decor', 'sound insulation']
    room_adjs = ['clean', 'dirty', 'comfortable', 'smelly', 'spacious', 'tiny', 'modern', 'old', 'broken', 'cozy', 'dusty', 'noisy', 'quiet', 'terrible', 'questionable', 'trapping', 'dated', 'worn out']
    
    for _ in range(n_samples_per_class):
        template = random.choice([
            f"The {random.choice(room_items)} in my room was {random.choice(room_adjs)}.",
            f"Our room was very {random.choice(room_adjs)} and had a great {random.choice(room_items)}.",
            f"Housekeeping did not clean the {random.choice(room_items)}.",
            f"We loved the {random.choice(room_items)}, it was so {random.choice(room_adjs)}.",
            f"The suite needs renovation, especially the {random.choice(room_items)}.",
            f"Sleeping was hard because of the {random.choice(room_items)}.",
            f"Bathroom {random.choice(room_items)} was {random.choice(room_adjs)}.",
            f"The {random.choice(room_items)} made the stay {random.choice(room_adjs)}."
        ])
        data.append({'text': template, 'category': 'Rooms'})

    # --- 3. SERVICES (Enhanced) ---
    staff_roles = ['waiter', 'receptionist', 'manager', 'bellboy', 'cleaning staff', 'bartender', 'staff', 'concierge', 'driver', 'porter', 'front desk agent', 'hospitality team', 'service personnel']
    staff_adjs = ['rude', 'friendly', 'polite', 'slow', 'fast', 'helpful', 'unhelpful', 'arrogant', 'professional', 'warm', 'welcoming', 'disinterested', 'attentive', 'ignored', 'careless']
    
    for _ in range(n_samples_per_class):
        template = random.choice([
            f"The {random.choice(staff_roles)} was very {random.choice(staff_adjs)}.",
            f"Service was {random.choice(staff_adjs)}, especially the {random.choice(staff_roles)}.",
            f"Check-in was {random.choice(staff_adjs)} thanks to the {random.choice(staff_roles)}.",
            f"We were greeted by a {random.choice(staff_adjs)} {random.choice(staff_roles)}.",
            f"The staff at the front desk were {random.choice(staff_adjs)}.",
            f"Amazing hospitality from the {random.choice(staff_roles)}.",
            f"I complained to the {random.choice(staff_roles)}.",
            f"We waited too long for the {random.choice(staff_roles)}."
        ])
        data.append({'text': template, 'category': 'Services'})

    # --- 4. RECREATION (Enhanced) ---
    activities = ['pool', 'gym', 'spa', 'beach', 'massage', 'tennis court', 'kids club', 'entertainment', 'music', 'bar', 'club', 'excursion', 'tour', 'aquatic facility', 'wellness center', 'fitness area']
    rec_adjs = ['fun', 'boring', 'relaxing', 'crowded', 'clean', 'dirty', 'expensive', 'cheap', 'amazing', 'closed', 'overcrowded', 'lacking', 'dangerous', 'unsafe']
    
    for _ in range(n_samples_per_class):
        template = random.choice([
            f"The {random.choice(activities)} area was {random.choice(rec_adjs)}.",
            f"We spent all day at the {random.choice(activities)}.",
            f"My kids loved the {random.choice(activities)}.",
            f"The hotel has a great {random.choice(activities)}.",
            f"Evening {random.choice(activities)} was {random.choice(rec_adjs)}.",
            f"Swimming in the {random.choice(activities)} was the highlight.",
            f"Don't miss the {random.choice(activities)}, it is {random.choice(rec_adjs)}.",
            f"The {random.choice(activities)} equipment was {random.choice(rec_adjs)}."
        ])
        data.append({'text': template, 'category': 'Recreation'})
        
    df = pd.DataFrame(data).sample(frac=1).reset_index(drop=True)
    df['cleaned_text'] = df['text'].apply(lambda x: preprocessor.clean_text(x, remove_stopwords=True, lemmatize=True))
    return df

if __name__ == "__main__":
    df = generate_category_data(n_samples_per_class=1000)
    
    # Save split files
    from sklearn.model_selection import train_test_split
    train, temp = train_test_split(df, test_size=0.3, random_state=42, stratify=df['category'])
    val, test = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp['category'])
    
    train.to_csv('d:/Thisaru/data/processed/synthetic_category_train.csv', index=False)
    val.to_csv('d:/Thisaru/data/processed/synthetic_category_val.csv', index=False)
    test.to_csv('d:/Thisaru/data/processed/synthetic_category_test.csv', index=False)
    
    print(f"Generated {len(df)} category samples.")
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
