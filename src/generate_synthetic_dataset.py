import pandas as pd
import random
import numpy as np

def generate_batch(n_samples_per_class=600):
    data = []
    
    # --- 1. Review Menu & Kitchen Quality ---
    foods = ['steak', 'pasta', 'soup', 'rice', 'chicken', 'fish', 'curry', 'burger', 'pizza', 'breakfast', 'buffet', 'coffee', 'tea', 'dessert', 'cake']
    food_adjs = ['cold', 'bland', 'salty', 'undercooked', 'overcooked', 'stale', 'tasteless', 'oily', 'greasy', 'raw', 'dry', 'burned', 'inadible', 'awful']
    food_actions = ['tasted terrible', 'was not good', 'smelled bad', 'made me sick', 'was disappointing', 'lacked flavor', 'was waste of money', 'was not fresh']
    
    for _ in range(n_samples_per_class):
        template = random.choice([
            f"The {random.choice(foods)} was {random.choice(food_adjs)}.",
            f"I ordered {random.choice(foods)} but it {random.choice(food_actions)}.",
            f"Food quality is {random.choice(food_adjs)}, especially the {random.choice(foods)}.",
            f"Kitchen needs improvement, {random.choice(foods)} was {random.choice(food_adjs)}.",
            f"We found the {random.choice(foods)} to be {random.choice(food_adjs)} and {random.choice(food_adjs)}.",
            f"Terrible dining experience, {random.choice(foods)} {random.choice(food_actions)}."
        ])
        data.append({'text': template, 'category': 'Food', 'action': 'Review Menu & Kitchen Quality'})

    # --- 2. Staff Training Required ---
    staff_roles = ['waiter', 'receptionist', 'manager', 'bellboy', 'cleaning staff', 'bartender', 'staff', 'concierge', 'employee', 'server']
    staff_adjs = ['rude', 'slow', 'unhelpful', 'arrogant', 'lazy', 'impolite', 'unfriendly', 'aggressive', 'incompetent', 'unprofessional', 'mean', 'angry']
    staff_actions = ['ignored us', 'yelled at me', 'rolled their eyes', 'forgot our order', 'kept us waiting', 'refused to help', 'was on their phone', 'argued with us', 'did check on us', 'scolded me']
    
    for _ in range(n_samples_per_class):
        template = random.choice([
            f"The {random.choice(staff_roles)} was extremely {random.choice(staff_adjs)}.",
            f"Service was terrible, the {random.choice(staff_roles)} {random.choice(staff_actions)}.",
            f"We were treated poorly by a {random.choice(staff_adjs)} {random.choice(staff_roles)}.",
            f"Staff training is needed, {random.choice(staff_roles)} was {random.choice(staff_adjs)}.",
            f"I complained to the {random.choice(staff_roles)} but they {random.choice(staff_actions)}.",
            f"Very {random.choice(staff_adjs)} behavior from the {random.choice(staff_roles)}.",
            # HARD EXAMPLES: Service issues at Recreation locations (Bar, Spa, Pool)
            f"Nobody at the {random.choice(['bar', 'pool bar', 'spa desk'])} seemed interested in serving us.",
            f"The guy at the {random.choice(['gym', 'tennis court', 'pool'])} was very {random.choice(staff_adjs)}.",
            f"I asked for a towel at the pool but the staff {random.choice(staff_actions)}.",
            f"Zero service at the {random.choice(['beach bar', 'lounge', 'rooftop bar'])}, we waited 20 mins."
        ])
        data.append({'text': template, 'category': 'Services', 'action': 'Staff Training Required'})

    # --- 3. Inspect Room & Maintenance ---
    room_items = ['bed', 'AC', 'air conditioner', 'toilet', 'shower', 'sink', 'TV', 'lights', 'door', 'window', 'carpet', 'sheets', 'wifi', 'fridge', 'balcony', 'lock', 'ceiling', 'fan']
    room_issues = ['broken', 'dirty', 'smelly', 'not working', 'leaking', 'stained', 'dusty', 'moldy', 'noisy', 'clogged', 'old', 'damaged', 'damp', 'wet', 'smells bad', 'smells like dog', 'wont lock']
    
    for _ in range(n_samples_per_class):
        template = random.choice([
            f"The {random.choice(room_items)} in my room was {random.choice(room_issues)}.",
            f"Room maintenance is poor, {random.choice(room_items)} kept {random.choice(['making noise', 'failing', 'leaking'])}.",
            f"Bathroom was disgusting, {random.choice(room_items)} was {random.choice(room_issues)}.",
            f"We had issues with the {random.choice(room_items)}, it was {random.choice(room_issues)}.",
            f"Please fix the {random.choice(room_items)}, it is {random.choice(room_issues)}.",
            f"Room 201 has a {random.choice(room_issues)} {random.choice(room_items)}.",
            # HARD EXAMPLES (Fixed via generalization test)
            f"There is a weird dripping sound coming from the {random.choice(['ceiling', 'bathroom', 'AC'])}.",
            f"The {random.choice(['carpet', 'curtain', 'bed', 'room'])} smells like {random.choice(['wet dog', 'smoke', 'mildew', 'sewage'])}.",
            f"I cannot get the {random.choice(['balcony door', 'window', 'safe', 'main door'])} to {random.choice(['lock', 'close', 'open'])}.",
            f"The {random.choice(room_items)} makes a loud banging noise all night."
        ])
        data.append({'text': template, 'category': 'Rooms', 'action': 'Inspect Room & Maintenance'})

    # --- 4. Upgrade Pool & Activities ---
    activities = ['pool', 'gym', 'spa', 'beach', 'entertainment', 'kids club', 'tennis court', 'bar', 'music', 'activities', 'games']
    activity_adjs = ['boring', 'crowded', 'dirty', 'small', 'outdated', 'closed', 'cold', 'expensive', 'useless', 'empty']
    
    for _ in range(n_samples_per_class):
        template = random.choice([
            f"The {random.choice(activities)} was very {random.choice(activity_adjs)}.",
            f"Nothing to do here, {random.choice(activities)} is {random.choice(activity_adjs)}.",
            f"We were bored, the {random.choice(activities)} needs an upgrade.",
            f"The {random.choice(activities)} area is {random.choice(activity_adjs)} and needs attention.",
            f"Disappointed with the {random.choice(activities)}, it was {random.choice(activity_adjs)}.",
            f"Hotel needs more fun, {random.choice(activities)} was {random.choice(activity_adjs)}."
        ])
        data.append({'text': template, 'category': 'Recreation', 'action': 'Upgrade Pool & Activities'})
        
    return pd.DataFrame(data).sample(frac=1).reset_index(drop=True)

if __name__ == "__main__":
    # Generate Train
    train_df = generate_batch(n_samples_per_class=600)
    train_df.to_csv('d:/Thisaru/data/processed/synthetic_recommendation_train.csv', index=False)
    print(f"Saved Train: {len(train_df)} rows")
    
    # Generate Validation
    val_df = generate_batch(n_samples_per_class=50) # 200 total
    val_df.to_csv('d:/Thisaru/data/processed/synthetic_recommendation_val.csv', index=False)
    print(f"Saved Val: {len(val_df)} rows")
