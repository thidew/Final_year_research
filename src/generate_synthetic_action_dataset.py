import pandas as pd
import random
from pathlib import Path
import sys

# Add src to path to import utils
sys.path.append(str(Path(__file__).parent))
from utils import TextPreprocessor

def generate_action_data(n_samples_per_action=500):
    """
    Generates synthetic data for granular recommendation actions.
    Output columns: text, action
    """
    data = []
    
    # --- ACTION SCHEMA DEFINITION ---
    # Structure: Category -> { Action -> [Templates] }
    
    action_schema = {
        'Food': {
            'Inspect Kitchen Hygiene': [
                "I found a hair in my food.",
                "The food tasted slightly spoiled and smelled bad.",
                "There was a strange object in the soup.",
                "The plates were dirty and had leftover food stains.",
                "I got food poisoning after eating the seafood.",
                "The buffet area looked unclean and messy.",
                "Saw a cockroach near the kitchen entrance.",
                "Glass had lipstick marks on it.",
                "The salad was wilted and brown.",
                "Hygiene standards in the restaurant seem very low."
            ],
            'Revise Menu Quality': [
                "The steak was tough and flavorless.",
                "The menu options are very limited and boring.",
                "Food was cold when it arrived at the table.",
                "The pasta was overcooked and mushy.",
                "Everything tastes like frozen food reheated.",
                "The presentation of the dishes was poor.",
                "Not enough vegetarian or vegan options.",
                "The coffee was burnt and bitter.",
                "Seasonal dishes listed on the menu were not available.",
                "The dessert was stale and dry."
            ],
            'Staff Training (Food)': [
                "The waiter spilled soup on me and didn't apologize.",
                "Service in the restaurant was incredibly slow.",
                "Staff got my order wrong three times.",
                "The server was rude when I asked for water.",
                "Waiters were ignoring us and chatting among themselves.",
                "Asked for no onions but got onions anyway.",
                "Staff didn't know the ingredients of the dishes.",
                "Restaurant manager was shouting at the staff.",
                "No one came to clear our table for 30 minutes.",
                "The hostess was welcoming but the servers were grumpy."
            ],
            'Compensate Guest': [
                "We waited 2 hours for our food, this is unacceptable.",
                "They charged us for items we didn't order.",
                "Our anniversary dinner was ruined by the terrible service.",
                "I want a refund for this terrible meal.",
                "The bill included a service charge that was not mentioned.",
                "We were promised a complimentary drink but never got it.",
                "Worst dining experience ever, I expect compensation.",
                "They lost our reservation even though we booked weeks ago.",
                "Charged 5 stars price for 1 star food.",
                "They refused to accept my voucher."
            ]
        },
        'Rooms': {
            'Deep Clean & Maintenance': [
                "The carpet was stained and smelled of mold.",
                "Dust everywhere, especially under the bed.",
                "The bathroom tiles had mold in the grout.",
                "Found trash from previous guests in the drawer.",
                "The sheets had spots and didn't look fresh.",
                "Curtains were dusty and triggered my allergies.",
                "The room smelled like stale cigarette smoke.",
                "Windows were dirty and hard to see through.",
                "The toilet wasn't flushed when we arrived.",
                "Dead bugs on the windowsill."
            ],
            'Upgrade Room Amenities': [
                "The TV is tiny and has no channels.",
                "WiFi signal is too weak to work in the room.",
                "The hairdryer provided is useless and overheating.",
                "No iron or ironing board in the room.",
                "The pillows are lumpy and uncomfortable.",
                "Towels are thin, scratchy, and old.",
                "Minibar was empty and not working.",
                "Need better lighting, the room is too dim.",
                "Air conditioner is loud and barely cools the room.",
                "The furniture is worn out and chipped."
            ],
            'Pest Control Investigation': [
                "Bed bugs bit me all over my legs.",
                "Saw ants marching across the bathroom sink.",
                "There was a spider on the pillow.",
                "Mosquitoes in the room kept me up all night.",
                "I think I saw a mouse behind the curtain.",
                "Cockroach in the shower.",
                "Flies buzzing around the room constantly.",
                "Evidence of termites in the wooden furniture.",
                "Lizard on the ceiling.",
                "The room is infested with insects."
            ]
        },
        'Services': {
            'Staff Communication Training': [
                "Front desk staff gave us wrong information about breakfast.",
                "They didn't tell us the pool was under maintenance.",
                "Concierge was unable to give simple directions.",
                "Misunderstanding about the room rate caused issues.",
                "Staff speaks very poor English, hard to communicate.",
                "Called reception but no one answered the phone.",
                "They forgot to give us our wake-up call.",
                "Bellboy didn't explain how to use the key card.",
                "No one told us about the construction noise.",
                "Conflicting information from different staff members."
            ],
            'Review Check-in Process': [
                "It took 45 minutes just to check in.",
                "They couldn't find my booking in the system.",
                "Check-in line was huge and only one counter open.",
                "Room wasn't ready at 4 PM check-in time.",
                "System crashed during check-in.",
                "They demanded a cash deposit unexpectedly.",
                "Key cards didn't work, had to go back down.",
                "Lobby is chaotic and disorganized.",
                "Priority check-in for members was not honored.",
                "Staff seemed confused by the check-in software."
            ],
            'Disciplinary Action': [
                "The concierge made a rude comment about my appearance.",
                "Staff member was smoking right outside the lobby door.",
                "Security guard was sleeping on the job.",
                "Housekeeper stole money from my wallet.",
                "Bartender was visibly drunk.",
                "Staff was arguing loudly in the hallway.",
                "Valet scratched my car and denied it.",
                "Employee was on their phone ignoring guests.",
                "Unprofessional behavior from the night manager.",
                "Staff member laughed at my complaint."
            ]
        },
        'Recreation': {
            'Pool Maintenance': [
                "The pool water was cloudy and green.",
                "Tiles around the pool are broken and sharp.",
                "Too much chlorine in the pool, burned my eyes.",
                "Pool was freezing cold, heater broken.",
                "Debris and leaves floating in the water.",
                "Pool ladders are loose and dangerous.",
                "Scum line around the edge of the pool.",
                "Pool shower wasn't working.",
                "Water looked stagnant and dirty.",
                " Filtration system was making a loud noise."
            ],
            'Update Gym Equipment': [
                "Gym equipment is ancient and rusty.",
                "Treadmills were all out of order.",
                "Not enough weights for a proper workout.",
                "The gym smells like sweat, no ventilation.",
                "Yoga mats are torn and dirty.",
                "Cable machine is stuck and dangerous.",
                "No water or towels available in the gym.",
                "Bike seat is broken.",
                "Elliptical machine makes a squeaking sound.",
                "Gym is too small for the number of guests."
            ],
            'Recreation Staff Review': [
                "Lifeguard was on their phone the whole time.",
                "Spa therapist was chatting during the massage.",
                "Kids club staff seemed bored and inattentive.",
                "Activity director was rude to the guests.",
                "Pool attendant refused to give me a towel.",
                "Tour guide gave us incorrect historical info.",
                "Entertainment team was low energy and boring.",
                "Tennis coach didn't show up for the lesson.",
                "Beach staff demanded tips for chairs.",
                "No supervision in the children's play area."
            ]
        }
    }
    
    print(f"Generating data with {n_samples_per_action} samples per action...")
    
    for category, actions_dict in action_schema.items():
        for action, templates in actions_dict.items():
            
            # Generate variations
            for _ in range(n_samples_per_action):
                base_text = random.choice(templates)
                
                # Simple variations to increase diversity
                variations = [
                    base_text,
                    base_text.lower(),
                    f"Honestly, {base_text.lower()}",
                    f"{base_text} Terrible.",
                    f"Not happy. {base_text}",
                    f"Action needed: {base_text}",
                    f"Review: {base_text}",
                    f"Feedback: {base_text}"
                ]
                
                text = random.choice(variations)
                
                data.append({
                    'text': text,
                    'action': action,
                    'category': category # Keep category for reference if needed
                })
                
    df = pd.DataFrame(data)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Generated {len(df)} total samples.")
    print("\nAction Distribution:")
    print(df['action'].value_counts())
    
    return df

if __name__ == "__main__":
    df = generate_action_data(n_samples_per_action=200) # 200 * ~12 actions = ~2400 samples
    
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "synthetic_recommendation_actions.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved to {output_path}")
