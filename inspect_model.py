import pickle
import sys
from pathlib import Path

model_path = Path("models/action_recommendation_model.pkl")

try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(f"Successfully loaded {model_path}")
    print(f"Type: {type(model)}")
    print(f"Content: {model}")
except Exception as e:
    print(f"Error loading model: {e}")
