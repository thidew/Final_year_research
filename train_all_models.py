"""
Master Training Script
Trains all three models in sequence
"""

import subprocess
import sys
from pathlib import Path

def run_training(script_name):
    """Run a training script"""
    print(f"\n{'='*70}")
    print(f"Running {script_name}...")
    print(f"{'='*70}\n")
    
    result = subprocess.run(
        [sys.executable, f"src/{script_name}"],
        capture_output=False,
        text=True
    )
    
    if result.returncode != 0:
        print(f"\n❌ {script_name} failed with exit code {result.returncode}")
        return False
    
    print(f"\n✅ {script_name} completed successfully!")
    return True

def main():
    """Train all models"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║    ADVANCED CUSTOMER FEEDBACK ANALYSIS SYSTEM TRAINING       ║
    ║                                                              ║
    ║    Training all three models for maximum accuracy            ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    scripts = [
        "train_category_model.py",
        "train_recommendation_model.py",
        "train_sentiment_model.py"
    ]
    
    results = {}
    
    for script in scripts:
        success = run_training(script)
        results[script] = success
        
    # Summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    
    for script, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{script:40} {status}")
    
    print("="*70)
    
    all_success = all(results.values())
    
    if all_success:
        print("\n🎉 ALL MODELS TRAINED SUCCESSFULLY!")
        print("\nModels saved in:")
        print("  - models/category_classifier/")
        print("  - models/recommendation_engine/")
        print("  - models/sentiment_analyzer/")
    else:
        print("\n⚠️  Some models failed to train. Check logs above.")
    
    return all_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
