# main.py
"""
Main Application Entry Point
Automobile Recommendation System
Author: Prathamesh Parab
BDA Project - SFIT 2023-24
"""

import os
import sys
from pathlib import Path

def check_requirements():
    """Check if all required files and packages are present"""
    required_files = ['cartest.csv', 'model_trainer.py', 'predictor.py', 'gui.py']
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    # Check if models exist
    if not os.path.exists('models/car_model.pkl'):
        print("📊 Models not found. Training new model...")
        from model_trainer import CarModelTrainer
        trainer = CarModelTrainer('cartest.csv')
        trainer.prepare_data()
        trainer.train_model()
        trainer.save_model()
        print("✓ Model training complete!\n")
    
    return True

def main():
    """Main application entry point"""
    print("=" * 60)
    print("AUTOMOBILE RECOMMENDATION SYSTEM")
    print("Developed by: Prathamesh Parab")
    print("=" * 60)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Please ensure all required files are present.")
        sys.exit(1)
    
    print("\n✓ All checks passed!")
    print("🚗 Starting application...\n")
    
    # Import and run GUI
    try:
        from gui import CarRecommendationGUI
        app = CarRecommendationGUI()
        app.run()
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()