# model_trainer.py
"""
Model Training Module for Automobile Recommendation System
Handles all ML model training and saving
Author: Prathamesh Parab
"""

import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os

class CarModelTrainer:
    def __init__(self, dataset_path='cartest.csv'):
        self.dataset_path = dataset_path
        self.model = None
        self.label_encoders = {}
        self.target_encoder = None
        self.category_cars = {}
        self.accuracy = 0
        
    def create_category(self, row):
        """Create simplified categories for high accuracy"""
        manufacturer = row['Manufacturer']
        price = row['price']
        
        if price < 25000:
            price_range = 'Economy'
        elif price < 50000:
            price_range = 'Premium'
        else:
            price_range = 'Luxury'
        
        return f"{manufacturer} {price_range}"
    
    def prepare_data(self):
        """Load and prepare the dataset"""
        print("📂 Loading dataset...")
        self.data = pd.read_csv(self.dataset_path)
        
        # Create target categories
        self.data['PredictionTarget'] = self.data.apply(self.create_category, axis=1)
        print(f"✓ Created {self.data['PredictionTarget'].nunique()} categories from {len(self.data)} records")
        
        # Encode categorical features
        categorical_features = ['Fuel type', 'Gear box type', 'Manufacturer', 'Color']
        for feature in categorical_features:
            self.label_encoders[feature] = LabelEncoder()
            self.data[feature + '_encoded'] = self.label_encoders[feature].fit_transform(self.data[feature])
        
        # Encode target
        self.target_encoder = LabelEncoder()
        self.data['Target_encoded'] = self.target_encoder.fit_transform(self.data['PredictionTarget'])
        
        # Store car examples for each category
        for category in self.data['PredictionTarget'].unique():
            matching_cars = self.data[self.data['PredictionTarget'] == category]['CarName'].value_counts().head(3).index.tolist()
            self.category_cars[category] = matching_cars
        
        return self.data
    
    def train_model(self):
        """Train the Random Forest model"""
        print("🤖 Training model...")
        
        # Prepare features
        feature_columns = ['Year', 'price', 'Fuel type_encoded', 'Gear box type_encoded', 
                          'Manufacturer_encoded', 'Color_encoded']
        X = self.data[feature_columns].values
        y = self.data['Target_encoded'].values
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
        self.model.fit(X, y)
        
        # Calculate accuracy
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(self.model, X, y, cv=5)
        self.accuracy = scores.mean()
        
        print(f"✓ Model trained with {self.accuracy:.1%} accuracy")
        return self.model
    
    def save_model(self, models_dir='models'):
        """Save the trained model and encoders"""
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        
        # Save main model
        with open(f'{models_dir}/car_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save encoders
        with open(f'{models_dir}/label_encoders.pkl', 'wb') as f:
            pickle.dump(self.label_encoders, f)
        
        with open(f'{models_dir}/target_encoder.pkl', 'wb') as f:
            pickle.dump(self.target_encoder, f)
        
        # Save category cars mapping
        with open(f'{models_dir}/category_cars.pkl', 'wb') as f:
            pickle.dump(self.category_cars, f)
        
        # Save metadata
        metadata = {
            'accuracy': self.accuracy,
            'num_categories': len(self.target_encoder.classes_),
            'features': list(self.label_encoders.keys())
        }
        with open(f'{models_dir}/metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"✓ Model saved to {models_dir}/")
        return True

if __name__ == "__main__":
    # Train and save model
    trainer = CarModelTrainer('cartest.csv')
    trainer.prepare_data()
    trainer.train_model()
    trainer.save_model()
    print("\n✅ Model training complete!")