# predictor.py
"""
Prediction Module for Automobile Recommendation System
Handles all prediction logic
Author: Prathamesh Parab
"""

import pickle
import numpy as np
import os

class CarPredictor:
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.model = None
        self.label_encoders = {}
        self.target_encoder = None
        self.category_cars = {}
        self.metadata = {}
        self.load_models()
    
    def load_models(self):
        """Load all saved models and encoders"""
        try:
            # Load main model
            with open(f'{self.models_dir}/car_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            
            # Load encoders
            with open(f'{self.models_dir}/label_encoders.pkl', 'rb') as f:
                self.label_encoders = pickle.load(f)
            
            with open(f'{self.models_dir}/target_encoder.pkl', 'rb') as f:
                self.target_encoder = pickle.load(f)
            
            # Load category cars mapping
            with open(f'{self.models_dir}/category_cars.pkl', 'rb') as f:
                self.category_cars = pickle.load(f)
            
            # Load metadata
            with open(f'{self.models_dir}/metadata.pkl', 'rb') as f:
                self.metadata = pickle.load(f)
            
            print(f"✓ Models loaded successfully (Accuracy: {self.metadata['accuracy']:.1%})")
            
        except FileNotFoundError:
            print("❌ Model files not found! Please run model_trainer.py first.")
            raise
    
    def validate_input(self, year, price, fuel_type, gear_type, manufacturer, color):
        """Validate user inputs"""
        errors = []
        
        # Year validation
        if year < 2005 or year > 2024:
            errors.append("Year must be between 2005-2024")
        
        # Price validation
        if price < 5000 or price > 200000:
            errors.append("Price must be between $5,000-200,000")
        
        # Check if categorical values exist
        if fuel_type not in self.label_encoders['Fuel type'].classes_:
            errors.append(f"Invalid fuel type: {fuel_type}")
        
        if gear_type not in self.label_encoders['Gear box type'].classes_:
            errors.append(f"Invalid transmission type: {gear_type}")
        
        if manufacturer not in self.label_encoders['Manufacturer'].classes_:
            errors.append(f"Invalid manufacturer: {manufacturer}")
        
        if color not in self.label_encoders['Color'].classes_:
            errors.append(f"Invalid color: {color}")
        
        return errors
    
    def predict(self, year, price, fuel_type, gear_type, manufacturer, color):
        """Make a car recommendation"""
        
        # Validate inputs
        errors = self.validate_input(year, price, fuel_type, gear_type, manufacturer, color)
        if errors:
            return {
                'success': False,
                'errors': errors
            }
        
        try:
            # Encode inputs
            fuel_encoded = self.label_encoders['Fuel type'].transform([fuel_type])[0]
            gear_encoded = self.label_encoders['Gear box type'].transform([gear_type])[0]
            manufacturer_encoded = self.label_encoders['Manufacturer'].transform([manufacturer])[0]
            color_encoded = self.label_encoders['Color'].transform([color])[0]
            
            # Prepare features
            features = np.array([[year, price, fuel_encoded, gear_encoded, 
                                 manufacturer_encoded, color_encoded]])
            
            # Make prediction
            prediction_encoded = self.model.predict(features)[0]
            prediction_category = self.target_encoder.inverse_transform([prediction_encoded])[0]
            
            # Get specific car recommendations
            specific_cars = self.category_cars.get(prediction_category, 
                                                   ["Model based on your preferences"])
            
            return {
                'success': True,
                'category': prediction_category,
                'recommended_cars': specific_cars[:3],
                'confidence': self.metadata['accuracy']
            }
            
        except Exception as e:
            return {
                'success': False,
                'errors': [str(e)]
            }
    
    def get_available_options(self):
        """Get all available options for dropdowns"""
        return {
            'fuel_types': list(self.label_encoders['Fuel type'].classes_),
            'gear_types': list(self.label_encoders['Gear box type'].classes_),
            'manufacturers': list(self.label_encoders['Manufacturer'].classes_),
            'colors': list(self.label_encoders['Color'].classes_),
            'accuracy': self.metadata['accuracy']
        }