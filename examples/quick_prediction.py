#!/usr/bin/env python3
"""Quick example of using the calorie predictor model."""

import pandas as pd
import pickle
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def predict_workout_calories():
    """Example of making a calorie prediction for a planned workout."""
    
    # Define your planned workout
    workout_plan = pd.DataFrame({
        'DURATION_ACTUAL': [1.5],          # 1.5 hours
        'DISTANCE_ACTUAL': [15000],        # 15 km
        'HRMAX': [185],                    # Expected max HR
        'HRAVG': [155],                    # Expected avg HR
        'AGE': [32],                       # Your age
        'WEIGHT': [72],                    # Your weight in kg
        'SEX_ENCODED': [1],                # 1 for male, 0 for female
        'ELEVATIONAVG': [120],             # Average elevation
        'ELEVATIONGAIN': [150],            # Total elevation gain
        'TRAININGSTRESSSCOREACTUAL': [85], # Training stress score
        'PACE': [0.1],                     # Will be calculated
        'SPEED': [10],                     # Will be calculated
        'INTENSITY_RATIO': [0.84]          # HRAVG/HRMAX
    })
    
    # Load the pre-trained model
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'calorie_prediction_model.pkl')
    
    if not os.path.exists(model_path):
        print("Model not found. Please run the training script first:")
        print("python scripts/train_model.py")
        return
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    # Extract model and feature names
    model = model_data['model']
    feature_names = model_data['feature_names']
    
    # Ensure all required features are present
    workout_features = workout_plan[feature_names]
    
    # Make prediction
    predicted_calories = model.predict(workout_features)
    
    print("\nWorkout Calorie Prediction")
    print("=" * 40)
    print(f"Duration: {workout_plan['DURATION_ACTUAL'][0]:.1f} hours")
    print(f"Distance: {workout_plan['DISTANCE_ACTUAL'][0]/1000:.1f} km")
    print(f"Expected Avg HR: {workout_plan['HRAVG'][0]} bpm")
    print(f"Weight: {workout_plan['WEIGHT'][0]} kg")
    print("=" * 40)
    print(f"Predicted Calorie Burn: {predicted_calories[0]:.0f} calories")
    print(f"Calories per km: {predicted_calories[0]/(workout_plan['DISTANCE_ACTUAL'][0]/1000):.1f}")
    print(f"Calories per hour: {predicted_calories[0]/workout_plan['DURATION_ACTUAL'][0]:.1f}")
    
    # Provide nutrition recommendations
    print("\nNutrition Recommendations:")
    if predicted_calories[0] < 300:
        print("- Light workout: Water should be sufficient")
    elif predicted_calories[0] < 600:
        print("- Moderate workout: Consider a small snack and water")
    elif predicted_calories[0] < 1000:
        print("- Long workout: Bring energy gel/bar and electrolyte drink")
    else:
        print("- Ultra workout: Plan multiple nutrition stops with varied fuel sources")


if __name__ == "__main__":
    predict_workout_calories()