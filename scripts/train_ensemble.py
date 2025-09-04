# Training script that integrates ensemble models with your existing pipeline
# Builds on your current data processing and adds ensemble capabilities

import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_model import DataManager, ModelTrainer
from scripts.ensemble_model import (
    ActivityRouter, 
    WeightedEnsemble, 
    StackingEnsemble, 
    HybridEnsemble,
    compare_ensemble_approaches
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import pickle
from datetime import datetime


def prepare_data_with_activities():
    """
    Load and prepare data including activity types for ensemble training
    """
    print("="*60)
    print("LOADING DATA WITH ACTIVITY TYPES")
    print("="*60)
    
    # Load your Strava data
    data_path = '../data/processed/workout_data.csv'
    df = pd.read_csv(data_path)
    
    # Standardize column names
    df.columns = df.columns.str.upper()
    
    print(f"Loaded {len(df)} total workouts")
    
    # Check if activity type column exists
    activity_columns = ['ACTIVITY_TYPE', 'ACTIVITYTYPE', 'TYPE', 'SPORT']
    activity_col = None
    for col in activity_columns:
        if col in df.columns:
            activity_col = col
            break
    
    if not activity_col:
        print("WARNING: No activity type column found!")
        print("Available columns:", list(df.columns))
        return None, None, None
    
    print(f"Using activity type column: {activity_col}")
    
    # Rename to standard name
    df['ACTIVITY_TYPE'] = df[activity_col]
    
    # Show activity distribution
    print("\nActivity Distribution:")
    activity_counts = df['ACTIVITY_TYPE'].value_counts()
    for activity, count in activity_counts.items():
        print(f"  {activity}: {count}")
    
    # Categorize activities
    distance_activities = ['Run', 'Ride', 'Walk', 'Hike', 'Swim', 'VirtualRide']
    stationary_activities = ['WeightTraining', 'Yoga', 'Workout', 'StairStepper', 'Elliptical']
    
    df['ACTIVITY_CATEGORY'] = 'Other'
    df.loc[df['ACTIVITY_TYPE'].isin(distance_activities), 'ACTIVITY_CATEGORY'] = 'Distance'
    df.loc[df['ACTIVITY_TYPE'].isin(stationary_activities), 'ACTIVITY_CATEGORY'] = 'Stationary'
    
    print("\nActivity Categories:")
    print(df['ACTIVITY_CATEGORY'].value_counts())
    
    return df


def create_features_for_ensemble(df):
    """
    Create features appropriate for different activity types
    """
    print("\n" + "="*60)
    print("FEATURE ENGINEERING FOR ENSEMBLE")
    print("="*60)
    
    # Map Strava columns to expected names
    column_mapping = {
        'DURATION_HOURS': 'DURATION_ACTUAL',
        'DISTANCE_M': 'DISTANCE_ACTUAL',
        'CALORIES': 'CALORIES',
        'HR_AVG': 'HRAVG',
        'HR_MAX': 'HRMAX',
        'ELEV_GAIN_M': 'ELEVATIONGAIN',
        'SEX': 'SEX',
        'AGE': 'AGE',
        'WEIGHT_KG': 'WEIGHT'
    }
    
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns:
            df[new_name] = df[old_name]
    
    # Create activity-appropriate features
    features = []
    
    # Universal features (work for all activities)
    universal_features = ['DURATION_ACTUAL', 'HRAVG', 'HRMAX', 'AGE']
    
    # Distance-specific features
    distance_features = ['DISTANCE_ACTUAL', 'ELEVATIONGAIN', 'PACE', 'ELEVATION_PER_KM']
    
    # Check which features are available
    for feat in universal_features:
        if feat in df.columns:
            features.append(feat)
    
    # Create derived features
    if 'DURATION_ACTUAL' in df.columns and 'DISTANCE_ACTUAL' in df.columns:
        # Handle division by zero for non-distance activities
        df['PACE'] = np.where(
            df['DISTANCE_ACTUAL'] > 0,
            df['DURATION_ACTUAL'] / (df['DISTANCE_ACTUAL'] / 1000),
            0
        )
        
        df['SPEED'] = np.where(
            df['DISTANCE_ACTUAL'] > 0,
            (df['DISTANCE_ACTUAL'] / 1000) / df['DURATION_ACTUAL'],
            0
        )
    
    if 'ELEVATIONGAIN' in df.columns and 'DISTANCE_ACTUAL' in df.columns:
        df['ELEVATION_PER_KM'] = np.where(
            df['DISTANCE_ACTUAL'] > 0,
            df['ELEVATIONGAIN'] / (df['DISTANCE_ACTUAL'] / 1000),
            0
        )
    
    # Intensity features
    if 'HRAVG' in df.columns and 'HRMAX' in df.columns:
        df['INTENSITY_RATIO'] = df['HRAVG'] / df['HRMAX']
        df['HR_RESERVE'] = df['HRMAX'] - 70
        df['HR_ZONE'] = (df['HRAVG'] - 70) / df['HR_RESERVE']
        features.extend(['INTENSITY_RATIO', 'HR_RESERVE', 'HR_ZONE'])
    
    # Sex encoding
    if 'SEX' in df.columns:
        df['SEX_ENCODED'] = (df['SEX'] == 'M').astype(int)
        features.append('SEX_ENCODED')
    
    # Add distance features where appropriate
    distance_mask = df['ACTIVITY_CATEGORY'] == 'Distance'
    
    # For distance activities, add distance-specific features
    distance_feature_list = ['DISTANCE_ACTUAL', 'ELEVATIONGAIN', 'PACE', 'ELEVATION_PER_KM']
    for feat in distance_feature_list:
        if feat in df.columns:
            # Only include if the feature has variance for distance activities
            if df.loc[distance_mask, feat].std() > 0:
                features.append(feat)
    
    # Remove duplicates
    features = list(set(features))
    
    # Remove any features with no variance
    features_to_keep = []
    for feat in features:
        if feat in df.columns and df[feat].std() > 0:
            features_to_keep.append(feat)
    
    print(f"Selected {len(features_to_keep)} features:")
    print(f"  {', '.join(sorted(features_to_keep))}")
    
    return df, features_to_keep


def train_ensemble_comparison():
    """
    Main function to train and compare ensemble approaches
    """
    # Load data with activity types
    df = prepare_data_with_activities()
    if df is None:
        print("Error: Could not load activity types")
        return
    
    # Create features
    df, feature_columns = create_features_for_ensemble(df)
    
    # Clean data
    print("\n" + "="*60)
    print("DATA CLEANING")
    print("="*60)
    
    # Remove invalid records
    initial_rows = len(df)
    
    # Basic cleaning
    df = df.dropna(subset=['CALORIES'] + feature_columns)
    df = df[(df['CALORIES'] > 5) & (df['CALORIES'] < 2000)]
    
    # Activity-specific cleaning
    distance_mask = df['ACTIVITY_CATEGORY'] == 'Distance'
    stationary_mask = df['ACTIVITY_CATEGORY'] == 'Stationary'
    
    # For distance activities, ensure valid distance
    df.loc[distance_mask] = df.loc[distance_mask][
        (df.loc[distance_mask, 'DISTANCE_ACTUAL'] > 0) &
        (df.loc[distance_mask, 'DISTANCE_ACTUAL'] < 100000)  # Less than 100km
    ]
    
    print(f"Cleaned data: {initial_rows} -> {len(df)} records")
    
    # Prepare for modeling
    X = df[feature_columns]
    y = df['CALORIES']
    activity_type = df['ACTIVITY_TYPE']
    
    print(f"\nFinal dataset shape: {X.shape}")
    print(f"Target range: {y.min():.0f} - {y.max():.0f} calories")
    
    # Compare ensemble approaches
    print("\n" + "="*60)
    print("COMPARING ENSEMBLE APPROACHES")
    print("="*60)
    
    results = compare_ensemble_approaches(X, y, activity_type)
    
    # Save best model
    print("\n" + "="*60)
    print("SAVING BEST MODEL")
    print("="*60)
    
    # Retrain best model on full dataset
    best_approach = max(results.items(), key=lambda x: x[1]['r2'])[0]
    
    if best_approach == 'router':
        print("Training final ActivityRouter model...")
        final_model = ActivityRouter()
    elif best_approach == 'weighted':
        print("Training final WeightedEnsemble model...")
        final_model = WeightedEnsemble()
    elif best_approach == 'stacking':
        print("Training final StackingEnsemble model...")
        final_model = StackingEnsemble()
    elif best_approach == 'hybrid':
        print("Training final HybridEnsemble model...")
        final_model = HybridEnsemble()
    else:
        print("Using baseline model...")
        import xgboost as xgb
        final_model = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
    
    # Train on full dataset
    if hasattr(final_model, 'fit') and 'activity' in best_approach.lower():
        final_model.fit(X, y, activity_type)
    else:
        final_model.fit(X, y)
    
    # Save model and metadata
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f'../model_outputs/ensemble_model_{timestamp}.pkl'
    
    model_data = {
        'model': final_model,
        'feature_columns': feature_columns,
        'model_type': best_approach,
        'performance': results[best_approach],
        'activity_types': df['ACTIVITY_TYPE'].unique().tolist(),
        'timestamp': timestamp
    }
    
    os.makedirs('../model_outputs', exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\nModel saved to: {model_path}")
    print(f"Model type: {best_approach}")
    print(f"R² Score: {results[best_approach]['r2']:.4f}")
    print(f"MAE: {results[best_approach]['mae']:.2f} calories")
    
    return model_data


if __name__ == "__main__":
    print("ENSEMBLE MODEL TRAINING FOR CALORIE PREDICTOR")
    print("=" * 60)
    print("\nThis script will:")
    print("1. Load your workout data with activity types")
    print("2. Create appropriate features for different activities")
    print("3. Compare 4 ensemble approaches:")
    print("   - Activity Router (separate models)")
    print("   - Weighted Ensemble (optimal weights)")
    print("   - Stacking (meta-learning)")
    print("   - Hybrid (router + ensembles)")
    print("4. Save the best performing model")
    print("\nStarting training...\n")
    
    train_ensemble_comparison()