#!/usr/bin/env python3
"""
Split Model Trainer for Fitness Calorie Prediction

Trains separate models for distance-based vs non-distance activities.
Uses only essential features to avoid data leakage and achieve realistic performance.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SplitModelTrainer:
    """Trains separate models for distance vs non-distance activities."""
    
    def __init__(self):
        self.distance_models = {}
        self.non_distance_models = {}
        self.best_distance_model = None
        self.best_non_distance_model = None
        self.distance_features = None
        self.non_distance_features = None
        
    def load_and_prepare_data(self, filepath='data/workouts.csv'):
        """Load and prepare the workout data."""
        print("Loading workout data...")
        
        # Try to load varied weight data first, fallback to regular data
        try:
            df = pd.read_csv('data/workouts_with_varied_weight.csv')
            print("Loaded data with varied weight")
        except FileNotFoundError:
            df = pd.read_csv(filepath)
            print("Loaded standard data (constant weight)")
        
        # Standardize column names
        df.columns = df.columns.str.upper()
        
        # Map Strava columns to expected names
        column_mapping = {
            'DURATION_HOURS': 'DURATION_ACTUAL',
            'DISTANCE_M': 'DISTANCE_ACTUAL',
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
        
        # Basic cleaning
        df = df.dropna(subset=['CALORIES', 'DURATION_ACTUAL', 'HRAVG', 'HRMAX'])
        df = df[(df['CALORIES'] > 5) & (df['CALORIES'] < 2000)]
        df = df[(df['DURATION_ACTUAL'] > 0.0167) & (df['DURATION_ACTUAL'] < 5)]  # 1 min to 5 hours
        
        print(f"Loaded {len(df)} valid workouts")
        return df
    
    def classify_activities(self, df):
        """Classify activities into distance vs non-distance categories."""
        distance_activities = ['Run', 'Ride', 'Walk', 'Hike', 'Swim']
        non_distance_activities = ['WeightTraining', 'Yoga', 'Workout', 'Elliptical']
        
        # Create activity type mapping
        df['ACTIVITY_CATEGORY'] = 'Other'
        
        for activity in distance_activities:
            df.loc[df['ACTIVITY_TYPE'].str.contains(activity, case=False, na=False), 'ACTIVITY_CATEGORY'] = 'Distance'
        
        for activity in non_distance_activities:
            df.loc[df['ACTIVITY_TYPE'].str.contains(activity, case=False, na=False), 'ACTIVITY_CATEGORY'] = 'Non-Distance'
        
        # Filter to only include classified activities
        classified_df = df[df['ACTIVITY_CATEGORY'].isin(['Distance', 'Non-Distance'])].copy()
        
        print(f"Activity distribution:")
        print(classified_df['ACTIVITY_CATEGORY'].value_counts())
        
        return classified_df
    
    def create_features(self, df):
        """Create features for both activity types."""
        # Distance activities features (include duration and distance)
        distance_features = ['DURATION_ACTUAL', 'DISTANCE_ACTUAL', 'HRAVG', 'HRMAX', 'ELEVATIONGAIN', 'AGE', 'SEX']
        
        # Non-distance activities features (include duration but NO distance)
        non_distance_features = ['DURATION_ACTUAL', 'HRAVG', 'HRMAX', 'AGE', 'SEX']
        
        # NOTE: Removed PACE calculation to avoid feature redundancy
        # PACE = DURATION_ACTUAL / DISTANCE_ACTUAL is just a mathematical transformation
        # of the same information, which can cause overfitting
        
        # Check if age is constant and calculate dynamic age if needed
        if 'AGE' in df.columns and 'DATE' in df.columns:
            age_variance = df['AGE'].nunique()
            date_range_years = (pd.to_datetime(df['DATE'].max()) - pd.to_datetime(df['DATE'].min())).days / 365.25
            
            if age_variance == 1 and date_range_years > 1:
                print(f"\n++ Age Analysis ++")
                print(f"Age variance: {age_variance} unique value(s)")
                print(f"Date range: {date_range_years:.1f} years")
                print(f"Static age: {df['AGE'].iloc[0]}")
                print(f"Calculating dynamic age from birthday...")
                
                # Use your birthday: 1991-03-13
                birthday = pd.to_datetime('1991-03-13')
                print(f"Birthday: {birthday.strftime('%Y-%m-%d')}")
                
                # Calculate dynamic age for each workout
                df['AGE_DYNAMIC'] = (pd.to_datetime(df['DATE']) - birthday).dt.days / 365.25
                print(f"Dynamic age range: {df['AGE_DYNAMIC'].min():.1f} to {df['AGE_DYNAMIC'].max():.1f} years")
                
                # Replace static age with dynamic age
                df['AGE'] = df['AGE_DYNAMIC']
                print("Replaced static age with dynamic age calculation")
        
        # Encode sex for both
        df['SEX_ENCODED'] = (df['SEX'] == 'M').astype(int)
        distance_features = [f.replace('SEX', 'SEX_ENCODED') for f in distance_features]
        non_distance_features = [f.replace('SEX', 'SEX_ENCODED') for f in non_distance_features]
        
        # Filter to only available features
        available_distance_features = [f for f in distance_features if f in df.columns]
        available_non_distance_features = [f for f in non_distance_features if f in df.columns]
        
        self.distance_features = available_distance_features
        self.non_distance_features = available_non_distance_features
        
        print(f"Distance features: {available_distance_features}")
        print(f"Non-distance features: {available_non_distance_features}")
        
        return df
    
    def train_models(self, df):
        """Train models for both activity types."""
        # Split data
        distance_df = df[df['ACTIVITY_CATEGORY'] == 'Distance'].copy()
        non_distance_df = df[df['ACTIVITY_CATEGORY'] == 'Non-Distance'].copy()
        
        print(f"\nData split:")
        print(f"  Distance activities: {len(distance_df)} samples")
        print(f"  Non-distance activities: {len(non_distance_df)} samples")
        
        # Train distance models
        if len(distance_df) > 10:
            print(f"\n{'='*50}")
            print("TRAINING DISTANCE ACTIVITY MODELS")
            print(f"{'='*50}")
            self.distance_models = self._train_activity_models(
                distance_df, self.distance_features, "Distance"
            )
        
        # Train non-distance models
        if len(non_distance_df) > 10:
            print(f"\n{'='*50}")
            print("TRAINING NON-DISTANCE ACTIVITY MODELS")
            print(f"{'='*50}")
            self.non_distance_models = self._train_activity_models(
                non_distance_df, self.non_distance_features, "Non-Distance"
            )
    
    def _train_activity_models(self, df, features, activity_type):
        """Train models for a specific activity type."""
        if len(df) < 10:
            print(f"Not enough data for {activity_type} activities")
            return {}
        
        # Prepare features and target
        X = df[features].copy()
        y = df['CALORIES']
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Define models
        models = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(random_state=42),
            'XGBoost': xgb.XGBRegressor(random_state=42, verbosity=0),
            'LightGBM': lgb.LGBMRegressor(random_state=42, verbosity=-1),
            'SVR': SVR(kernel='rbf')
        }
        
        results = {}
        
        print(f"\nTraining {activity_type} models...")
        print("-" * 50)
        
        for name, model in models.items():
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                
                # Test predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                cv_r2_mean = cv_scores.mean()
                cv_r2_std = cv_scores.std()
                
                results[name] = {
                    'model': model,
                    'r2': r2,
                    'mae': mae,
                    'rmse': rmse,
                    'cv_r2_mean': cv_r2_mean,
                    'cv_r2_std': cv_r2_std
                }
                
                print(f"{name:<20} R²={r2:.4f} MAE={mae:.1f} RMSE={rmse:.1f} CV={cv_r2_mean:.4f}±{cv_r2_std:.4f}")
                
            except Exception as e:
                print(f"{name:<20} FAILED: {str(e)}")
        
        return results
    
    def select_best_models(self):
        """Select the best model for each activity type."""
        print(f"\n{'='*50}")
        print("MODEL SELECTION")
        print(f"{'='*50}")
        
        # Select best distance model
        if self.distance_models:
            best_distance_name = max(self.distance_models.keys(), 
                                   key=lambda x: self.distance_models[x]['r2'])
            self.best_distance_model = best_distance_name
            best_distance_r2 = self.distance_models[best_distance_name]['r2']
            print(f"Best distance model: {best_distance_name} (R²={best_distance_r2:.4f})")
        
        # Select best non-distance model
        if self.non_distance_models:
            best_non_distance_name = max(self.non_distance_models.keys(), 
                                       key=lambda x: self.non_distance_models[x]['r2'])
            self.best_non_distance_model = best_non_distance_name
            best_non_distance_r2 = self.non_distance_models[best_non_distance_name]['r2']
            print(f"Best non-distance model: {best_non_distance_name} (R²={best_non_distance_r2:.4f})")
    
    def print_final_summary(self):
        """Print the final summary."""
        print(f"\n{'='*60}")
        print("FINAL SUMMARY")
        print(f"{'='*60}")
        
        if self.best_distance_model and self.distance_models:
            distance_r2 = self.distance_models[self.best_distance_model]['r2']
            print(f"Distance activities → {self.best_distance_model} (R²={distance_r2:.3f})")
        
        if self.best_non_distance_model and self.non_distance_models:
            non_distance_r2 = self.non_distance_models[self.best_non_distance_model]['r2']
            print(f"Non-distance activities → {self.best_non_distance_model} (R²={non_distance_r2:.3f})")
        
        print(f"\nOptimal combination:")
        if self.best_distance_model and self.best_non_distance_model:
            print(f"  Distance → {self.best_distance_model}")
            print(f"  Non-distance → {self.best_non_distance_model}")


def main():
    """Main training pipeline."""
    print("SPLIT MODEL TRAINER")
    print("=" * 50)
    print("Training separate models for distance vs non-distance activities")
    print("Using only essential features to avoid data leakage")
    
    # Initialize trainer
    trainer = SplitModelTrainer()
    
    # Load and prepare data
    df = trainer.load_and_prepare_data()
    
    # Classify activities
    df = trainer.classify_activities(df)
    
    # Create features
    df = trainer.create_features(df)
    
    # Train models
    trainer.train_models(df)
    
    # Select best models
    trainer.select_best_models()
    
    # Final summary
    trainer.print_final_summary()
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
