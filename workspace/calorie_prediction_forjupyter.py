"""
Calorie Prediction for Jupyter - Modular Version
Broken down into focused classes for easier use in notebooks
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import lightgbm as lgb
import pickle
import os

class DataManager:
    """Handles data loading, cleaning, and basic operations"""
    
    def __init__(self, filepath, sample_size=None):
        self.filepath = filepath
        self.sample_size = sample_size
        self.df = None
        self.df_cleaned = None
        
    def load_data(self):
        """Load and sample data"""
        print("Loading data...")
        self.df = pd.read_csv(self.filepath)
        self.df.columns = self.df.columns.str.upper()
        
        if self.sample_size:
            # Shuffle the data first to ensure random sampling
            self.df = self.df.sample(frac=1.0, random_state=42).reset_index(drop=True)
            self.df = self.df.sample(n=min(self.sample_size, len(self.df)), random_state=42).reset_index(drop=True)
            print(f"Randomly sampled {len(self.df):,} records")
        else:
            print(f"Using all {len(self.df):,} records")
            
        return self.df
    
    def analyze_missing_data(self):
        """Analyze missing data patterns"""
        print("\n--- MISSING DATA ANALYSIS ---")
        
        key_features = ['CALORIES', 'DURATION_ACTUAL', 'DISTANCE_ACTUAL', 'HRMAX', 'HRAVG', 
                       'ELEVATIONAVG', 'ELEVATIONGAIN', 'TRAININGSTRESSSCOREACTUAL', 'AGE', 'WEIGHT', 'SEX']
        
        missing_stats = {}
        for col in key_features:
            if col in self.df.columns:
                missing_count = self.df[col].isnull().sum()
                missing_pct = (missing_count / len(self.df)) * 100
                missing_stats[col] = {'count': missing_count, 'percentage': missing_pct}
                print(f"{col:<25}: {missing_count:>6,} missing ({missing_pct:>5.1f}%)")
        
        return missing_stats
    
    def clean_data(self):
        """Clean data by removing invalid records"""
        print("\nCleaning data...")
        initial_rows = len(self.df)
        
        # Analyze extreme values before removal
        extreme_calories = (self.df['CALORIES'] < 5) | (self.df['CALORIES'] > 2000) | (self.df['CALORIES'].isnull())
        extreme_duration = (self.df['DURATION_ACTUAL'] < 0.0167) | (self.df['DURATION_ACTUAL'] > 5)
        extreme_distance = (self.df['DISTANCE_ACTUAL'] < 100) | (self.df['DISTANCE_ACTUAL'] > 100000)
        
        calories_removed = extreme_calories.sum()
        duration_removed = extreme_duration.sum()
        distance_removed = extreme_distance.sum()
        
        # Remove invalid records (using NOT logic to remove outliers)
        self.df = self.df[~(extreme_calories | extreme_duration | extreme_distance)]
        
        final_rows = len(self.df)
        total_removed = initial_rows - final_rows
        
        print(f"\n--- CLEANING SUMMARY ---")
        print(f"Records flagged for invalid calories (< 5 or > 2000): {calories_removed:,}")
        print(f"Records flagged for invalid duration (< 1 minute or > 5 hours): {duration_removed:,}")
        print(f"Records flagged for invalid distance (< 100 m or > 100 km): {distance_removed:,}")
        print(f"Total rows removed: {total_removed:,}")
        print(f"Final rows after cleaning: {final_rows:,}")
        print(f"Data retention: {((final_rows/initial_rows)*100):.1f}%")
        
        self.df_cleaned = self.df.copy()
        return self.df
    
    def analyze_athlete_distribution(self):
        """Analyze athlete distribution"""
        if 'ATHLETE_ID' not in self.df.columns:
            print("No ATHLETE_ID column found")
            return
        
        print("\n--- ATHLETE DISTRIBUTION ANALYSIS ---")
        
        athlete_counts = self.df['ATHLETE_ID'].value_counts()
        total_athletes = len(athlete_counts)
        total_activities = len(self.df)
        
        print(f"Total athletes: {total_athletes:,}")
        print(f"Total activities: {total_activities:,}")
        print(f"Average activities per athlete: {total_activities/total_athletes:.1f}")
        
        # Activity count distribution
        activity_ranges = [
            (1, 5, "1-5 activities"),
            (6, 15, "6-15 activities"),
            (16, 30, "16-30 activities"),
            (31, 50, "31-50 activities"),
            (51, float('inf'), "50+ activities")
        ]
        
        print("\nAthletes by activity count:")
        for min_act, max_act, label in activity_ranges:
            if max_act == float('inf'):
                count = (athlete_counts >= min_act).sum()
            else:
                count = ((athlete_counts >= min_act) & (athlete_counts <= max_act)).sum()
            percentage = (count / total_athletes) * 100
            print(f"  {label:<15}: {count:>6,} athletes ({percentage:>5.1f}%)")
    
    def get_data(self):
        """Return the current dataset"""
        return self.df


class DataVisualizer:
    """Handles all data visualization and analysis"""
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.df = data_manager.df
    
    def visualize_raw_data(self):
        """Visualize data before cleaning"""
        print("Creating pre-cleaning visualizations...")
        
        # Update dataframe reference
        self.df = self.data_manager.df
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Data Distribution Before Cleaning', fontsize=16)
        
        # Calories distribution
        axes[0,0].hist(self.df['CALORIES'].dropna(), bins=50, alpha=0.7, edgecolor='black')
        axes[0,0].set_title('Calories Distribution')
        axes[0,0].set_xlabel('Calories')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].axvline(x=2000, color='red', linestyle='--', label='Upper limit')
        axes[0,0].legend()
        
        # Duration distribution
        axes[0,1].hist(self.df['DURATION_ACTUAL'].dropna(), bins=50, alpha=0.7, edgecolor='black')
        axes[0,1].set_title('Duration Distribution (hours)')
        axes[0,1].set_xlabel('Duration (hours)')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].axvline(x=5, color='red', linestyle='--', label='Upper limit')
        axes[0,1].legend()
        
        # Distance distribution
        axes[0,2].hist(self.df['DISTANCE_ACTUAL'].dropna(), bins=50, alpha=0.7, edgecolor='black')
        axes[0,2].set_title('Distance Distribution (meters)')
        axes[0,2].set_xlabel('Distance (meters)')
        axes[0,2].set_ylabel('Frequency')
        axes[0,2].axvline(x=100000, color='red', linestyle='--', label='Upper limit')
        axes[0,2].legend()
        
        # Weight distribution
        if 'WEIGHT' in self.df.columns:
            axes[1,0].hist(self.df['WEIGHT'].dropna(), bins=30, alpha=0.7, edgecolor='black')
            axes[1,0].set_title('Weight Distribution (kg)')
            axes[1,0].set_xlabel('Weight (kg)')
            axes[1,0].set_ylabel('Frequency')
        
        # Age distribution
        if 'AGE' in self.df.columns:
            axes[1,1].hist(self.df['AGE'].dropna(), bins=30, alpha=0.7, edgecolor='black')
            axes[1,1].set_title('Age Distribution')
            axes[1,1].set_xlabel('Age (years)')
            axes[1,1].set_ylabel('Frequency')
        
        # Heart rate distribution
        if 'HRAVG' in self.df.columns:
            axes[1,2].hist(self.df['HRAVG'].dropna(), bins=30, alpha=0.7, edgecolor='black')
            axes[1,2].set_title('Average Heart Rate Distribution')
            axes[1,2].set_xlabel('Heart Rate (bpm)')
            axes[1,2].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
        
        # Activity ranges analysis
        print("\n--- ACTIVITY RANGES BEFORE CLEANING ---")
        print(f"Calories range: {self.df['CALORIES'].min():.0f} - {self.df['CALORIES'].max():.0f}")
        print(f"Duration range: {self.df['DURATION_ACTUAL'].min():.3f} - {self.df['DURATION_ACTUAL'].max():.3f} hours")
        print(f"Distance range: {self.df['DISTANCE_ACTUAL'].min():.0f} - {self.df['DISTANCE_ACTUAL'].max():.0f} meters")
    
    def visualize_cleaned_data(self):
        """Visualize data after cleaning"""
        print("\nCreating post-cleaning visualizations...")
        
        # Update dataframe reference
        self.df = self.data_manager.df
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Data Distribution After Cleaning', fontsize=16)
        
        # Calories distribution
        axes[0,0].hist(self.df['CALORIES'], bins=50, alpha=0.7, edgecolor='black', color='green')
        axes[0,0].set_title('Calories Distribution (Cleaned)')
        axes[0,0].set_xlabel('Calories')
        axes[0,0].set_ylabel('Frequency')
        
        # Duration distribution
        axes[0,1].hist(self.df['DURATION_ACTUAL'], bins=50, alpha=0.7, edgecolor='black', color='green')
        axes[0,1].set_title('Duration Distribution (Cleaned)')
        axes[0,1].set_xlabel('Duration (hours)')
        axes[0,1].set_ylabel('Frequency')
        
        # Distance distribution
        axes[0,2].hist(self.df['DISTANCE_ACTUAL'], bins=50, alpha=0.7, edgecolor='black', color='green')
        axes[0,2].set_title('Distance Distribution (Cleaned)')
        axes[0,2].set_xlabel('Distance (meters)')
        axes[0,2].set_ylabel('Frequency')
        
        # Weight distribution
        if 'WEIGHT' in self.df.columns:
            axes[1,0].hist(self.df['WEIGHT'].dropna(), bins=30, alpha=0.7, edgecolor='black', color='green')
            axes[1,0].set_title('Weight Distribution (Cleaned)')
            axes[1,0].set_xlabel('Weight (kg)')
            axes[1,0].set_ylabel('Frequency')
        
        # Age distribution
        if 'AGE' in self.df.columns:
            axes[1,1].hist(self.df['AGE'].dropna(), bins=30, alpha=0.7, edgecolor='black', color='green')
            axes[1,1].set_title('Age Distribution (Cleaned)')
            axes[1,1].set_xlabel('Age (years)')
            axes[1,1].set_ylabel('Frequency')
        
        # Heart rate distribution
        if 'HRAVG' in self.df.columns:
            axes[1,2].hist(self.df['HRAVG'].dropna(), bins=30, alpha=0.7, edgecolor='black', color='green')
            axes[1,2].set_title('Average Heart Rate Distribution (Cleaned)')
            axes[1,2].set_xlabel('Heart Rate (bpm)')
            axes[1,2].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
        
        # Activity ranges after cleaning
        print("\n--- ACTIVITY RANGES AFTER CLEANING ---")
        print(f"Calories range: {self.df['CALORIES'].min():.0f} - {self.df['CALORIES'].max():.0f}")
        print(f"Duration range: {self.df['DURATION_ACTUAL'].min():.3f} - {self.df['DURATION_ACTUAL'].max():.3f} hours")
        print(f"Distance range: {self.df['DISTANCE_ACTUAL'].min():.0f} - {self.df['DISTANCE_ACTUAL'].max():.0f} meters")
    
    def update_data(self):
        """Update the dataframe reference after data changes"""
        self.df = self.data_manager.df


class CaloriePredictor:
    """Handles feature engineering, modeling, and predictions"""
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.df = data_manager.df
        self.model = None
        self.feature_columns = None
        
    def create_features(self):
        """Create derived features"""
        print("\nCreating features...")
        
        # Update dataframe reference
        self.df = self.data_manager.df
        
        # Basic derived features
        self.df['PACE'] = self.df['DURATION_ACTUAL'] / (self.df['DISTANCE_ACTUAL'] / 1000)  # hours per km
        self.df['SPEED'] = (self.df['DISTANCE_ACTUAL'] / 1000) / self.df['DURATION_ACTUAL']  # km per hour
        
        # Intensity ratio (HR_avg / HR_max) - common metric in exercise physiology
        # This ratio indicates exercise intensity relative to max capacity
        self.df['INTENSITY_RATIO'] = self.df['HRAVG'] / self.df['HRMAX']
        
        # Encode categorical variables
        if 'SEX' in self.df.columns:
            self.df['SEX_ENCODED'] = (self.df['SEX'] == 'M').astype(int)
        
        # Define feature columns
        self.feature_columns = [
            'DURATION_ACTUAL', 'DISTANCE_ACTUAL', 'HRMAX', 'HRAVG',
            'ELEVATIONAVG', 'ELEVATIONGAIN', 'TRAININGSTRESSSCOREACTUAL',
            'AGE', 'WEIGHT', 'SEX_ENCODED', 'PACE', 'SPEED', 'INTENSITY_RATIO'
        ]
        
        # Filter to only include available features
        self.feature_columns = [col for col in self.feature_columns if col in self.df.columns]
        
        print(f"Created {len(self.feature_columns)} features")
        return self.df
    
    def train_model(self):
        """Train the LightGBM model with proper imputation timing"""
        print("\nTraining model...")
        
        # Update dataframe reference
        self.df = self.data_manager.df
        
        # Prepare features and target
        X = self.df[self.feature_columns].copy()
        y = self.df['CALORIES']
        
        # Split data FIRST (before imputation to avoid data leakage)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Clean gender encoding (standardize F/f and M/m)
        if 'SEX' in X_train.columns:
            X_train['SEX'] = X_train['SEX'].str.upper()
            X_test['SEX'] = X_test['SEX'].str.upper()
        
        # Handle missing values using ONLY training data (imputation after split)
        print("Handling missing values using training data only...")
        
        # Calculate demographic-specific imputation values from training data
        demographic_imputation = {}
        
        if 'SEX' in X_train.columns:
            # Gender-specific imputation for weight
            if 'WEIGHT' in X_train.columns:
                weight_by_sex = X_train.groupby('SEX')['WEIGHT'].median()
                demographic_imputation['WEIGHT'] = weight_by_sex.to_dict()
                print(f"  Weight imputation by gender: {weight_by_sex.to_dict()}")
            
            # Gender-specific imputation for heart rate
            if 'HRMAX' in X_train.columns:
                hrmax_by_sex = X_train.groupby('SEX')['HRMAX'].median()
                demographic_imputation['HRMAX'] = hrmax_by_sex.to_dict()
                print(f"  HRMAX imputation by gender: {hrmax_by_sex.to_dict()}")
            
            if 'HRAVG' in X_train.columns:
                hravg_by_sex = X_train.groupby('SEX')['HRAVG'].median()
                demographic_imputation['HRAVG'] = hravg_by_sex.to_dict()
                print(f"  HRAVG imputation by gender: {hravg_by_sex.to_dict()}")
        
        # Apply imputation
        for col in X_train.columns:
            if X_train[col].isnull().sum() > 0:
                if col in demographic_imputation and 'SEX' in X_train.columns:
                    # Use demographic-specific imputation
                    for sex in ['F', 'M']:
                        if sex in demographic_imputation[col]:
                            # Impute for specific gender
                            sex_mask_train = (X_train['SEX'] == sex) & X_train[col].isnull()
                            sex_mask_test = (X_test['SEX'] == sex) & X_test[col].isnull()
                            
                            X_train.loc[sex_mask_train, col] = demographic_imputation[col][sex]
                            X_test.loc[sex_mask_test, col] = demographic_imputation[col][sex]
                    
                    # Fallback to global median for any remaining missing values
                    remaining_missing_train = X_train[col].isnull()
                    remaining_missing_test = X_test[col].isnull()
                    
                    if remaining_missing_train.any() or remaining_missing_test.any():
                        global_median = X_train[col].median()
                        X_train.loc[remaining_missing_train, col] = global_median
                        X_test.loc[remaining_missing_test, col] = global_median
                        print(f"  Applied global median fallback for {col}: {global_median:.1f}")
                else:
                    # Use global imputation for other features
                    if X_train[col].dtype in ['int64', 'float64']:
                        impute_value = X_train[col].median()
                    else:
                        impute_value = X_train[col].mode()[0]
                    
                    X_train[col] = X_train[col].fillna(impute_value)
                    X_test[col] = X_test[col].fillna(impute_value)
        
        # Train model
        self.model = lgb.LGBMRegressor(random_state=42, verbose=-1)
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
        cv_mae = -cv_scores.mean()
        cv_std = cv_scores.std()
        
        print(f"\nModel Performance (with demographic imputation):")
        print(f"  MAE: {mae:.1f} calories")
        print(f"  R²: {r2:.3f}")
        print(f"  CV MAE: {cv_mae:.1f} ± {cv_std:.1f} calories")
        
        return mae, r2, cv_mae
    
    def show_feature_importance(self):
        """Display feature importance"""
        if self.model is None:
            print("Model not trained yet")
            return
        
        importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        for i, row in importance.iterrows():
            print(f"  {row['feature']:<20}: {row['importance']:.3f}")
        
        # Create feature importance visualization
        plt.figure(figsize=(10, 6))
        plt.barh(importance['feature'], importance['importance'])
        plt.xlabel('Feature Importance')
        plt.title('LightGBM Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        
        return importance
    
    def save_model(self):
        """Save the trained model for production use"""
        if self.model is None:
            print("No model to save")
            return
        
        model_filename = 'calorie_prediction_model.pkl'
        
        # Save model and metadata
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'training_data_stats': {
                'total_records': len(self.df),
                'feature_count': len(self.feature_columns),
                'imputation_values': {}
            }
        }
        
        # Store imputation values for production use
        for col in self.feature_columns:
            if col in self.df.columns:
                if self.df[col].dtype in ['int64', 'float64']:
                    model_data['training_data_stats']['imputation_values'][col] = self.df[col].median()
                else:
                    model_data['training_data_stats']['imputation_values'][col] = self.df[col].mode()[0]
        
        with open(model_filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        return model_filename
    
    def create_prediction_function(self):
        """Create a function for making predictions on new data"""
        def predict_calories(duration, distance=None, hr_max=None, hr_avg=None, 
                           elevation_avg=None, elevation_gain=None, tss=None,
                           age=None, weight=None, sex=None):
            """
            Predict calories for a planned run
            
            Args:
                duration: Duration in hours (required)
                distance: Distance in meters (optional)
                hr_max: Maximum heart rate (optional)
                hr_avg: Average heart rate (optional)
                elevation_avg: Average elevation in meters (optional)
                elevation_gain: Elevation gain in meters (optional)
                tss: Training stress score (optional)
                age: Age in years (optional)
                weight: Weight in kg (optional)
                sex: 'M' or 'F' (optional)
            
            Returns:
                Predicted calories
            """
            # Create input dataframe
            input_data = pd.DataFrame({
                'DURATION_ACTUAL': [duration],
                'DISTANCE_ACTUAL': [distance] if distance is not None else [np.nan],
                'HRMAX': [hr_max] if hr_max is not None else [np.nan],
                'HRAVG': [hr_avg] if hr_avg is not None else [np.nan],
                'ELEVATIONAVG': [elevation_avg] if elevation_avg is not None else [np.nan],
                'ELEVATIONGAIN': [elevation_gain] if elevation_gain is not None else [np.nan],
                'TRAININGSTRESSSCOREACTUAL': [tss] if tss is not None else [np.nan],
                'AGE': [age] if age is not None else [np.nan],
                'WEIGHT': [weight] if weight is not None else [np.nan],
                'SEX_ENCODED': [1 if sex == 'M' else 0] if sex is not None else [np.nan]
            })
            
            # Create derived features
            if distance is not None and distance > 0:
                input_data['PACE'] = duration / (distance / 1000)
                input_data['SPEED'] = (distance / 1000) / duration
            else:
                input_data['PACE'] = np.nan
                input_data['SPEED'] = np.nan
            
            if hr_avg is not None and hr_max is not None and hr_max > 0:
                input_data['INTENSITY_RATIO'] = hr_avg / hr_max
            else:
                input_data['INTENSITY_RATIO'] = np.nan
            
            # Handle missing values using training data medians/modes
            for col in input_data.columns:
                if input_data[col].isnull().any():
                    if col in self.df.columns:
                        if self.df[col].dtype in ['int64', 'float64']:
                            input_data[col] = input_data[col].fillna(self.df[col].median())
                        else:
                            input_data[col] = input_data[col].fillna(self.df[col].mode()[0])
            
            # Ensure all required features are present
            for col in self.feature_columns:
                if col not in input_data.columns:
                    input_data[col] = 0  # Default value
            
            # Make prediction
            prediction = self.model.predict(input_data[self.feature_columns])[0]
            return max(0, prediction)  # Ensure non-negative
        
        return predict_calories
    
    def update_data(self):
        """Update the dataframe reference after data changes"""
        self.df = self.data_manager.df


# Convenience function for easy usage
def run_complete_analysis(filepath, sample_size=None):
    """
    Run the complete analysis pipeline using the modular classes
    
    Args:
        filepath: Path to the CSV file
        sample_size: Number of records to sample (optional)
    
    Returns:
        Dictionary with results and components
    """
    print("=== MODULAR CALORIE PREDICTION ANALYSIS ===")
    print("=" * 50)
    
    # Initialize components
    data_manager = DataManager(filepath, sample_size)
    visualizer = DataVisualizer(data_manager)
    predictor = CaloriePredictor(data_manager)
    
    # Load and analyze data
    data_manager.load_data()
    data_manager.analyze_missing_data()
    data_manager.analyze_athlete_distribution()
    
    # Visualize raw data
    visualizer.visualize_raw_data()
    
    # Clean data
    data_manager.clean_data()
    visualizer.update_data()
    predictor.update_data()
    
    # Visualize cleaned data
    visualizer.visualize_cleaned_data()
    
    # Create features and train model
    predictor.create_features()
    mae, r2, cv_mae = predictor.train_model()
    
    # Show results
    predictor.show_feature_importance()
    
    # Save model
    model_file = predictor.save_model()
    
    # Create prediction function and examples
    predict_func = predictor.create_prediction_function()
    
    print("\nExample Predictions:")
    print(f"  30 min run: {predict_func(0.5):.0f} calories")
    print(f"  1 hour run: {predict_func(1.0):.0f} calories")
    print(f"  2 hour run: {predict_func(2.0):.0f} calories")
    
    print(f"\nAnalysis complete! Model saved to {model_file}")
    
    return {
        'data_manager': data_manager,
        'visualizer': visualizer,
        'predictor': predictor,
        'mae': mae,
        'r2': r2,
        'cv_mae': cv_mae,
        'predict_function': predict_func,
        'model_file': model_file
    }


# Example usage for Jupyter notebooks
if __name__ == "__main__":
    # Example of how to use the modular classes
    results = run_complete_analysis('workout_data.csv', sample_size=50000) 