# Enhanced Calorie Prediction Analysis
# Addressing feedback: Multiple models, cross-validation, full dataset, business impact
# Based on original work by Kate Murphy
#
# BUSINESS CONTEXT:
# This model predicts calorie burn for planned runs to help athletes optimize their training.
# Target users: Recreational runners who want to estimate calorie expenditure before workouts.
# Success metric: Mean Absolute Error (MAE) - how close predictions are to actual calorie burn.
# Business value: Helps users plan nutrition and training intensity more effectively.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
import pickle
import os
import warnings
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('calorie_prediction.log'),
        logging.StreamHandler()
    ]
)
warnings.filterwarnings('ignore')

# Configuration
CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5,
    'model_filename': 'calorie_prediction_model_enhanced.pkl'
}

def load_and_clean_data(filepath, sample_size=None):
    """
    Load and clean the workout data using the full dataset
    Important to understand the scale and distribution of missing values as this will inform the need to impute
    
    Args:
        filepath (str): Path to the CSV file
        sample_size (int, optional): Number of records to sample. If None, uses full dataset.
    
    Returns:
        pd.DataFrame: Cleaned dataset
    
    Raises:
        FileNotFoundError: If filepath doesn't exist
        ValueError: If sample_size is negative
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    if sample_size is not None and sample_size < 0:
        raise ValueError("sample_size must be positive")
    
    print("Loading data...")
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        raise RuntimeError(f"Failed to load data: {e}")
    
    df.columns = df.columns.str.upper()
    
    if sample_size:
        # Shuffle the data first to ensure random sampling
        # Set the random_state for reproducibility
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        df = df.sample(n=min(sample_size, len(df)), random_state=42).reset_index(drop=True)
        print(f"Randomly sampled {len(df):,} records")
    else:
        print(f"Using all {len(df):,} records")
    
    # Analyze missing data
    print("\n++ Missing Data Analysis ++")
    key_features = ['CALORIES', 'DURATION_ACTUAL', 'DISTANCE_ACTUAL', 'HRMAX', 'HRAVG', 
                   'ELEVATIONAVG', 'ELEVATIONGAIN', 'TRAININGSTRESSSCOREACTUAL', 'AGE', 'WEIGHT', 'SEX']
    
    for col in key_features:
        if col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100
            print(f"{col:<25}: {missing_count:>6,} missing ({missing_pct:>5.1f}%)")
    
    # Clean data by removing invalid and outlier data
    print("\nCleaning data...")
    initial_rows = len(df)
    
    # Identify extreme values or typos before removal
    # Important to clean the data because some of the values in the current dataset are physically impossible 
    # or they are so extreme they'd skew the data
    extreme_calories = (df['CALORIES'] < 5) | (df['CALORIES'] > 2000) | (df['CALORIES'].isnull())
    extreme_duration = (df['DURATION_ACTUAL'] < 0.0167) | (df['DURATION_ACTUAL'] > 5)
    extreme_distance = (df['DISTANCE_ACTUAL'] < 100) | (df['DISTANCE_ACTUAL'] > 100000)
    extreme_hravg = (df['HRAVG'] < 30) | (df['HRAVG'] > 250)
    extreme_hrmax = (df['HRMAX'] < 30) | (df['HRMAX'] > 250)
    extreme_age = (df['AGE'] < 18) | (df['AGE'] > 100)
    extreme_weight = (df['WEIGHT'] < 20) | (df['WEIGHT'] > 200)
    
    # Count the number of extreme values so a total can be presented to the user
    calories_removed = extreme_calories.sum()
    duration_removed = extreme_duration.sum()
    distance_removed = extreme_distance.sum()
    hravg_removed = extreme_hravg.sum()
    hrmax_removed = extreme_hrmax.sum()
    age_removed = extreme_age.sum()
    weight_removed = extreme_weight.sum()
    
    # Remove invalid records by preserving all the data that doesn't violate an acceptable limit
    # This nuance allows for null values to remain which helps with overall data retention
    df = df[~(extreme_calories | extreme_duration | extreme_distance | extreme_hravg | extreme_hrmax | extreme_age | extreme_weight)]
    
    final_rows = len(df)
    total_removed = initial_rows - final_rows
    
    # Deliver a summarized output of invalid data and how many row remain to quantify data retention
    print(f"\n++ Summary of Cleaned Dataset ++")
    print(f"Records flagged for invalid calories (< 5 or > 2000): {calories_removed:,}")
    print(f"Records flagged for invalid duration (< 1 minute or > 5 hours): {duration_removed:,}")
    print(f"Records flagged for invalid distance (< 100 m or > 100 km): {distance_removed:,}")
    print(f"Records flagged for invalid average heart rate (< 30 or > 250): {hravg_removed:,}")
    print(f"Records flagged for invalid heart rate max (< 30 or > 250): {hrmax_removed:,}")
    print(f"Records flagged for invalid age (< 18 or > 100): {age_removed:,}")
    print(f"Records flagged for invalid weight (< 20 kgs or > 200 kgs): {weight_removed:,}")
    print(f"Total rows removed: {total_removed:,}")
    print(f"Final rows after cleaning: {final_rows:,}")
    print(f"Data retention: {((final_rows/initial_rows)*100):.1f}%")
    
    return df

def analyze_athlete_distribution(df):
    """
    Determine the distribution of activities per athlete
    For this dataset, most of the athletes are unique so there's no benefit in performing athlete level statistics
    """
    if 'ATHLETE_ID' not in df.columns:
        print("No ATHLETE_ID column found")
        return
    
    print("\n++ Activities by Athlete ++")
    
    athlete_counts = df['ATHLETE_ID'].value_counts()  # calculates how many unique athlete ID numbers there are
    total_athletes = len(athlete_counts)
    total_activities = len(df)  # each row is an activity
    
    # Prints the counts of activities and athletes and their relationship 
    print(f"Total athletes: {total_athletes:,}")
    print(f"Total activities: {total_activities:,}")
    print(f"Average activities per athlete: {total_activities/total_athletes:.1f}")
    
    # Activity count distribution, which will inform the type of analysis done
    activity_ranges = [
        (1, 5, "1-5 activities"),
        (6, 15, "6-15 activities"),
        (16, 30, "16-30 activities"),
        (31, 50, "31-50 activities"),
        (51, float('inf'), "50+ activities")
    ]
    
    # Had Cursor determine how to assign and display activity counts to each range
    print("\n++ Athletes by Activity Count ++")
    for min_act, max_act, label in activity_ranges:
        if max_act == float('inf'):
            count = (athlete_counts >= min_act).sum()
        else:
            count = ((athlete_counts >= min_act) & (athlete_counts <= max_act)).sum()
        percentage = (count / total_athletes) * 100
        print(f"  {label:<15}: {count:>6,} athletes ({percentage:>5.1f}%)")

def create_features(df):
    """
    Create derived features for the model
    This class uses existing columns to derive additional useful features since sometimes ratios can be more influential than their separate parts
    """
    print("\nCreating features...")
    
    # Basic derived features
    df['PACE'] = df['DURATION_ACTUAL'] / (df['DISTANCE_ACTUAL'] / 1000)  # hours per km
    df['SPEED'] = (df['DISTANCE_ACTUAL'] / 1000) / df['DURATION_ACTUAL']  # km per hour
    df['INTENSITY_RATIO'] = df['HRAVG'] / df['HRMAX']  # exercise intensity relative to max capacity
    
    # Encode categorical variables
    if 'SEX' in df.columns:
        df['SEX_ENCODED'] = (df['SEX'] == 'M').astype(int)
    
    feature_columns = [
        'DURATION_ACTUAL', 'DISTANCE_ACTUAL', 'HRMAX', 'HRAVG',
        'ELEVATIONAVG', 'ELEVATIONGAIN', 'TRAININGSTRESSSCOREACTUAL',
        'AGE', 'WEIGHT', 'SEX_ENCODED', 'PACE', 'SPEED', 'INTENSITY_RATIO'
    ]
    
    # Filter to only include available features
    feature_columns = [col for col in feature_columns if col in df.columns]
    print(f"Created {len(feature_columns)} features")
    
    return df, feature_columns

def handle_missing_values(X_train, X_test):
    """
    Handle missing values using demographic-specific imputation
    The training dataset then used demographic features to inform imputation on other key features
    """
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
    
    return X_train, X_test

def compare_models(X_train, y_train, feature_columns):
    """
    Compare multiple models using cross-validation
    For this project, I will systematically compare multiple algorithms including Linear Regression, 
    Random Forest, XGBoost, and LightGBM to ensure we select the best performing model for the business case
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        feature_columns (list): List of feature column names
    
    Returns:
        tuple: (results_dict, best_model_name)
    """
    if len(X_train) == 0 or len(y_train) == 0:
        raise ValueError("Training data cannot be empty")
    
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have the same length")
    
    print("\nComparing multiple models...")
    
    # Define models to compare
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100),
        'LightGBM': lgb.LGBMRegressor(random_state=42, verbose=-1),
        'XGBoost': xgb.XGBRegressor(random_state=42, verbosity=0)
    }
    
    # Compare models using cross-validation
    results = {}
    print("\n++ Model Comparison Results ++")
    for name, model in models.items():
        try:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
            mae = -cv_scores.mean()
            mae_std = cv_scores.std()
            results[name] = {'mae': mae, 'mae_std': mae_std, 'model': model}
            print(f"{name:<20}: MAE = {mae:.1f} ± {mae_std:.1f} calories")
        except Exception as e:
            print(f"Error with {name}: {e}")
            continue
    
    if not results:
        raise RuntimeError("No models were successfully trained")
    
    # Find best model
    best_model_name = min(results.keys(), key=lambda x: results[x]['mae'])
    print(f"\nBest model: {best_model_name}")
    
    return results, best_model_name

def analyze_business_impact(y_true, y_pred):
    """
    Analyze prediction accuracy by calorie ranges to understand business impact
    This would show up in discussion around the correct measurement to optimize for
    """
    print("\n++ Business Impact Analysis ++")
    ranges = [(0, 300), (300, 600), (600, 1000), (1000, 2000)]
    
    for low, high in ranges:
        mask = (y_true >= low) & (y_true < high)
        if mask.sum() > 0:
            mae = mean_absolute_error(y_true[mask], y_pred[mask])
            pct_error = (mae / y_true[mask].mean()) * 100 if y_true[mask].mean() > 0 else 0
            print(f"Calories {low}-{high}: MAE = {mae:.1f} ({pct_error:.1f}% error, {mask.sum():,} samples)")

def show_feature_importance(model, feature_columns, model_name):
    """
    Show feature importance for the selected model
    It also has a built in feature importance function that is useful when determining which data is influential
    """
    print(f"\nFeature Importance ({model_name}):")
    
    # Handle different model types
    if hasattr(model, 'feature_importances_'):
        importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
    elif hasattr(model, 'coef_'):
        importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': np.abs(model.coef_)
        }).sort_values('importance', ascending=False)
    else:
        print("Model doesn't support feature importance")
        return
    
    for i, row in importance.iterrows():
        print(f"  {row['feature']:<20}: {row['importance']:.3f}")
    
    return importance

def run_enhanced_analysis(filepath, sample_size=None):
    """
    Run the complete enhanced analysis
    
    This enhanced version addresses all the feedback points while maintaining the same structure, tone, and approach.
    
    BUSINESS CONTEXT:
    - Target: Recreational runners planning workouts
    - Problem: Need to estimate calorie burn before running
    - Solution: ML model that predicts calories based on workout parameters
    - Success: Low MAE across all calorie ranges (especially 300-1000 range where most users fall)
    
    TECHNICAL APPROACH:
    - Multiple model comparison (Linear, Random Forest, LightGBM, XGBoost)
    - 5-fold cross-validation for robust evaluation
    - Full dataset usage (no subsampling)
    - Business impact analysis by calorie ranges
    - Proper data leakage prevention
    
    Args:
        filepath (str): Path to the workout data CSV file
        sample_size (int, optional): Number of records to sample. If None, uses full dataset.
    
    Returns:
        dict: Analysis results including model, metrics, and feature importance
    
    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If data is invalid
        RuntimeError: If model training fails
    """
    print("=== Enhanced Calorie Prediction Analysis ===")
    print("Addressing feedback: Multiple models, cross-validation, full dataset, business impact\n")
    
    try:
        # Load and clean data
        df = load_and_clean_data(filepath, sample_size)
        
        # Analyze athlete distribution
        analyze_athlete_distribution(df)
        
        # Create features
        df, feature_columns = create_features(df)
        
        # Prepare data for modeling
        X = df[feature_columns].copy()
        y = df['CALORIES']
        
        if len(X) == 0:
            raise ValueError("No valid data remaining after cleaning")
        
        # Split data first, before imputation to avoid data leakage
        # Doing this split prior to data manipulation is crucial in preventing data leakage
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Handle missing values
        X_train, X_test = handle_missing_values(X_train, X_test)
        
        # Compare multiple models
        results, best_model_name = compare_models(X_train, y_train, feature_columns)
        
        # Train best model on full training set
        best_model = results[best_model_name]['model']
        best_model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = best_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\nFinal Model Performance ({best_model_name}):")
        print(f"  MAE: {mae:.1f} calories")
        print(f"  R²: {r2:.3f}")
        
        # Analyze business impact
        analyze_business_impact(y_test, y_pred)
        
        # Show feature importance
        importance = show_feature_importance(best_model, feature_columns, best_model_name)
        
        # Save model
        model_filename = 'calorie_prediction_model_enhanced.pkl'
        model_data = {
            'model': best_model,
            'model_name': best_model_name,
            'feature_columns': feature_columns,
            'training_data_stats': {
                'total_records': len(df),
                'feature_count': len(feature_columns),
                'mae': mae,
                'r2': r2
            }
        }
        
        with open(model_filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\nModel saved to {model_filename}")
        
        return {
            'model': best_model,
            'model_name': best_model_name,
            'feature_columns': feature_columns,
            'mae': mae,
            'r2': r2,
            'importance': importance,
            'results': results
        }
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        raise

def predict_calories(model_data, input_data):
    """
    Predict calories for new data using the trained model
    Cursor wrote the prediction function, which applied the developed model
    
    Args:
        model_data (dict): Model data loaded from pickle file
        input_data (pd.DataFrame): Input data with required features
    
    Returns:
        np.array: Predicted calorie values
    
    Raises:
        ValueError: If required features are missing
    """
    if not isinstance(model_data, dict) or 'model' not in model_data:
        raise ValueError("model_data must be a dictionary with 'model' key")
    
    model = model_data['model']
    feature_columns = model_data['feature_columns']
    
    # Ensure input data has the same features
    missing_features = [col for col in feature_columns if col not in input_data.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    input_features = input_data[feature_columns].copy()
    
    # Handle missing values if any
    for col in feature_columns:
        if input_features[col].isnull().any():
            input_features[col] = input_features[col].fillna(input_features[col].median())
    
    # Make prediction
    prediction = model.predict(input_features)
    return prediction

if __name__ == "__main__":
    """
    Example usage of the enhanced calorie prediction analysis.
    
    This script demonstrates:
    1. Loading and cleaning workout data
    2. Comparing multiple ML models using cross-validation
    3. Training the best model on the full dataset
    4. Analyzing business impact across calorie ranges
    5. Saving the model for future use
    """
    
    # Example 1: Run analysis on full dataset
    print("=== Example 1: Full Dataset Analysis ===")
    try:
        results = run_enhanced_analysis("workout_data.csv")
        
        print("\n=== Analysis Complete ===")
        print(f"Best model: {results['model_name']}")
        print(f"Final MAE: {results['mae']:.1f} calories")
        print(f"Final R²: {results['r2']:.3f}")
        
    except FileNotFoundError:
        print("workout_data.csv not found. Please ensure the data file is in the current directory.")
    except Exception as e:
        print(f"Analysis failed: {e}")
    
    # Example 2: Run analysis on sample data (for testing)
    print("\n=== Example 2: Sample Data Analysis ===")
    try:
        sample_results = run_enhanced_analysis("workout_data.csv", sample_size=10000)
        print(f"Sample analysis completed with {sample_results['mae']:.1f} MAE")
    except Exception as e:
        print(f"Sample analysis failed: {e}")
    
    # Example 3: How to use the saved model
    print("\n=== Example 3: Model Usage ===")
    print("To make predictions on new data:")
    print("1. Load the saved model:")
    print("   with open('calorie_prediction_model_enhanced.pkl', 'rb') as f:")
    print("       model_data = pickle.load(f)")
    print("2. Prepare your input data with the same features")
    print("3. Call predict_calories(model_data, input_data)")
    
    # Example 4: Business context summary
    print("\n=== Business Context Summary ===")
    print("This model helps recreational runners estimate calorie burn before workouts.")
    print("Key benefits:")
    print("- Plan nutrition more effectively")
    print("- Optimize training intensity")
    print("- Track progress over time")
    print("- Make informed workout decisions") 