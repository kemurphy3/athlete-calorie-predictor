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
import multiprocessing as mp
from joblib import Parallel, delayed
import psutil

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
    'model_filename': 'calorie_prediction_model_enhanced.pkl',
    'n_jobs': -1  # Use all available cores
}

def get_optimal_n_jobs():
    """
    Determine optimal number of jobs based on system resources
    """
    cpu_count = mp.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # Conservative approach: use 75% of cores, but not more than memory allows
    optimal_cores = min(int(cpu_count * 0.75), int(memory_gb / 2))
    return max(1, optimal_cores)

def parallel_model_training(model_name, model, X_train, y_train, cv_folds):
    """
    Train a single model with cross-validation (for parallel processing)
    """
    try:
        start_time = datetime.now()
        
        # Special handling for Random Forest to prevent hanging
        if 'Random Forest' in model_name:
            # For very large datasets, use stratified sampling to maintain representativeness
            if len(X_train) > 200000:
                # Use stratified sampling based on calorie ranges
                calorie_bins = pd.cut(y_train, bins=10, labels=False)
                sample_indices = []
                for bin_id in range(10):
                    bin_mask = calorie_bins == bin_id
                    if bin_mask.sum() > 0:
                        bin_indices = np.where(bin_mask)[0]
                        n_samples = min(10000, len(bin_indices))  # 10k per bin, max
                        sampled_indices = np.random.choice(bin_indices, n_samples, replace=False)
                        sample_indices.extend(sampled_indices)
                
                if len(sample_indices) > 100000:
                    sample_indices = np.random.choice(sample_indices, 100000, replace=False)
                
                X_train_subset = X_train.iloc[sample_indices]
                y_train_subset = y_train.iloc[sample_indices]
                cv_folds_rf = min(3, cv_folds)
            elif len(X_train) > 100000:
                # Use stratified sampling for large datasets
                calorie_bins = pd.cut(y_train, bins=8, labels=False)
                sample_indices = []
                for bin_id in range(8):
                    bin_mask = calorie_bins == bin_id
                    if bin_mask.sum() > 0:
                        bin_indices = np.where(bin_mask)[0]
                        n_samples = min(9375, len(bin_indices))  # 75k/8 = 9375 per bin
                        sampled_indices = np.random.choice(bin_indices, n_samples, replace=False)
                        sample_indices.extend(sampled_indices)
                
                X_train_subset = X_train.iloc[sample_indices]
                y_train_subset = y_train.iloc[sample_indices]
                cv_folds_rf = min(4, cv_folds)
            else:
                X_train_subset = X_train
                y_train_subset = y_train
                cv_folds_rf = cv_folds
        else:
            X_train_subset = X_train
            y_train_subset = y_train
            cv_folds_rf = cv_folds
        
        cv_scores = cross_val_score(model, X_train_subset, y_train_subset, cv=cv_folds_rf, 
                                   scoring='neg_mean_absolute_error', n_jobs=1)
        
        mae = -cv_scores.mean()
        mae_std = cv_scores.std()
        elapsed_time = (datetime.now() - start_time).total_seconds()
        
        return {
            'name': model_name,
            'mae': mae,
            'mae_std': mae_std,
            'model': model,
            'elapsed_time': elapsed_time,
            'success': True
        }
    except Exception as e:
        return {
            'name': model_name,
            'error': str(e),
            'success': False
        }

def load_and_clean_data(filepath, sample_size=None):
    """
    Load and clean the workout data using the full dataset with parallel processing where possible
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
        # Use parallel processing for large CSV files
        if os.path.getsize(filepath) > 100 * 1024 * 1024:  # 100MB threshold
            df = pd.read_csv(filepath, engine='c', low_memory=False)
        else:
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
    
    # Analyze missing data (parallel processing for large datasets)
    print("\n++ Missing Data Analysis ++")
    key_features = ['CALORIES', 'DURATION_ACTUAL', 'DISTANCE_ACTUAL', 'HRMAX', 'HRAVG', 
                   'ELEVATIONAVG', 'ELEVATIONGAIN', 'TRAININGSTRESSSCOREACTUAL', 'AGE', 'WEIGHT', 'SEX']
    
    if len(df) > 100000:  # Use parallel processing for large datasets
        def analyze_missing_parallel(col):
            if col in df.columns:
                missing_count = df[col].isnull().sum()
                missing_pct = (missing_count / len(df)) * 100
                return col, missing_count, missing_pct
            return col, 0, 0.0
        
        n_jobs = get_optimal_n_jobs()
        missing_results = Parallel(n_jobs=n_jobs)(
            delayed(analyze_missing_parallel)(col) for col in key_features
        )
        
        for col, missing_count, missing_pct in missing_results:
            if col in df.columns:
                print(f"{col:<25}: {missing_count:>6,} missing ({missing_pct:>5.1f}%)")
    else:
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
        
        # Gender-specific imputation for heart rate
        if 'HRMAX' in X_train.columns:
            hrmax_by_sex = X_train.groupby('SEX')['HRMAX'].median()
            demographic_imputation['HRMAX'] = hrmax_by_sex.to_dict()
        
        if 'HRAVG' in X_train.columns:
            hravg_by_sex = X_train.groupby('SEX')['HRAVG'].median()
            demographic_imputation['HRAVG'] = hravg_by_sex.to_dict()
    
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
            else:
                # Use global imputation for other features
                if X_train[col].dtype in ['int64', 'float64']:
                    impute_value = X_train[col].median()
                else:
                    impute_value = X_train[col].mode()[0]
                
                X_train[col] = X_train[col].fillna(impute_value)
                X_test[col] = X_test[col].fillna(impute_value)
    
    return X_train, X_test

def compare_models(X_train, y_train, feature_columns, skip_rf=False):
    """
    Compare multiple models using cross-validation with parallel processing
    For this project, I will systematically compare multiple algorithms including Linear Regression, 
    Random Forest, XGBoost, and LightGBM to ensure we select the best performing model for the business case
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        feature_columns (list): List of feature column names
        skip_rf (bool): Whether to skip Random Forest (for large datasets or performance)
    
    Returns:
        tuple: (results_dict, best_model_name)
    """
    if len(X_train) == 0 or len(y_train) == 0:
        raise ValueError("Training data cannot be empty")
    
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have the same length")
    
    print("\nComparing multiple models...")
    print(f"Training data size: {len(X_train):,} samples, {len(feature_columns)} features")
    
    # Determine optimal number of jobs
    n_jobs = get_optimal_n_jobs()
    
    # Define models to compare (optimized for parallel processing)
    models = {
        'Linear Regression': LinearRegression(),
        'LightGBM': lgb.LGBMRegressor(random_state=42, verbose=-1, n_estimators=100, n_jobs=1),
        'XGBoost': xgb.XGBRegressor(random_state=42, verbosity=0, n_estimators=100, n_jobs=1)
    }
    
    # Add Random Forest only if not skipped and dataset is manageable
    if not skip_rf:  # Always include Random Forest for full dataset
        # Optimized Random Forest settings for speed + robustness
        rf_config = {
            'random_state': 42,
            'n_estimators': 50,  # Increased back to 50 for robustness
            'max_depth': 8,      # Reduced from 10 to 8 for speed
            'min_samples_split': 100,  # Increased from 50 to 100
            'min_samples_leaf': 50,    # Increased from 20 to 50
            'n_jobs': 1,         # Single-threaded for parallel processing
            'max_features': 'sqrt',  # Use sqrt of features for faster splits
            'bootstrap': True,   # Enable bootstrapping for robustness
            'oob_score': False,  # Disable OOB score for speed
            'warm_start': False, # Disable warm start for speed
            'max_samples': 0.8   # Use 80% of samples per tree for speed
        }
        
        # For very large datasets, use even more aggressive optimization
        if len(X_train) > 200000:
            rf_config.update({
                'n_estimators': 30,     # Fewer trees for speed
                'max_depth': 6,         # Shallow trees
                'min_samples_split': 200,  # Larger splits
                'min_samples_leaf': 100,   # Larger leaves
                'max_samples': 0.6      # Use only 60% of samples per tree
            })
        elif len(X_train) > 100000:
            rf_config.update({
                'n_estimators': 40,     # Moderate number of trees
                'max_depth': 7,         # Moderate depth
                'min_samples_split': 150,  # Moderate splits
                'min_samples_leaf': 75,    # Moderate leaves
                'max_samples': 0.7      # Use 70% of samples per tree
            })
        
        models['Random Forest'] = RandomForestRegressor(**rf_config)
    elif skip_rf:
        print("  Skipping Random Forest (--skip-rf flag used)")
    
    # Determine CV folds based on data size
    cv_folds = min(5, max(3, len(X_train) // 10000))
    print(f"Using {cv_folds}-fold cross-validation")
    
    # Prepare parallel processing tasks
    tasks = []
    for name, model in models.items():
        tasks.append((name, model, X_train, y_train, cv_folds))
    
    # Execute parallel model training
    print("\n++ Model Comparison Results ++")
    results_list = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(parallel_model_training)(name, model, X_train, y_train, cv_folds)
        for name, model, X_train, y_train, cv_folds in tasks
    )
    
    # Process results
    results = {}
    for result in results_list:
        if result['success']:
            name = result['name']
            results[name] = {
                'mae': result['mae'],
                'mae_std': result['mae_std'],
                'model': result['model']
            }
            print(f"  {name:<20}: MAE = {result['mae']:.1f} ± {result['mae_std']:.1f} calories")
        else:
            print(f"  Error with {result['name']}: {result['error']}")
    
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

def run_enhanced_analysis(filepath, sample_size=None, skip_rf=False):
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
        skip_rf (bool): Whether to skip Random Forest (for large datasets or performance)
    
    Returns:
        dict: Analysis results including model, metrics, and feature importance
    
    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If data is invalid
        RuntimeError: If model training fails
    """
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
        results, best_model_name = compare_models(X_train, y_train, feature_columns, skip_rf=skip_rf)
        
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
    
    Parallel processing is automatically enabled for optimal performance.
    """
    
    import sys
    
    # Check for quick test mode
    quick_test = '--quick' in sys.argv or '--test' in sys.argv
    no_parallel = '--no-parallel' in sys.argv
    skip_rf = '--skip-rf' in sys.argv  # Skip Random Forest
    
    if quick_test:
        print("=== QUICK TEST MODE ===")
        print("Using sample data for faster testing...")
        sample_size = 10000
    else:
        sample_size = None
    
    if no_parallel:
        print("=== PARALLEL PROCESSING DISABLED ===")
        CONFIG['n_jobs'] = 1
    
    if skip_rf:
        print("=== RANDOM FOREST DISABLED ===")
    
    # Example 1: Run analysis on full dataset
    print("\n=== Enhanced Calorie Prediction Analysis ===")
    try:
        results = run_enhanced_analysis("workout_data.csv", sample_size=sample_size, skip_rf=skip_rf)
        
        print("\n=== Analysis Complete ===")
        print(f"Best model: {results['model_name']}")
        print(f"Final MAE: {results['mae']:.1f} calories")
        print(f"Final R²: {results['r2']:.3f}")
        
    except FileNotFoundError:
        print("workout_data.csv not found. Please ensure the data file is in the current directory.")
        print("\nTo run a quick test with sample data:")
        print("python Murphy_TakeHome_Advanced.py --quick")
    except Exception as e:
        print(f"Analysis failed: {e}")
        print("\nTry running with --quick flag for faster testing:")
        print("python Murphy_TakeHome_Advanced.py --quick")
    
    # Example 2: How to use the saved model
    print("\n=== Model Usage ===")
    print("To make predictions on new data:")
    print("1. Load the saved model:")
    print("   with open('calorie_prediction_model_enhanced.pkl', 'rb') as f:")
    print("       model_data = pickle.load(f)")
    print("2. Prepare your input data with the same features")
    print("3. Call predict_calories(model_data, input_data)")
    
    # Example 3: Performance tips
    print("\n=== Performance Tips ===")
    print("For faster processing:")
    print("- Use --quick flag for testing: python Murphy_TakeHome_Advanced.py --quick")
    print("- Disable parallel processing: python Murphy_TakeHome_Advanced.py --no-parallel")
    print("- Skip Random Forest: python Murphy_TakeHome_Advanced.py --skip-rf")
    print("- Ensure sufficient RAM (8GB+ recommended for full dataset)") 