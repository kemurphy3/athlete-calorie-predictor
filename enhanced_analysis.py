# Enhanced Calorie Prediction Analysis
# Addressing feedback: Multiple models, cross-validation, full dataset, business impact

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
import pickle

def load_and_clean_data(filepath):
    """
    Load and clean the workout data using the full dataset
    """
    print("Loading data...")
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.upper()
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
    
    # Clean data
    print("\nCleaning data...")
    initial_rows = len(df)
    
    # Identify extreme values
    extreme_calories = (df['CALORIES'] < 5) | (df['CALORIES'] > 2000) | (df['CALORIES'].isnull())
    extreme_duration = (df['DURATION_ACTUAL'] < 0.0167) | (df['DURATION_ACTUAL'] > 5)
    extreme_distance = (df['DISTANCE_ACTUAL'] < 100) | (df['DISTANCE_ACTUAL'] > 100000)
    extreme_hravg = (df['HRAVG'] < 30) | (df['HRAVG'] > 250)
    extreme_hrmax = (df['HRMAX'] < 30) | (df['HRMAX'] > 250)
    extreme_age = (df['AGE'] < 18) | (df['AGE'] > 100)
    extreme_weight = (df['WEIGHT'] < 20) | (df['WEIGHT'] > 200)
    
    # Remove invalid records
    df = df[~(extreme_calories | extreme_duration | extreme_distance | extreme_hravg | extreme_hrmax | extreme_age | extreme_weight)]
    
    final_rows = len(df)
    print(f"\n++ Summary of Cleaned Dataset ++")
    print(f"Total rows removed: {initial_rows - final_rows:,}")
    print(f"Final rows after cleaning: {final_rows:,}")
    print(f"Data retention: {((final_rows/initial_rows)*100):.1f}%")
    
    return df

def create_features(df):
    """
    Create derived features for the model
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
    """
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
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
        mae = -cv_scores.mean()
        mae_std = cv_scores.std()
        results[name] = {'mae': mae, 'mae_std': mae_std, 'model': model}
        print(f"{name:<20}: MAE = {mae:.1f} ± {mae_std:.1f} calories")
    
    # Find best model
    best_model_name = min(results.keys(), key=lambda x: results[x]['mae'])
    print(f"\nBest model: {best_model_name}")
    
    return results, best_model_name

def analyze_business_impact(y_true, y_pred):
    """
    Analyze prediction accuracy by calorie ranges to understand business impact
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

def run_enhanced_analysis(filepath):
    """
    Run the complete enhanced analysis
    """
    print("=== Enhanced Calorie Prediction Analysis ===")
    print("Addressing feedback: Multiple models, cross-validation, full dataset, business impact\n")
    
    # Load and clean data
    df = load_and_clean_data(filepath)
    
    # Create features
    df, feature_columns = create_features(df)
    
    # Prepare data for modeling
    X = df[feature_columns].copy()
    y = df['CALORIES']
    
    # Split data first, before imputation to avoid data leakage
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
            'feature_count': len(feature_columns)
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

if __name__ == "__main__":
    # Run the enhanced analysis
    results = run_enhanced_analysis("workout_data.csv")
    
    print("\n=== Analysis Complete ===")
    print(f"Best model: {results['model_name']}")
    print(f"Final MAE: {results['mae']:.1f} calories")
    print(f"Final R²: {results['r2']:.3f}") 