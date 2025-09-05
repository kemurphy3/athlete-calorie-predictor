# Enhanced Model Training Script for Athlete Calorie Predictor
# Comprehensive ML pipeline with advanced evaluation and hyperparameter optimization
# Uses ONLY real data from your workout dataset - no synthetic data generation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import lightgbm as lgb
import xgboost as xgb
import pickle
import os
import warnings
from datetime import datetime
import joblib
import scipy.stats

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class DataManager:
    # Data management class for handling workout data loading and cleaning
    # Works with your exact CSV structure without any modifications
    
    def __init__(self, filepath, sample_size=None):
        self.filepath = filepath
        self.sample_size = sample_size
        self.df = None
        
    def load_data(self):
        # Load workout data from CSV file
        print("Loading workout data...")
        self.df = pd.read_csv(self.filepath)
        
        # Standardize column names to uppercase for consistency
        self.df.columns = self.df.columns.str.upper()
        
        if self.sample_size:
            # Random sampling for faster development/testing
            self.df = self.df.sample(n=min(self.sample_size, len(self.df)), random_state=42).reset_index(drop=True)
            print(f"Randomly sampled {len(self.df):,} records")
        else:
            print(f"Using all {len(self.df):,} records")
            
        return self.df
    
    def clean_data(self):
        # Clean data by removing invalid records based on physiological constraints
        print("\n++ Data Cleaning ++")
        initial_rows = len(self.df)
        
        # Map Strava column names to expected names first
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
        
        # Rename columns to match expected names
        print("Mapping Strava columns to expected names...")
        for old_name, new_name in column_mapping.items():
            if old_name in self.df.columns:
                self.df.loc[:, new_name] = self.df[old_name]
                print(f"  Mapped {old_name} -> {new_name}")
            else:
                print(f"  Warning: {old_name} not found in dataset")
        
        print(f"Columns after mapping: {list(self.df.columns)}")
        
        # Check which mapped columns exist and handle missing ones
        required_columns = ['DURATION_ACTUAL', 'DISTANCE_ACTUAL', 'CALORIES', 'HRAVG', 'HRMAX', 'ELEVATIONGAIN', 'AGE', 'WEIGHT']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        
        if missing_columns:
            print(f"Warning: Missing required columns: {missing_columns}")
            print("Available columns:", list(self.df.columns))
            return self.df
        
        # Define reasonable ranges based on exercise physiology
        extreme_calories = (self.df['CALORIES'] < 5) | (self.df['CALORIES'] > 2000) | (self.df['CALORIES'].isnull())
        extreme_duration = (self.df['DURATION_ACTUAL'] < 0.0167) | (self.df['DURATION_ACTUAL'] > 5)  # 1 min to 5 hours
        
        # Smart distance validation: only check distance for distance-based activities
        distance_based_activities = ['Run', 'Ride', 'Walk', 'Hike', 'Swim']
        if 'ACTIVITY_TYPE' in self.df.columns:
            # Only validate distance for activities that should have distance
            is_distance_activity = self.df['ACTIVITY_TYPE'].isin(distance_based_activities)
            extreme_distance = (is_distance_activity & 
                              ((self.df['DISTANCE_ACTUAL'] < 100) | (self.df['DISTANCE_ACTUAL'] > 100000)))
            print(f"Found {(~is_distance_activity).sum()} non-distance activities (yoga, weights, etc.) - keeping these")
        else:
            # Fallback to original logic if no activity type column
            extreme_distance = (self.df['DISTANCE_ACTUAL'] < 100) | (self.df['DISTANCE_ACTUAL'] > 100000)
        
        extreme_hravg = (self.df['HRAVG'] < 30) | (self.df['HRAVG'] > 250)  # HR range 30-250 bpm
        extreme_hrmax = (self.df['HRMAX'] < 30) | (self.df['HRMAX'] > 250)  # HR range 30-250 bpm
        extreme_age = (self.df['AGE'] < 18) | (self.df['AGE'] > 100)  # Age range 18-100 years
        extreme_weight = (self.df['WEIGHT'] < 20) | (self.df['WEIGHT'] > 200)  # Weight range 20-200 kg
        
        # Count records to be removed
        calories_removed = extreme_calories.sum()
        duration_removed = extreme_duration.sum()
        distance_removed = extreme_distance.sum()
        hravg_removed = extreme_hravg.sum()
        hrmax_removed = extreme_hrmax.sum()
        age_removed = extreme_age.sum()
        weight_removed = extreme_weight.sum()
        
        # Remove invalid records
        self.df = self.df[~(extreme_calories | extreme_duration | extreme_distance | 
                           extreme_hravg | extreme_hrmax | extreme_age | extreme_weight)]
        
        final_rows = len(self.df)
        total_removed = initial_rows - final_rows
        
        # Report cleaning results
        print(f"Records flagged for invalid calories: {calories_removed:,}")
        print(f"Records flagged for invalid duration: {duration_removed:,}")
        print(f"Records flagged for invalid distance: {distance_removed:,}")
        print(f"Records flagged for invalid heart rates: {hravg_removed + hrmax_removed:,}")
        print(f"Records flagged for invalid age: {age_removed:,}")
        print(f"Records flagged for invalid weight: {weight_removed:,}")
        print(f"Total rows removed: {total_removed:,}")
        print(f"Final rows after cleaning: {final_rows:,}")
        print(f"Data retention: {((final_rows/initial_rows)*100):.1f}%")
        
        return self.df

class EnhancedCaloriePredictor:
    # Enhanced calorie prediction model with comprehensive evaluation
    # Creates ONLY derived features from your real data - no synthetic generation
    
    def __init__(self, data_manager, output_dir="model_outputs"):
        self.data_manager = data_manager
        self.df = data_manager.df
        self.model = None
        self.feature_columns = None
        self.best_model_name = None
        self.output_dir = output_dir
        self.results = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def create_advanced_features(self):
        # Create derived features from your existing workout data
        print("\n++ Creating Derived Features ++")
        
        # Create a copy to avoid warnings
        self.df = self.data_manager.df.copy()
        
        # Check for potential data leakage
        print("\n++ Data Leakage Detection ++")
        leakage_info = self._detect_data_leakage()
        
        # Automatically remove completely missing features
        if leakage_info['completely_missing']:
            print("\n++ Removing Completely Missing Features ++")
            for col in leakage_info['completely_missing']:
                if col in self.df.columns:
                    print(f"  Removing {col} (100% missing data) - no predictive value")
                    self.df = self.df.drop(columns=[col])
        
        # Automatically remove high-correlation features that cause data leakage
        if leakage_info['high_correlation']:
            print("\n++ Removing Data Leakage Features ++")
            for col, corr in leakage_info['high_correlation']:
                if col in self.df.columns:
                    print(f"  Removing {col} (R²={corr:.3f} with target) - causes data leakage")
                    self.df = self.df.drop(columns=[col])
        
        # Check if age is constant and prompt for birthday if needed
        if 'AGE' in self.df.columns and 'DATE' in self.df.columns:
            age_variance = self.df['AGE'].nunique()
            date_range_years = (pd.to_datetime(self.df['DATE'].max()) - pd.to_datetime(self.df['DATE'].min())).days / 365.25
            
            if age_variance == 1 and date_range_years > 1:
                print(f"\n++ Age Analysis ++")
                print(f"Age variance: {age_variance} unique value(s)")
                print(f"Date range: {date_range_years:.1f} years")
                print(f"Static age: {self.df['AGE'].iloc[0]}")
                print(f"Recommendation: Calculate dynamic age from birthday for better predictions")
                
                # Prompt for birthday
                while True:
                    try:
                        birthday_input = input("Enter your birthday (YYYY-MM-DD format): ")
                        birthday = pd.to_datetime(birthday_input)
                        print(f"Birthday confirmed: {birthday.strftime('%Y-%m-%d')}")
                        break
                    except ValueError:
                        print("Invalid format. Please use YYYY-MM-DD (e.g., 1991-03-13)")
                
                # Calculate dynamic age for each workout
                self.df.loc[:, 'AGE_DYNAMIC'] = (pd.to_datetime(self.df['DATE']) - birthday).dt.days / 365.25
                print(f"Dynamic age range: {self.df['AGE_DYNAMIC'].min():.1f} to {self.df['AGE_DYNAMIC'].max():.1f} years")
                
                # Replace static age with dynamic age
                self.df.loc[:, 'AGE'] = self.df['AGE_DYNAMIC']
                print("Replaced static age with dynamic age calculation")
        
        # Check for other constant features that should be dropped
        features_to_drop = []
        
        if 'WEIGHT' in self.df.columns and self.df['WEIGHT'].nunique() == 1:
            print(f"Warning: Weight is constant ({self.df['WEIGHT'].iloc[0]}) - will be dropped from features")
            features_to_drop.append('WEIGHT')
        
        if 'SEX' in self.df.columns and self.df['SEX'].nunique() == 1:
            print(f"Warning: Sex is constant ({self.df['SEX'].iloc[0]}) - will be dropped from features")
            features_to_drop.append('SEX')
        
        if features_to_drop:
            print(f"Features to drop due to zero variance: {', '.join(features_to_drop)}")
        
        # NOTE: Removed pace/speed features to prevent data leakage
        # These features directly incorporate duration and distance which are proportional to calories
        print("  Skipping pace/speed features (prevent data leakage)")
        
        # NOTE: Removed intensity ratio and HR features to simplify
        # Only keeping essential HRAVG and HRMAX features
        print("  Skipping intensity ratio and HR-derived features (simplifying feature set)")
        
        # NOTE: Removed elevation per km to simplify
        # Only keeping essential ELEVATIONGAIN feature
        print("  Skipping elevation per km (simplifying feature set)")
        
        # Encode categorical variables
        if 'SEX' in self.df.columns:
            self.df.loc[:, 'SEX_ENCODED'] = (self.df['SEX'] == 'M').astype(int)
        
        # NOTE: Removed interaction features to simplify
        # Only keeping essential features without mathematical combinations
        print("  Skipping interaction features (simplifying feature set)")
        
        # Define feature set (includes essential features)
        # NOTE: Including DURATION_ACTUAL and DISTANCE_ACTUAL as they are legitimate inputs
        # These are the primary drivers of calorie burn in exercise
        potential_features = [
            'DURATION_ACTUAL', 'DISTANCE_ACTUAL', 'HRMAX', 'HRAVG', 'ELEVATIONGAIN', 'AGE', 'WEIGHT', 'SEX_ENCODED'
        ]
        
        # Note: TRAININGSTRESSSCOREACTUAL and ELEVATIONAVG removed due to 100% missing data
        # Note: INTENSITY_RATIO, HR_RESERVE, HR_ZONE, ELEVATION_PER_KM removed to simplify
        # Note: HR_WEIGHT removed to simplify feature set
        
        # Filter to only include available features and exclude constant ones
        self.feature_columns = [col for col in potential_features if col in self.df.columns and col not in features_to_drop]
        
        # Additional feature quality filtering
        print("\n++ Feature Quality Assessment ++")
        self.feature_columns = self._filter_low_quality_features(self.feature_columns)
        
        print(f"Created {len(self.feature_columns)} derived features from your real data")
        print(f"Features: {', '.join(self.feature_columns)}")
        
        return self.df
    
    def _detect_data_leakage(self):
        # Comprehensive data leakage detection
        print("Checking for potential data leakage...")
        
        # 1. Check for completely missing features (100% NaN)
        completely_missing_features = []
        for col in self.df.columns:
            if col != 'CALORIES' and self.df[col].isna().sum() == len(self.df):
                completely_missing_features.append(col)
                print(f"  WARNING: Completely missing feature - {col} (100% NaN)")
        
        # 2. Check for direct target leakage (features that directly contain calorie info)
        target_leakage_features = []
        for col in self.df.columns:
            if col != 'CALORIES' and 'CALORIE' in col.upper():
                target_leakage_features.append(col)
                print(f"  WARNING: Potential target leakage - {col} contains 'calorie'")
        
        # 2. Check for extremely high correlations with target
        high_corr_features = []
        for col in self.df.columns:
            if col != 'CALORIES' and col in self.df.select_dtypes(include=[np.number]).columns:
                corr = abs(self.df[col].corr(self.df['CALORIES']))
                if corr > 0.95:  # Suspiciously high correlation
                    high_corr_features.append((col, corr))
                    print(f"  WARNING: Extremely high correlation with target - {col}: {corr:.3f}")
        
        # 3. Check for feature redundancy (highly correlated features)
        feature_correlations = self.df[self.feature_columns].corr() if self.feature_columns else pd.DataFrame()
        redundant_features = []
        
        if not feature_correlations.empty:
            for i in range(len(feature_correlations.columns)):
                for j in range(i+1, len(feature_correlations.columns)):
                    col1 = feature_correlations.columns[i]
                    col2 = feature_correlations.columns[j]
                    corr = abs(feature_correlations.iloc[i, j])
                    if corr > 0.95:  # Nearly perfect correlation
                        redundant_features.append((col1, col2, corr))
                        print(f"  WARNING: Feature redundancy - {col1} vs {col2}: {corr:.3f}")
        
        # 4. Check for suspicious feature distributions
        suspicious_features = []
        for col in self.df.columns:
            if col != 'CALORIES' and col in self.df.select_dtypes(include=[np.number]).columns:
                # Check if feature has suspiciously low variance
                if self.df[col].std() < 1e-6:
                    suspicious_features.append(col)
                    print(f"  WARNING: Near-zero variance feature - {col}")
                
                # Check if feature is perfectly linear with target
                if len(self.df) > 10 and self.df[col].std() > 1e-6:  # Need enough samples and variance
                    try:
                        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
                            self.df[col], self.df['CALORIES']
                        )
                        if abs(r_value) > 0.99:  # Nearly perfect linear relationship
                            suspicious_features.append(col)
                            print(f"  WARNING: Nearly perfect linear relationship with target - {col}: R²={r_value**2:.3f}")
                    except ValueError:
                        # Skip features with no variance
                        pass
        
        # 5. Summary
        total_warnings = len(target_leakage_features) + len(high_corr_features) + len(redundant_features) + len(suspicious_features)
        if total_warnings == 0:
            print("  No obvious data leakage detected")
        else:
            print(f"  Total warnings: {total_warnings}")
            print("  Recommendation: Investigate these features for data leakage")
        
        return {
            'completely_missing': completely_missing_features,
            'target_leakage': target_leakage_features,
            'high_correlation': high_corr_features,
            'redundant_features': redundant_features,
            'suspicious_features': suspicious_features
        }
    
    def _filter_low_quality_features(self, candidate_features):
        # Remove low-quality features that could cause overfitting
        print("Filtering low-quality features...")
        
        if not candidate_features:
            return []
        
        # Get numeric features only
        numeric_features = [col for col in candidate_features if col in self.df.select_dtypes(include=[np.number]).columns]
        
        if not numeric_features:
            return candidate_features
        
        # 1. Remove features with near-zero variance
        low_variance_features = []
        for col in numeric_features:
            if self.df[col].std() < 1e-6:
                low_variance_features.append(col)
                print(f"  Removing low variance feature: {col}")
        
        # 2. Remove features with extremely high correlation with target (>0.95)
        high_target_corr_features = []
        for col in numeric_features:
            if col != 'CALORIES':
                corr = abs(self.df[col].corr(self.df['CALORIES']))
                if corr > 0.95:
                    high_target_corr_features.append(col)
                    print(f"  Removing high target correlation feature: {col} (R²={corr:.3f})")
        
        # 3. Remove redundant features (keep one from each highly correlated pair)
        redundant_pairs = []
        feature_corr_matrix = self.df[numeric_features].corr()
        
        for i in range(len(feature_corr_matrix.columns)):
            for j in range(i+1, len(feature_corr_matrix.columns)):
                col1 = feature_corr_matrix.columns[i]
                col2 = feature_corr_matrix.columns[j]
                corr = abs(feature_corr_matrix.iloc[i, j])
                if corr > 0.95:
                    redundant_pairs.append((col1, col2, corr))
        
        # Remove one feature from each redundant pair (keep the one with higher correlation to target)
        redundant_features_to_remove = []
        for col1, col2, corr in redundant_pairs:
            corr1 = abs(self.df[col1].corr(self.df['CALORIES'])) if col1 != 'CALORIES' else 0
            corr2 = abs(self.df[col2].corr(self.df['CALORIES'])) if col2 != 'CALORIES' else 0
            
            if corr1 < corr2:
                redundant_features_to_remove.append(col1)
                print(f"  Removing redundant feature: {col1} (kept {col2}, corr={corr:.3f})")
            else:
                redundant_features_to_remove.append(col2)
                print(f"  Removing redundant feature: {col2} (kept {col1}, corr={corr:.3f})")
        
        # Combine all features to remove
        features_to_remove = set(low_variance_features + high_target_corr_features + redundant_features_to_remove)
        
        # Filter features
        filtered_features = [col for col in candidate_features if col not in features_to_remove]
        
        print(f"Removed {len(features_to_remove)} low-quality features")
        print(f"Remaining features: {len(filtered_features)}")
        
        return filtered_features
    
    def _validate_performance_suspicious(self):
        # Check if performance metrics suggest overfitting or data leakage
        print("Validating performance metrics...")
        
        best_model_metrics = self.results[self.best_model_name]
        
        # Check for suspiciously high R²
        if best_model_metrics['r2_cv_mean'] > 0.95:
            print(f"  WARNING: Extremely high R² CV ({best_model_metrics['r2_cv_mean']:.3f}) - investigate for data leakage")
            print("  Possible causes:")
            print("    - Features directly derived from target variable")
            print("    - Temporal data leakage (using future information)")
            print("    - Overly complex model for dataset size")
            print("    - Insufficient cross-validation folds")
        
        # Check for suspiciously low MAE
        if best_model_metrics['mae_cv_mean'] < 5:  # Less than 5 calories error
            print(f"  WARNING: Extremely low MAE CV ({best_model_metrics['mae_cv_mean']:.1f}) - investigate for data leakage")
        
        # Check cross-validation stability
        if best_model_metrics['r2_cv_std'] > 0.1:  # High variance across folds
            print(f"  WARNING: High CV variance (R² std: {best_model_metrics['r2_cv_std']:.3f}) - model may be unstable")
        
        # Check feature count vs sample size ratio
        feature_ratio = len(self.feature_columns) / len(self.df)
        if feature_ratio > 0.1:  # More than 10% features to samples
            print(f"  WARNING: High feature-to-sample ratio ({feature_ratio:.3f}) - risk of overfitting")
        
        # If no warnings, confirm good practices
        if (best_model_metrics['r2_cv_mean'] <= 0.95 and 
            best_model_metrics['mae_cv_mean'] >= 5 and 
            best_model_metrics['r2_cv_std'] <= 0.1 and 
            feature_ratio <= 0.1):
            print("  Performance metrics appear reasonable - no obvious issues detected")
    
    def handle_missing_values(self, X_train, X_test, y_train, y_test):
        # Handle missing values using only training data statistics
        print("\n++ Handling Missing Values ++")
        
        # First, convert string 'nan' to actual NaN values and ensure numeric types
        for col in X_train.columns:
            if X_train[col].dtype == 'object':
                # Replace various forms of missing values
                X_train[col] = X_train[col].replace(['nan', 'None', ''], np.nan)
                X_test[col] = X_test[col].replace(['nan', 'None', ''], np.nan)
                
                # Try to convert to numeric
                try:
                    X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
                    X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
                except:
                    pass
        
        # Calculate imputation values from training data only
        for col in X_train.columns:
            if X_train[col].isnull().sum() > 0:
                if X_train[col].dtype in ['int64', 'float64']:
                    impute_value = X_train[col].median()
                else:
                    impute_value = X_train[col].mode()[0]
                
                # Apply to both training and test sets
                X_train[col] = X_train[col].fillna(impute_value)
                X_test[col] = X_test[col].fillna(impute_value)
                print(f"  Imputed {col} with {impute_value:.2f}")
        
        # Final check for any remaining NaN values
        remaining_nans_train = X_train.isnull().sum().sum()
        remaining_nans_test = X_test.isnull().sum().sum()
        
        if remaining_nans_train > 0 or remaining_nans_test > 0:
            print(f"Warning: {remaining_nans_train} NaN values remain in training set")
            print(f"Warning: {remaining_nans_test} NaN values remain in test set")
            
            # Instead of dropping all rows, let's see what's causing the issue
            print("Columns with remaining NaN values:")
            for col in X_train.columns:
                if X_train[col].isnull().sum() > 0:
                    print(f"  {col}: {X_train[col].isnull().sum()} NaN values")
            
            # Try a more aggressive imputation approach
            for col in X_train.columns:
                if X_train[col].isnull().sum() > 0:
                    if X_train[col].dtype in ['int64', 'float64']:
                        # Use 0 for numeric columns if median fails
                        impute_value = X_train[col].median() if not pd.isna(X_train[col].median()) else 0
                    else:
                        # Use 'Unknown' for categorical columns if mode fails
                        impute_value = X_train[col].mode()[0] if len(X_train[col].mode()) > 0 else 'Unknown'
                    
                    X_train[col] = X_train[col].fillna(impute_value)
                    X_test[col] = X_test[col].fillna(impute_value)
                    print(f"  Aggressive imputation of {col} with {impute_value}")
        
        # Final verification
        final_nans_train = X_train.isnull().sum().sum()
        final_nans_test = X_test.isnull().sum().sum()
        print(f"Final NaN count - Training: {final_nans_train}, Test: {final_nans_test}")
        
        if final_nans_train > 0 or final_nans_test > 0:
            print("ERROR: Still have NaN values after aggressive imputation")
            return None, None, None, None
        
        return X_train, X_test, y_train, y_test
    
    def compare_models(self):
        # Compare multiple models using cross-validation with parallel processing
        print("\n++ Model Comparison ++")
        
        # Prepare features and target
        X = self.df[self.feature_columns].copy()
        y = self.df['CALORIES']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )
        
        # Handle missing values
        result = self.handle_missing_values(X_train, X_test, y_train, y_test)
        if result is None:
            print("ERROR: Failed to handle missing values. Cannot proceed with training.")
            return None, None
        
        X_train, X_test, y_train, y_test = result
        
        # Scale features to improve model convergence
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()  # Save scaler as instance variable
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrames to maintain column names
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
        print("Features scaled for better model convergence")
        
        # Define models to compare with expanded hyperparameter grids for better optimization
        models = {
            'Linear Regression': {
                'model': LinearRegression(),
                'params': {}
                # Baseline model - interpretable, fast, good for understanding feature relationships
                # No hyperparameters needed - serves as performance baseline
            },
            'Ridge Regression': {
                'model': Ridge(),
                'params': {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
                # L2 regularization - prevents overfitting, handles multicollinearity
                # Good for when features are correlated (common in exercise physiology)
            },
            'Lasso Regression': {
                'model': Lasso(max_iter=5000, tol=1e-3, warm_start=True),
                'params': {'alpha': [0.01, 0.1, 1.0, 10.0, 50.0, 100.0]}
                # L1 regularization - performs feature selection, creates sparse models
                # Useful for identifying most important physiological factors
                # Increased max_iter, relaxed tolerance, warm_start for better convergence
            },
            'Random Forest': {
                'model': RandomForestRegressor(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],      # More trees = better performance
                    'max_depth': [10, 15, 20, None],       # Control tree complexity
                    'min_samples_split': [2, 5, 10],       # Prevent overfitting
                    'min_samples_leaf': [1, 2, 4]          # Ensure leaf nodes have enough samples
                }
                # Ensemble method - robust, handles non-linear relationships well
                # Good for exercise data where relationships may be complex
            },
            'Gradient Boosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],       # Number of boosting stages
                    'learning_rate': [0.01, 0.05, 0.1],    # Lower = more robust, higher = faster
                    'max_depth': [3, 5, 7],                 # Shallow trees prevent overfitting
                    'subsample': [0.8, 0.9, 1.0]           # Fraction of samples for each tree
                }
                # Sequential boosting - often achieves highest accuracy
                # Good for complex patterns in athletic performance data
            },
            'LightGBM': {
                'model': lgb.LGBMRegressor(random_state=42, verbose=-1),
                'params': {
                    'n_estimators': [100, 200, 300],       # Number of boosting iterations
                    'learning_rate': [0.01, 0.05, 0.1],    # Controls boosting speed
                    'max_depth': [5, 7, 10],                # Tree depth - balance complexity
                    'num_leaves': [31, 63, 127],            # Maximum leaves per tree
                    'subsample': [0.8, 0.9, 1.0]           # Row sampling for each tree
                }
                # Gradient boosting with leaf-wise growth - very fast and accurate
                # Excellent for large datasets like yours (50k+ records)
            },
            'XGBoost': {
                'model': xgb.XGBRegressor(random_state=42, verbosity=0),
                'params': {
                    'n_estimators': [100, 200, 300],       # Number of boosting rounds
                    'learning_rate': [0.01, 0.05, 0.1],    # Step size shrinkage
                    'max_depth': [3, 5, 7],                 # Maximum tree depth
                    'subsample': [0.8, 0.9, 1.0],          # Row sampling ratio
                    'colsample_bytree': [0.8, 0.9, 1.0]    # Column sampling ratio
                }
                # Extreme gradient boosting - industry standard, excellent performance
                # Robust regularization and handles missing values well
            },
        }
        
        # Compare models using cross-validation with parallel processing
        results = {}
        print("\n++ Model Performance Comparison (Parallel Processing Enabled) ++")
        print(f"{'Model':<20} {'R² CV':<8} {'MAE CV':<10} {'RMSE CV':<10}")
        print("-" * 60)
        
        # Determine optimal number of jobs for parallel processing
        import multiprocessing
        n_jobs = min(multiprocessing.cpu_count(), 8)  # Cap at 8 to avoid memory issues
        print(f"Using {n_jobs} parallel jobs for hyperparameter tuning")
        
        # Calculate total grid search combinations for progress tracking
        total_combinations = 0
        for model_info in models.values():
            if model_info['params']:
                # Calculate combinations for this model
                param_combinations = 1
                for param_values in model_info['params'].values():
                    param_combinations *= len(param_values)
                total_combinations += param_combinations
            else:
                total_combinations += 1
        print(f"Total hyperparameter combinations to evaluate: {total_combinations}")
        
        for i, (name, model_info) in enumerate(models.items(), 1):
            print(f"Training {name} ({i}/{len(models)})...")
            model = model_info['model']
            params = model_info['params']
            
            if params:
                # Hyperparameter tuning with parallel processing
                grid_search = GridSearchCV(
                    model, params, cv=5, scoring='neg_mean_absolute_error',
                    n_jobs=n_jobs, verbose=0, return_train_score=True
                )
                grid_search.fit(X_train_scaled, y_train)
                best_model = grid_search.best_estimator_
                print(f"  Best params: {grid_search.best_params_}")
            else:
                best_model = model
            
            # Cross-validation with multiple metrics (also parallel)
            cv_r2 = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring='r2', n_jobs=n_jobs)
            cv_mae = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring='neg_mean_absolute_error', n_jobs=n_jobs)
            cv_rmse = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error', n_jobs=n_jobs)
            
            # Store results
            results[name] = {
                'model': best_model,
                'r2_cv_mean': cv_r2.mean(),
                'r2_cv_std': cv_r2.std(),
                'mae_cv_mean': -cv_mae.mean(),
                'mae_cv_std': cv_mae.std(),
                'rmse_cv_mean': np.sqrt(-cv_rmse.mean()),
                'rmse_cv_std': np.sqrt(cv_rmse.std())
            }
            
            # Print results
            print(f"{name:<20} {cv_r2.mean():<8.3f} {-cv_mae.mean():<10.1f} "
                  f"{np.sqrt(-cv_rmse.mean()):<10.1f}")
        
        # Find best model
        best_model_name = min(results.keys(), key=lambda x: results[x]['mae_cv_mean'])
        self.best_model_name = best_model_name
        self.results = results
        
        # Check for suspiciously high performance (potential overfitting)
        print("\n++ Performance Validation ++")
        self._validate_performance_suspicious()
        
        print(f"\n++ Best Model: {best_model_name} ++")
        print(f"R² CV: {results[best_model_name]['r2_cv_mean']:.3f} ± {results[best_model_name]['r2_cv_std']:.3f}")
        print(f"MAE CV: {results[best_model_name]['mae_cv_mean']:.1f} ± {results[best_model_name]['mae_cv_std']:.1f}")
        print(f"RMSE CV: {results[best_model_name]['rmse_cv_mean']:.1f} ± {results[best_model_name]['rmse_cv_std']:.1f}")
        
        # Train best model on full training set
        self.model = results[best_model_name]['model']
        self.model.fit(X_train_scaled, y_train)
        
        # Final evaluation on test set
        y_pred = self.model.predict(X_test_scaled)
        test_metrics = self.calculate_test_metrics(y_test, y_pred)
        
        return test_metrics, results
    
    def calculate_test_metrics(self, y_true, y_pred):
        # Calculate comprehensive test set metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # Business metrics
        mean_calories = y_true.mean()
        mae_percentage = (mae / mean_calories) * 100
        
        test_metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2,
            'MAE_Percentage': mae_percentage,
            'Mean_Calories': mean_calories
        }
        
        print(f"\n++ Final Test Set Performance ({self.best_model_name}) ++")
        print(f"MAE: {mae:.1f} calories ({mae_percentage:.1f}% of mean)")
        print(f"RMSE: {rmse:.1f} calories")
        print(f"R²: {r2:.3f}")
        
        return test_metrics
    
    def analyze_business_impact(self, y_true, y_pred):
        # Analyze prediction accuracy across workout categories
        print("\n++ Business Impact Analysis ++")
        
        # Define workout categories based on calorie ranges
        workout_categories = [
            (0, 300, "Easy Workout", "Light jog, walking, recovery"),
            (300, 600, "Moderate Workout", "5K-10K run, moderate cycling"),
            (600, 1000, "Intense Workout", "10K-21K run, intense cycling"),
            (1000, 1500, "Very Intense", "Half marathon, long cycling"),
            (1500, 2000, "Extreme Workout", "Marathon, ultra-endurance")
        ]
        
        impact_analysis = []
        
        for low, high, category, description in workout_categories:
            mask = (y_true >= low) & (y_true < high)
            if mask.sum() > 0:
                mae = mean_absolute_error(y_true[mask], y_pred[mask])
                rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
                pct_error = (mae / y_true[mask].mean()) * 100 if y_true[mask].mean() > 0 else 0
                
                impact_analysis.append({
                    'Category': category,
                    'Description': description,
                    'Calorie_Range': f"{low}-{high}",
                    'Sample_Count': mask.sum(),
                    'MAE': mae,
                    'RMSE': rmse,
                    'Error_Percentage': pct_error,
                    'Mean_Calories': y_true[mask].mean()
                })
                
                print(f"{category:<20}: MAE = {mae:.1f} ({pct_error:.1f}% error, {mask.sum():,} samples)")
        
        return pd.DataFrame(impact_analysis)
    
    def create_visualizations(self, y_true, y_pred, save_plots=True):
        # Create model performance visualizations
        print("\n++ Creating Performance Visualizations ++")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Model Performance: {self.best_model_name}', fontsize=16, fontweight='bold')
        
        # 1. Prediction vs Actual
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6, s=20)
        axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Calories')
        axes[0, 0].set_ylabel('Predicted Calories')
        axes[0, 0].set_title('Prediction vs Actual')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add metrics
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        axes[0, 0].text(0.05, 0.95, f'R² = {r2:.3f}\nMAE = {mae:.1f}', 
                        transform=axes[0, 0].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 2. Residuals plot
        residuals = y_true - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6, s=20)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('Predicted Calories')
        axes[0, 1].set_ylabel('Residuals (Actual - Predicted)')
        axes[0, 1].set_title('Residuals Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            axes[1, 0].barh(importance['feature'], importance['importance'])
            axes[1, 0].set_xlabel('Feature Importance')
            axes[1, 0].set_title('Feature Importance')
            axes[1, 0].grid(True, alpha=0.3)
        elif hasattr(self.model, 'coef_'):
            importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': np.abs(self.model.coef_)
            }).sort_values('importance', ascending=True)
            
            axes[1, 0].barh(importance['feature'], importance['importance'])
            axes[1, 0].set_xlabel('Feature Coefficient (Absolute)')
            axes[1, 0].set_title('Feature Coefficients')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Model comparison
        model_names = list(self.results.keys())
        r2_scores = [self.results[name]['r2_cv_mean'] for name in model_names]
        
        bars = axes[1, 1].bar(range(len(model_names)), r2_scores)
        axes[1, 1].set_xlabel('Models')
        axes[1, 1].set_ylabel('R² Score (CV)')
        axes[1, 1].set_title('Model Comparison (R² Scores)')
        axes[1, 1].set_xticks(range(len(model_names)))
        axes[1, 1].set_xticklabels(model_names, rotation=45, ha='right')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Highlight best model
        best_idx = model_names.index(self.best_model_name)
        bars[best_idx].set_color('green')
        bars[best_idx].set_alpha(0.7)
        
        plt.tight_layout()
        
        if save_plots:
            plot_filename = os.path.join(self.output_dir, f'model_performance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Performance plots saved to: {plot_filename}")
        
        plt.show()
        return fig
    
    def save_model(self):
        # Save the trained model with metadata
        if self.model is None:
            print("No model to save")
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = os.path.join(self.output_dir, f'calorie_model_{timestamp}.pkl')
        
        # Model data including preprocessing information
        model_data = {
            'model': self.model,
            'model_name': self.best_model_name,
            'feature_columns': self.feature_columns,
            'training_results': self.results,
            'scaler': self.scaler,  # Save scaler for consistent feature scaling
            'metadata': {
                'training_timestamp': timestamp,
                'dataset_size': len(self.df),
                'feature_count': len(self.feature_columns),
                'cross_validation_folds': 5
            }
        }
        
        # Save with joblib
        joblib.dump(model_data, model_filename)
        
        print(f"\n++ Model Saved ++")
        print(f"Filename: {model_filename}")
        print(f"Model: {self.best_model_name}")
        print(f"Features: {len(self.feature_columns)}")
        print(f"Training Records: {len(self.df):,}")
        
        return model_filename
    
    def generate_report(self, test_metrics, business_impact):
        # Generate comprehensive model report
        print("\n++ Generating Model Report ++")
        
        report_filename = os.path.join(self.output_dir, f'model_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
        
        with open(report_filename, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("ATHLETE CALORIE PREDICTOR - MODEL REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("MODEL OVERVIEW\n")
            
            f.write("MODEL OVERVIEW\n")
            f.write("-" * 40 + "\n")
            f.write(f"Best Model: {self.best_model_name}\n")
            f.write(f"Training Records: {len(self.df):,}\n")
            f.write(f"Features: {len(self.feature_columns)}\n")
            f.write(f"Cross-Validation Folds: 5\n\n")
            
            f.write("TECHNICAL PERFORMANCE\n")
            f.write("-" * 40 + "\n")
            f.write(f"R² Score: {test_metrics['R²']:.3f}\n")
            f.write(f"MAE: {test_metrics['MAE']:.1f} calories\n")
            f.write(f"RMSE: {test_metrics['RMSE']:.1f} calories\n")
            f.write(f"MAE Percentage: {test_metrics['MAE_Percentage']:.1f}%\n\n")
            
            f.write("MODEL COMPARISON (Cross-Validation)\n")
            f.write("-" * 40 + "\n")
            for name, metrics in self.results.items():
                f.write(f"{name}:\n")
                f.write(f"  R² CV: {metrics['r2_cv_mean']:.3f} ± {metrics['r2_cv_std']:.3f}\n")
                f.write(f"  MAE CV: {metrics['mae_cv_mean']:.1f} ± {metrics['mae_cv_std']:.1f}\n")
                f.write(f"  RMSE CV: {metrics['rmse_cv_mean']:.1f} ± {metrics['rmse_cv_std']:.1f}\n\n")
            
            f.write("BUSINESS IMPACT ANALYSIS\n")
            f.write("-" * 40 + "\n")
            f.write(business_impact.to_string(index=False))
            f.write("\n\n")
            
            f.write("FEATURE LIST\n")
            f.write("-" * 40 + "\n")
            for i, feature in enumerate(self.feature_columns, 1):
                f.write(f"{i:2d}. {feature}\n")
            f.write("\n")
            
            f.write("TRAINING SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write("Model trained on authentic workout data with derived features.\n")
            f.write("All features are mathematical transformations of existing physiological data.\n")
        
        print(f"Comprehensive report saved to: {report_filename}")
        return report_filename

def main():
    # Main execution function for the enhanced model training pipeline
    print("=" * 80)
    print("ENHANCED ATHLETE CALORIE PREDICTOR - MODEL TRAINING PIPELINE")
    print("=" * 80)
    print("=" * 80)
    
    # Initialize data manager and load data
    print("\n++ Initializing Data Pipeline ++")
    data_manager = DataManager('data/workouts.csv', sample_size=None)
    df = data_manager.load_data()
    data_manager.clean_data()
    
    print(f"Dataset loaded: {len(df):,} records")
    print(f"Features available: {len(df.columns)}")
    
    # Initialize enhanced predictor
    predictor = EnhancedCaloriePredictor(data_manager)
    
    # Create derived features (no synthetic data)
    df_with_features = predictor.create_advanced_features()
    
    # Train and compare models
    test_metrics, model_results = predictor.compare_models()
    
    if test_metrics is None:
        print("ERROR: Model training failed. Cannot proceed with analysis.")
        return
    
    # Business impact analysis - use the ACTUAL test set from training
    # Get the test set that was properly separated during training
    X = df_with_features[predictor.feature_columns].copy()
    y = df_with_features['CALORIES']
    
    # Use the same split as the model training
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    # Scale the test features using the same scaler
    X_test_scaled = predictor.scaler.transform(X_test)
    
    # Make predictions on the PROPER test set
    y_pred = predictor.model.predict(X_test_scaled)
    
    # Business impact analysis
    business_impact = predictor.analyze_business_impact(y_test, y_pred)
    
    # Create visualizations using the proper test set
    predictor.create_visualizations(y_test, y_pred)
    
    # Save model
    model_file = predictor.save_model()
    
    # Generate report
    report_file = predictor.generate_report(test_metrics, business_impact)
    
    print("\n" + "=" * 80)
    print("MODEL TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Model saved: {model_file}")
    print(f"Report generated: {report_file}")
    print(f"Best model: {predictor.best_model_name}")
    print(f"Final R²: {test_metrics['R²']:.3f}")
    print(f"Final MAE: {test_metrics['MAE']:.1f} calories")
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main() 