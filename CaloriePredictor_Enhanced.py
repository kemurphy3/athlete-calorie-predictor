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
import os

class CaloriePredictor:
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.df = data_manager.df
        self.model = None
        self.feature_columns = None
        self.best_model_name = None
        
    def create_features(self):
        
        print("\nCreating features...")
        
        # Create a copy of the dataframe to avoid SettingWithCopyWarning
        self.df = self.data_manager.df.copy()
        
        # Basic derived features
        self.df.loc[:, 'PACE'] = self.df['DURATION_ACTUAL'] / (self.df['DISTANCE_ACTUAL'] / 1000)  # hours per km
        self.df.loc[:, 'SPEED'] = (self.df['DISTANCE_ACTUAL'] / 1000) / self.df['DURATION_ACTUAL']  # km per hour
        
        # Intensity ratio (HR_avg / HR_max) - common metric in exercise physiology
        # This ratio indicates exercise intensity relative to max capacity
        self.df.loc[:, 'INTENSITY_RATIO'] = self.df['HRAVG'] / self.df['HRMAX']
        
        # Encode categorical variables
        if 'SEX' in self.df.columns:
            self.df.loc[:, 'SEX_ENCODED'] = (self.df['SEX'] == 'M').astype(int)
        
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
    
    def compare_models(self):
        # Compare multiple models using cross-validation
        print("\nComparing multiple models...")
        
        # Prepare features and target
        X = self.df[self.feature_columns].copy()
        y = self.df['CALORIES']
        
        # Split data first, before imputation to avoid data leakage
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Handle missing values using ONLY training data 
        print("Handling missing values using training data only...")
        
        # Calculate demographic-specific imputation values from training data
        # Using a dictionary to store the imputation values makes it easy to apply later using their key
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
        self.best_model_name = best_model_name
        print(f"\nBest model: {best_model_name}")
        
        # Train best model on full training set
        self.model = results[best_model_name]['model']
        self.model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\nFinal Model Performance ({best_model_name}):")
        print(f"  MAE: {mae:.1f} calories")
        print(f"  R²: {r2:.3f}")
        
        return mae, r2, results
    
    def analyze_business_impact(self, y_true, y_pred):
        # Analyze prediction accuracy by calorie ranges to understand business impact
        print("\n++ Business Impact Analysis ++")
        ranges = [(0, 300), (300, 600), (600, 1000), (1000, 2000)]
        
        for low, high in ranges:
            mask = (y_true >= low) & (y_true < high)
            if mask.sum() > 0:
                mae = mean_absolute_error(y_true[mask], y_pred[mask])
                pct_error = (mae / y_true[mask].mean()) * 100 if y_true[mask].mean() > 0 else 0
                print(f"Calories {low}-{high}: MAE = {mae:.1f} ({pct_error:.1f}% error, {mask.sum():,} samples)")
    
    def train_model(self):
        # Train the best model from comparison
        return self.compare_models()
    
    def show_feature_importance(self):
        # Rank and display feature importance
        if self.model is None:
            print("Model not trained yet")
            return
        
        # Handle different model types
        if hasattr(self.model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        elif hasattr(self.model, 'coef_'):
            importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': np.abs(self.model.coef_)
            }).sort_values('importance', ascending=False)
        else:
            print("Model doesn't support feature importance")
            return
        
        print(f"\nFeature Importance ({self.best_model_name}):")
        for i, row in importance.iterrows():
            print(f"  {row['feature']:<20}: {row['importance']:.3f}")
        
        # Create feature importance bar chart so the scale of importance is easily visible
        plt.figure(figsize=(10, 6))
        plt.barh(importance['feature'], importance['importance'])
        plt.xlabel('Feature Importance')
        plt.title(f'{self.best_model_name} Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        
        return importance
    
    def save_model(self):
        # Save the model so it can be easily shared for production use
        if self.model is None:
            print("No model to save")
            return
        
        model_filename = 'calorie_prediction_model.pkl'
        
        # Save model and metadata
        model_data = {
            'model': self.model,
            'model_name': self.best_model_name,
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
        # Apply the model and use it to predict calories based on runs with informing metadata
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
            # Create input dataframe. Cursor informed how to handle missing values to prevent model from failing on input.
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
            
            # Create derived features of known important ratios in the fitness world
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
        # Update the dataframe reference after data changes
        self.df = self.data_manager.df 