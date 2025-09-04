# Split Model Trainer - Trains separate models for distance vs non-distance activities
# Tests all model types on each split to find the optimal combination

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb
import pickle
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class ActivitySplitTrainer:
    """
    Trains separate models for distance and non-distance activities.
    Tests multiple model types to find the best for each activity category.
    """
    
    def __init__(self):
        # Define activity categories
        self.distance_activities = [
            'Run', 'Ride', 'Walk', 'Hike', 'Swim', 'VirtualRide', 
            'Running', 'Cycling', 'Walking', 'Hiking', 'Swimming'
        ]
        
        self.non_distance_activities = [
            'WeightTraining', 'Yoga', 'Workout', 'StairStepper', 
            'Elliptical', 'Strength', 'Pilates', 'CrossFit',
            'Weight Training', 'Gym'
        ]
        
        # Model configurations
        self.models = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1),
            'elastic_net': ElasticNet(alpha=0.1),
            'random_forest': RandomForestRegressor(
                n_estimators=100, 
                max_depth=10, 
                min_samples_split=5,
                random_state=42
            ),
            'gradient_boost': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ),
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            ),
            'svr': SVR(kernel='rbf', C=100, gamma=0.001)
        }
        
        self.best_distance_model = None
        self.best_non_distance_model = None
        self.distance_features = None
        self.non_distance_features = None
        self.results = {}
    
    def create_distance_features(self, df):
        """
        Create features specific to distance-based activities.
        Includes pace, speed, elevation per km, etc.
        """
        print("\nCreating distance-specific features...")
        df = df.copy()
        
        # Core distance features
        features = []
        
        # Essential features for distance activities
        required = ['DURATION_ACTUAL', 'DISTANCE_ACTUAL', 'HRAVG', 'HRMAX']
        for feat in required:
            if feat in df.columns:
                features.append(feat)
        
        # Calculate pace and speed (meaningful for distance activities)
        if 'DURATION_ACTUAL' in df.columns and 'DISTANCE_ACTUAL' in df.columns:
            # Remove any zero distance to avoid division errors
            df = df[df['DISTANCE_ACTUAL'] > 0]
            
            # Pace in minutes per km
            df['PACE'] = (df['DURATION_ACTUAL'] * 60) / (df['DISTANCE_ACTUAL'] / 1000)
            
            # Speed in km/hr
            df['SPEED'] = (df['DISTANCE_ACTUAL'] / 1000) / df['DURATION_ACTUAL']
            
            # Add only one (they're inverses)
            features.append('PACE')
        
        # Elevation features (very relevant for distance activities)
        if 'ELEVATIONGAIN' in df.columns:
            features.append('ELEVATIONGAIN')
            
            if 'DISTANCE_ACTUAL' in df.columns:
                # Elevation per km is crucial for runs/rides
                df['ELEVATION_PER_KM'] = df['ELEVATIONGAIN'] / (df['DISTANCE_ACTUAL'] / 1000)
                features.append('ELEVATION_PER_KM')
        
        # Heart rate features
        if 'HRAVG' in df.columns and 'HRMAX' in df.columns:
            df['INTENSITY_RATIO'] = df['HRAVG'] / df['HRMAX']
            df['HR_RESERVE'] = df['HRMAX'] - 70
            df['HR_ZONE'] = (df['HRAVG'] - 70) / df['HR_RESERVE']
            features.extend(['INTENSITY_RATIO', 'HR_RESERVE', 'HR_ZONE'])
        
        # Efficiency metrics (distance-specific)
        if all(x in df.columns for x in ['DISTANCE_ACTUAL', 'HRAVG']):
            # Distance per heartbeat (efficiency metric)
            df['DISTANCE_PER_BEAT'] = df['DISTANCE_ACTUAL'] / (df['HRAVG'] * df['DURATION_ACTUAL'] * 60)
            features.append('DISTANCE_PER_BEAT')
        
        # Demographics
        if 'AGE' in df.columns and df['AGE'].std() > 0:
            features.append('AGE')
        
        if 'SEX' in df.columns:
            df['SEX_ENCODED'] = (df['SEX'] == 'M').astype(int)
            features.append('SEX_ENCODED')
        
        # Remove any features with no variance
        features = [f for f in features if f in df.columns and df[f].std() > 0]
        
        print(f"Distance features ({len(features)}): {', '.join(features)}")
        return df, features
    
    def create_non_distance_features(self, df):
        """
        Create features for non-distance activities (yoga, weights).
        Excludes distance-related features that would be meaningless.
        """
        print("\nCreating non-distance-specific features...")
        df = df.copy()
        
        features = []
        
        # Core features (no distance data)
        essential = ['DURATION_ACTUAL', 'HRAVG', 'HRMAX']
        for feat in essential:
            if feat in df.columns:
                features.append(feat)
        
        # Heart rate intensity features (very important for stationary activities)
        if 'HRAVG' in df.columns and 'HRMAX' in df.columns:
            df['INTENSITY_RATIO'] = df['HRAVG'] / df['HRMAX']
            df['HR_RESERVE'] = df['HRMAX'] - 70
            df['HR_ZONE'] = (df['HRAVG'] - 70) / df['HR_RESERVE']
            
            # Average HR as percentage of max (good for stationary)
            df['HR_PERCENT_MAX'] = (df['HRAVG'] / df['HRMAX']) * 100
            
            features.extend(['INTENSITY_RATIO', 'HR_RESERVE', 'HR_ZONE', 'HR_PERCENT_MAX'])
        
        # Duration-based intensity
        if 'DURATION_ACTUAL' in df.columns and 'HRAVG' in df.columns:
            # Total heart beats during workout
            df['TOTAL_HEARTBEATS'] = df['HRAVG'] * df['DURATION_ACTUAL'] * 60
            
            # Intensity-duration interaction
            df['INTENSITY_DURATION'] = df['INTENSITY_RATIO'] * df['DURATION_ACTUAL']
            
            features.extend(['TOTAL_HEARTBEATS', 'INTENSITY_DURATION'])
        
        # Time features (might matter for gym/yoga)
        if 'HOUR' in df.columns:
            # Time of day might affect intensity
            df['MORNING'] = (df['HOUR'] < 12).astype(int)
            df['EVENING'] = (df['HOUR'] >= 18).astype(int)
            features.extend(['MORNING', 'EVENING'])
        
        # Demographics
        if 'AGE' in df.columns and df['AGE'].std() > 0:
            features.append('AGE')
        
        if 'SEX' in df.columns:
            df['SEX_ENCODED'] = (df['SEX'] == 'M').astype(int)
            features.append('SEX_ENCODED')
        
        # Note: Deliberately excluding:
        # - DISTANCE_ACTUAL (meaningless for yoga/weights)
        # - PACE/SPEED (undefined without distance)
        # - ELEVATION (not relevant for indoor activities)
        # - ELEVATION_PER_KM (no distance to normalize by)
        
        # Remove any features with no variance
        features = [f for f in features if f in df.columns and df[f].std() > 0]
        
        print(f"Non-distance features ({len(features)}): {', '.join(features)}")
        return df, features
    
    def train_all_models(self, X_train, y_train, X_test, y_test, activity_type):
        """
        Train all model types and return performance metrics.
        """
        results = {}
        
        print(f"\nTraining {activity_type} models...")
        print("-" * 50)
        
        for name, model in self.models.items():
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_train, y_train, 
                                           cv=5, scoring='r2')
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                results[name] = {
                    'model': model,
                    'r2': r2,
                    'mae': mae,
                    'rmse': rmse,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'predictions': y_pred
                }
                
                print(f"{name:15} R²={r2:.4f} MAE={mae:.1f} RMSE={rmse:.1f} CV={cv_mean:.4f}±{cv_std:.4f}")
                
            except Exception as e:
                print(f"{name:15} Failed: {str(e)}")
                results[name] = None
        
        # Find best model
        valid_results = {k: v for k, v in results.items() if v is not None}
        if valid_results:
            best_name = max(valid_results.items(), key=lambda x: x[1]['r2'])[0]
            print(f"\nBest {activity_type} model: {best_name} (R²={results[best_name]['r2']:.4f})")
        else:
            best_name = None
        
        return results, best_name
    
    def fit(self, df):
        """
        Main training function - splits data and trains models for each activity type.
        """
        print("="*60)
        print("SPLIT MODEL TRAINING")
        print("="*60)
        
        # Identify activity type
        if 'ACTIVITY_TYPE' not in df.columns:
            raise ValueError("ACTIVITY_TYPE column required")
        
        # Split into distance and non-distance
        distance_mask = df['ACTIVITY_TYPE'].isin(self.distance_activities)
        non_distance_mask = df['ACTIVITY_TYPE'].isin(self.non_distance_activities)
        
        df_distance = df[distance_mask].copy()
        df_non_distance = df[non_distance_mask].copy()
        
        print(f"\nData split:")
        print(f"  Distance activities: {len(df_distance)} samples")
        print(f"  Non-distance activities: {len(df_non_distance)} samples")
        print(f"  Other/Unknown: {len(df) - len(df_distance) - len(df_non_distance)} samples")
        
        # Store overall results
        self.results = {
            'distance': {},
            'non_distance': {}
        }
        
        # ========== TRAIN DISTANCE MODELS ==========
        if len(df_distance) > 20:  # Need minimum samples
            print("\n" + "="*60)
            print("TRAINING DISTANCE ACTIVITY MODELS")
            print("="*60)
            
            # Create distance-specific features
            df_distance_featured, self.distance_features = self.create_distance_features(df_distance)
            
            # Prepare data
            X = df_distance_featured[self.distance_features]
            y = df_distance_featured['CALORIES']
            
            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            print(f"\nTraining set: {len(X_train)} samples")
            print(f"Test set: {len(X_test)} samples")
            
            # Train all models
            distance_results, best_distance = self.train_all_models(
                X_train, y_train, X_test, y_test, "distance"
            )
            
            self.results['distance'] = distance_results
            self.best_distance_model = distance_results[best_distance]['model'] if best_distance else None
            self.best_distance_name = best_distance
        
        # ========== TRAIN NON-DISTANCE MODELS ==========
        if len(df_non_distance) > 20:  # Need minimum samples
            print("\n" + "="*60)
            print("TRAINING NON-DISTANCE ACTIVITY MODELS")
            print("="*60)
            
            # Create non-distance-specific features
            df_non_distance_featured, self.non_distance_features = self.create_non_distance_features(df_non_distance)
            
            # Prepare data
            X = df_non_distance_featured[self.non_distance_features]
            y = df_non_distance_featured['CALORIES']
            
            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            print(f"\nTraining set: {len(X_train)} samples")
            print(f"Test set: {len(X_test)} samples")
            
            # Train all models
            non_distance_results, best_non_distance = self.train_all_models(
                X_train, y_train, X_test, y_test, "non-distance"
            )
            
            self.results['non_distance'] = non_distance_results
            self.best_non_distance_model = non_distance_results[best_non_distance]['model'] if best_non_distance else None
            self.best_non_distance_name = best_non_distance
        
        # ========== SUMMARY ==========
        self.print_summary()
        
        return self
    
    def print_summary(self):
        """
        Print comprehensive summary of results.
        """
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        
        # Distance models summary
        if self.results['distance']:
            print("\nDISTANCE ACTIVITIES - Model Performance:")
            print("-" * 50)
            print(f"{'Model':<15} {'R²':>8} {'MAE':>8} {'RMSE':>8} {'CV R²':>12}")
            print("-" * 50)
            
            for name, result in self.results['distance'].items():
                if result:
                    print(f"{name:<15} {result['r2']:>8.4f} {result['mae']:>8.1f} "
                          f"{result['rmse']:>8.1f} {result['cv_mean']:>8.4f}±{result['cv_std']:.4f}")
            
            if hasattr(self, 'best_distance_name'):
                print(f"\n✓ Best distance model: {self.best_distance_name}")
                print(f"  Features used: {', '.join(self.distance_features)}")
        
        # Non-distance models summary
        if self.results['non_distance']:
            print("\nNON-DISTANCE ACTIVITIES - Model Performance:")
            print("-" * 50)
            print(f"{'Model':<15} {'R²':>8} {'MAE':>8} {'RMSE':>8} {'CV R²':>12}")
            print("-" * 50)
            
            for name, result in self.results['non_distance'].items():
                if result:
                    print(f"{name:<15} {result['r2']:>8.4f} {result['mae']:>8.1f} "
                          f"{result['rmse']:>8.1f} {result['cv_mean']:>8.4f}±{result['cv_std']:.4f}")
            
            if hasattr(self, 'best_non_distance_name'):
                print(f"\n✓ Best non-distance model: {self.best_non_distance_name}")
                print(f"  Features used: {', '.join(self.non_distance_features)}")
        
        # Overall best combination
        print("\n" + "="*60)
        print("OPTIMAL MODEL COMBINATION")
        print("="*60)
        if hasattr(self, 'best_distance_name') and hasattr(self, 'best_non_distance_name'):
            print(f"Distance activities → {self.best_distance_name}")
            print(f"Non-distance activities → {self.best_non_distance_name}")
    
    def explain_selection(self, n_samples_distance=None, n_samples_non_distance=None):
        """
        Explain why specific models were selected for each activity type.
        """
        try:
            from model_explainer import ModelExplainer
            explainer = ModelExplainer()
            
            print("\n" + "="*70)
            print(" MODEL SELECTION EXPLANATIONS ".center(70))
            print("="*70)
            
            # Explain distance model selection
            if self.results.get('distance') and hasattr(self, 'best_distance_name'):
                best_distance_result = self.results['distance'][self.best_distance_name]
                data_chars = {
                    'activity_type': 'distance',
                    'n_samples': n_samples_distance or 500,
                    'n_features': len(self.distance_features),
                    'has_nonlinear': True,
                    'has_interactions': True,
                    'has_outliers': True,
                    'comparison': self.results['distance']
                }
                
                distance_explanation = explainer.explain_model_selection(
                    self.best_distance_name,
                    best_distance_result,
                    data_chars
                )
                print(distance_explanation)
            
            # Explain non-distance model selection
            if self.results.get('non_distance') and hasattr(self, 'best_non_distance_name'):
                best_non_distance_result = self.results['non_distance'][self.best_non_distance_name]
                data_chars = {
                    'activity_type': 'non_distance',
                    'n_samples': n_samples_non_distance or 200,
                    'n_features': len(self.non_distance_features),
                    'has_nonlinear': True,
                    'has_interactions': True,
                    'comparison': self.results['non_distance']
                }
                
                non_distance_explanation = explainer.explain_model_selection(
                    self.best_non_distance_name,
                    best_non_distance_result,
                    data_chars
                )
                print("\n")
                print(non_distance_explanation)
            
            return explainer
        except ImportError:
            print("\nNote: Install model_explainer for detailed selection explanations")
            return None
        
    def predict(self, df):
        """
        Make predictions using the appropriate model based on activity type.
        """
        predictions = np.zeros(len(df))
        
        # Identify activity types
        distance_mask = df['ACTIVITY_TYPE'].isin(self.distance_activities)
        non_distance_mask = df['ACTIVITY_TYPE'].isin(self.non_distance_activities)
        
        # Predict for distance activities
        if distance_mask.any() and self.best_distance_model:
            df_distance = df[distance_mask].copy()
            df_distance_featured, _ = self.create_distance_features(df_distance)
            X_distance = df_distance_featured[self.distance_features]
            predictions[distance_mask] = self.best_distance_model.predict(X_distance)
        
        # Predict for non-distance activities
        if non_distance_mask.any() and self.best_non_distance_model:
            df_non_distance = df[non_distance_mask].copy()
            df_non_distance_featured, _ = self.create_non_distance_features(df_non_distance)
            X_non_distance = df_non_distance_featured[self.non_distance_features]
            predictions[non_distance_mask] = self.best_non_distance_model.predict(X_non_distance)
        
        return predictions
    
    def save(self, filepath):
        """
        Save the trained model combination.
        """
        model_data = {
            'best_distance_model': self.best_distance_model,
            'best_non_distance_model': self.best_non_distance_model,
            'distance_features': self.distance_features,
            'non_distance_features': self.non_distance_features,
            'distance_activities': self.distance_activities,
            'non_distance_activities': self.non_distance_activities,
            'results': self.results,
            'best_distance_name': getattr(self, 'best_distance_name', None),
            'best_non_distance_name': getattr(self, 'best_non_distance_name', None)
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\nModel saved to: {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """
        Load a saved model combination.
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        trainer = cls()
        trainer.best_distance_model = model_data['best_distance_model']
        trainer.best_non_distance_model = model_data['best_non_distance_model']
        trainer.distance_features = model_data['distance_features']
        trainer.non_distance_features = model_data['non_distance_features']
        trainer.distance_activities = model_data['distance_activities']
        trainer.non_distance_activities = model_data['non_distance_activities']
        trainer.results = model_data.get('results', {})
        trainer.best_distance_name = model_data.get('best_distance_name')
        trainer.best_non_distance_name = model_data.get('best_non_distance_name')
        
        return trainer


def load_and_prepare_data():
    """
    Load your workout data and prepare it for split training.
    """
    print("Loading workout data...")
    
    # Load data
    df = pd.read_csv('../data/processed/workout_data.csv')
    
    # Standardize column names
    df.columns = df.columns.str.upper()
    
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
        'TYPE': 'ACTIVITY_TYPE',
        'SPORT': 'ACTIVITY_TYPE',
        'ACTIVITYTYPE': 'ACTIVITY_TYPE'
    }
    
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns and new_name not in df.columns:
            df[new_name] = df[old_name]
    
    # Basic cleaning
    df = df.dropna(subset=['CALORIES', 'DURATION_ACTUAL', 'HRAVG', 'HRMAX'])
    df = df[(df['CALORIES'] > 5) & (df['CALORIES'] < 2000)]
    df = df[(df['DURATION_ACTUAL'] > 0.0167) & (df['DURATION_ACTUAL'] < 5)]  # 1 min to 5 hours
    
    print(f"Loaded {len(df)} valid workouts")
    
    # Show activity distribution
    if 'ACTIVITY_TYPE' in df.columns:
        print("\nActivity distribution:")
        print(df['ACTIVITY_TYPE'].value_counts())
    
    return df


if __name__ == "__main__":
    print("SPLIT MODEL TRAINER")
    print("="*60)
    print("This will train separate models for distance and non-distance activities")
    print("Testing 9 different model types for each category\n")
    
    # Load data
    df = load_and_prepare_data()
    
    # Initialize and train
    trainer = ActivitySplitTrainer()
    trainer.fit(df)
    
    # Count samples for each type for explanation
    distance_mask = df['ACTIVITY_TYPE'].isin(trainer.distance_activities)
    non_distance_mask = df['ACTIVITY_TYPE'].isin(trainer.non_distance_activities)
    n_distance = distance_mask.sum()
    n_non_distance = non_distance_mask.sum()
    
    # Explain model selection with detailed reasoning
    trainer.explain_selection(n_distance, n_non_distance)
    
    # Save the model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f'../model_outputs/split_model_{timestamp}.pkl'
    trainer.save(model_path)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Model saved to: {model_path}")
    print(f"\nOptimal combination:")
    if hasattr(trainer, 'best_distance_name'):
        print(f"  Distance → {trainer.best_distance_name}")
    if hasattr(trainer, 'best_non_distance_name'):
        print(f"  Non-distance → {trainer.best_non_distance_name}")