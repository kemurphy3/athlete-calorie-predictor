# Model Combiner - Intelligently combines the best models for different activity types
# Provides a unified interface for predictions across all activity types

import pandas as pd
import numpy as np
import pickle
from datetime import datetime


class SmartCaloriePredictor:
    """
    Unified predictor that automatically routes to the best model based on activity type.
    Combines the best distance and non-distance models into a single interface.
    """
    
    def __init__(self, split_model_path=None):
        self.distance_model = None
        self.non_distance_model = None
        self.distance_features = None
        self.non_distance_features = None
        self.distance_activities = None
        self.non_distance_activities = None
        self.model_info = {}
        
        if split_model_path:
            self.load(split_model_path)
    
    def load(self, model_path):
        """
        Load a trained split model.
        """
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.distance_model = model_data['best_distance_model']
        self.non_distance_model = model_data['best_non_distance_model']
        self.distance_features = model_data['distance_features']
        self.non_distance_features = model_data['non_distance_features']
        self.distance_activities = model_data['distance_activities']
        self.non_distance_activities = model_data['non_distance_activities']
        
        # Store model info for reporting
        self.model_info = {
            'distance_model_type': model_data.get('best_distance_name', 'Unknown'),
            'non_distance_model_type': model_data.get('best_non_distance_name', 'Unknown'),
            'distance_performance': self._get_model_performance(model_data, 'distance'),
            'non_distance_performance': self._get_model_performance(model_data, 'non_distance')
        }
        
        print(f"Loaded SmartCaloriePredictor:")
        print(f"  Distance model: {self.model_info['distance_model_type']}")
        print(f"  Non-distance model: {self.model_info['non_distance_model_type']}")
    
    def _get_model_performance(self, model_data, activity_type):
        """
        Extract performance metrics from model data.
        """
        if 'results' in model_data and activity_type in model_data['results']:
            best_name = model_data.get(f'best_{activity_type}_name')
            if best_name and best_name in model_data['results'][activity_type]:
                result = model_data['results'][activity_type][best_name]
                return {
                    'r2': result.get('r2', 0),
                    'mae': result.get('mae', 0),
                    'rmse': result.get('rmse', 0)
                }
        return None
    
    def identify_activity_type(self, activity_name):
        """
        Determine if an activity is distance-based or non-distance.
        Returns: 'distance', 'non_distance', or 'unknown'
        """
        activity_lower = activity_name.lower()
        
        # Check distance activities
        for dist_activity in self.distance_activities:
            if dist_activity.lower() in activity_lower or activity_lower in dist_activity.lower():
                return 'distance'
        
        # Check non-distance activities
        for non_dist_activity in self.non_distance_activities:
            if non_dist_activity.lower() in activity_lower or activity_lower in non_dist_activity.lower():
                return 'non_distance'
        
        # Try to make intelligent guess based on keywords
        distance_keywords = ['run', 'ride', 'walk', 'hike', 'swim', 'cycle', 'bike']
        non_distance_keywords = ['weight', 'strength', 'yoga', 'gym', 'lift', 'cross', 'elliptical']
        
        for keyword in distance_keywords:
            if keyword in activity_lower:
                return 'distance'
        
        for keyword in non_distance_keywords:
            if keyword in activity_lower:
                return 'non_distance'
        
        return 'unknown'
    
    def prepare_features(self, input_data, activity_type):
        """
        Prepare features based on activity type.
        input_data: dict or DataFrame with raw input
        activity_type: 'distance' or 'non_distance'
        """
        # Convert to DataFrame if dict
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
        
        # Standardize column names
        df.columns = df.columns.str.upper()
        
        if activity_type == 'distance':
            # Prepare distance features
            features = self.distance_features
            
            # Calculate derived features
            if 'DISTANCE_ACTUAL' in df.columns and df['DISTANCE_ACTUAL'].iloc[0] > 0:
                if 'PACE' in features:
                    df['PACE'] = (df['DURATION_ACTUAL'] * 60) / (df['DISTANCE_ACTUAL'] / 1000)
                if 'SPEED' in features:
                    df['SPEED'] = (df['DISTANCE_ACTUAL'] / 1000) / df['DURATION_ACTUAL']
                if 'ELEVATION_PER_KM' in features and 'ELEVATIONGAIN' in df.columns:
                    df['ELEVATION_PER_KM'] = df['ELEVATIONGAIN'] / (df['DISTANCE_ACTUAL'] / 1000)
                if 'DISTANCE_PER_BEAT' in features:
                    df['DISTANCE_PER_BEAT'] = df['DISTANCE_ACTUAL'] / (df['HRAVG'] * df['DURATION_ACTUAL'] * 60)
        else:
            # Prepare non-distance features
            features = self.non_distance_features
            
            # Calculate derived features
            if 'TOTAL_HEARTBEATS' in features:
                df['TOTAL_HEARTBEATS'] = df['HRAVG'] * df['DURATION_ACTUAL'] * 60
        
        # Common derived features
        if 'HRAVG' in df.columns and 'HRMAX' in df.columns:
            if 'INTENSITY_RATIO' in features:
                df['INTENSITY_RATIO'] = df['HRAVG'] / df['HRMAX']
            if 'HR_RESERVE' in features:
                df['HR_RESERVE'] = df['HRMAX'] - 70
            if 'HR_ZONE' in features:
                df['HR_ZONE'] = (df['HRAVG'] - 70) / df['HR_RESERVE']
            if 'HR_PERCENT_MAX' in features:
                df['HR_PERCENT_MAX'] = (df['HRAVG'] / df['HRMAX']) * 100
            if 'INTENSITY_DURATION' in features:
                df['INTENSITY_DURATION'] = (df['HRAVG'] / df['HRMAX']) * df['DURATION_ACTUAL']
        
        # Sex encoding
        if 'SEX_ENCODED' in features and 'SEX' in df.columns:
            df['SEX_ENCODED'] = (df['SEX'] == 'M').astype(int)
        
        # Select only required features
        return df[features]
    
    def predict_single(self, activity_type, duration, hr_avg, hr_max, 
                      distance=None, elevation_gain=None, age=None, sex=None):
        """
        Predict calories for a single workout.
        Automatically routes to appropriate model based on activity type.
        """
        # Identify activity category
        activity_category = self.identify_activity_type(activity_type)
        
        if activity_category == 'unknown':
            # Make best guess based on whether distance is provided
            if distance and distance > 0:
                activity_category = 'distance'
                print(f"Unknown activity '{activity_type}' - treating as distance activity")
            else:
                activity_category = 'non_distance'
                print(f"Unknown activity '{activity_type}' - treating as non-distance activity")
        
        # Prepare input data
        input_data = {
            'DURATION_ACTUAL': duration,
            'HRAVG': hr_avg,
            'HRMAX': hr_max
        }
        
        if distance is not None:
            input_data['DISTANCE_ACTUAL'] = distance
        if elevation_gain is not None:
            input_data['ELEVATIONGAIN'] = elevation_gain
        if age is not None:
            input_data['AGE'] = age
        if sex is not None:
            input_data['SEX'] = sex
        
        # Prepare features and predict
        if activity_category == 'distance':
            if self.distance_model is None:
                raise ValueError("No distance model available")
            
            X = self.prepare_features(input_data, 'distance')
            prediction = self.distance_model.predict(X)[0]
            model_used = self.model_info['distance_model_type']
            
        else:  # non_distance
            if self.non_distance_model is None:
                raise ValueError("No non-distance model available")
            
            X = self.prepare_features(input_data, 'non_distance')
            prediction = self.non_distance_model.predict(X)[0]
            model_used = self.model_info['non_distance_model_type']
        
        return {
            'calories': prediction,
            'activity_category': activity_category,
            'model_used': model_used,
            'features_used': X.columns.tolist()
        }
    
    def predict_batch(self, df):
        """
        Predict calories for multiple workouts.
        Automatically routes each to the appropriate model.
        """
        predictions = []
        
        for idx, row in df.iterrows():
            # Get activity type
            activity_type = row.get('ACTIVITY_TYPE', row.get('TYPE', 'Unknown'))
            
            # Prepare input
            pred_result = self.predict_single(
                activity_type=activity_type,
                duration=row['DURATION_ACTUAL'],
                hr_avg=row['HRAVG'],
                hr_max=row['HRMAX'],
                distance=row.get('DISTANCE_ACTUAL'),
                elevation_gain=row.get('ELEVATIONGAIN'),
                age=row.get('AGE'),
                sex=row.get('SEX')
            )
            
            predictions.append(pred_result['calories'])
        
        return np.array(predictions)
    
    def evaluate(self, df):
        """
        Evaluate model performance on a dataset.
        Shows performance broken down by activity category.
        """
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        
        # Get predictions
        predictions = self.predict_batch(df)
        actual = df['CALORIES'].values
        
        # Overall metrics
        overall_r2 = r2_score(actual, predictions)
        overall_mae = mean_absolute_error(actual, predictions)
        overall_rmse = np.sqrt(mean_squared_error(actual, predictions))
        
        print("="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        print(f"\nOverall Performance:")
        print(f"  R² Score: {overall_r2:.4f}")
        print(f"  MAE: {overall_mae:.1f} calories")
        print(f"  RMSE: {overall_rmse:.1f} calories")
        
        # Performance by activity category
        results = {}
        
        for activity in df['ACTIVITY_TYPE'].unique():
            activity_mask = df['ACTIVITY_TYPE'] == activity
            if activity_mask.sum() > 5:  # Need minimum samples
                activity_category = self.identify_activity_type(activity)
                
                act_pred = predictions[activity_mask]
                act_actual = actual[activity_mask]
                
                r2 = r2_score(act_actual, act_pred)
                mae = mean_absolute_error(act_actual, act_pred)
                
                results[activity] = {
                    'category': activity_category,
                    'n_samples': activity_mask.sum(),
                    'r2': r2,
                    'mae': mae
                }
        
        # Print by category
        print("\nDistance Activities:")
        print(f"{'Activity':<20} {'N':>8} {'R²':>8} {'MAE':>8}")
        print("-" * 50)
        for activity, metrics in sorted(results.items()):
            if metrics['category'] == 'distance':
                print(f"{activity:<20} {metrics['n_samples']:>8} "
                      f"{metrics['r2']:>8.4f} {metrics['mae']:>8.1f}")
        
        print("\nNon-Distance Activities:")
        print(f"{'Activity':<20} {'N':>8} {'R²':>8} {'MAE':>8}")
        print("-" * 50)
        for activity, metrics in sorted(results.items()):
            if metrics['category'] == 'non_distance':
                print(f"{activity:<20} {metrics['n_samples']:>8} "
                      f"{metrics['r2']:>8.4f} {metrics['mae']:>8.1f}")
        
        return {
            'overall': {'r2': overall_r2, 'mae': overall_mae, 'rmse': overall_rmse},
            'by_activity': results
        }
    
    def explain_prediction(self, activity_type, duration, hr_avg, hr_max,
                          distance=None, elevation_gain=None, age=None, sex=None):
        """
        Predict and explain which model and features were used.
        """
        result = self.predict_single(
            activity_type, duration, hr_avg, hr_max,
            distance, elevation_gain, age, sex
        )
        
        print("\n" + "="*60)
        print("PREDICTION EXPLANATION")
        print("="*60)
        
        print(f"\nActivity: {activity_type}")
        print(f"Category: {result['activity_category']}")
        print(f"Model Used: {result['model_used']}")
        
        if result['activity_category'] == 'distance':
            print(f"Model R²: {self.model_info['distance_performance']['r2']:.4f}")
            print("\nThis is a distance-based activity, so the model uses:")
            print("  - Distance and pace metrics")
            print("  - Elevation data")
            print("  - Heart rate efficiency relative to distance")
        else:
            print(f"Model R²: {self.model_info['non_distance_performance']['r2']:.4f}")
            print("\nThis is a non-distance activity, so the model uses:")
            print("  - Duration and heart rate intensity")
            print("  - Total heartbeats")
            print("  - Intensity-duration interaction")
            print("  (Distance/pace features excluded as not meaningful)")
        
        print(f"\nFeatures used ({len(result['features_used'])}):")
        for feat in result['features_used']:
            print(f"  - {feat}")
        
        print(f"\nPredicted Calories: {result['calories']:.0f}")
        
        return result


def demonstrate_usage():
    """
    Show how to use the combined model system.
    """
    print("="*60)
    print("SMART CALORIE PREDICTOR DEMONSTRATION")
    print("="*60)
    
    # Example predictions
    predictor = SmartCaloriePredictor()
    
    print("\nExample 1: Running (distance activity)")
    print("-" * 40)
    try:
        result = predictor.explain_prediction(
            activity_type="Run",
            duration=0.5,  # 30 minutes
            hr_avg=145,
            hr_max=165,
            distance=5000,  # 5km
            elevation_gain=50,
            age=30,
            sex='M'
        )
    except Exception as e:
        print(f"Model not loaded: {e}")
    
    print("\nExample 2: Yoga (non-distance activity)")
    print("-" * 40)
    try:
        result = predictor.explain_prediction(
            activity_type="Yoga",
            duration=1.0,  # 60 minutes
            hr_avg=95,
            hr_max=120,
            distance=0,  # No distance for yoga
            age=30,
            sex='F'
        )
    except Exception as e:
        print(f"Model not loaded: {e}")
    
    print("\nExample 3: Unknown activity with distance")
    print("-" * 40)
    try:
        result = predictor.explain_prediction(
            activity_type="Paddleboarding",
            duration=0.75,
            hr_avg=125,
            hr_max=145,
            distance=3000,  # Has distance, so treated as distance activity
            age=35
        )
    except Exception as e:
        print(f"Model not loaded: {e}")


if __name__ == "__main__":
    demonstrate_usage()