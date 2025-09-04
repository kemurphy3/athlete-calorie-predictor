# Ensemble Model for Mixed Activity Types
# Handles both distance-based (running, cycling) and stationary (yoga, weights) activities

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
import pickle
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class ActivityRouter:
    """
    Routes activities to appropriate model based on activity type.
    This is the simplest ensemble approach - train separate models for different activity types.
    """
    
    def __init__(self):
        self.distance_model = None
        self.stationary_model = None
        self.activity_types = {
            'distance': ['Run', 'Ride', 'Walk', 'Hike', 'Swim', 'VirtualRide'],
            'stationary': ['WeightTraining', 'Yoga', 'Workout', 'StairStepper', 'Elliptical']
        }
    
    def fit(self, X, y, activity_type):
        """Train separate models for different activity types"""
        
        # Split data by activity type
        distance_mask = activity_type.isin(self.activity_types['distance'])
        
        # Train distance model
        if distance_mask.any():
            X_distance = X[distance_mask]
            y_distance = y[distance_mask]
            
            self.distance_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
            self.distance_model.fit(X_distance, y_distance)
            print(f"Distance model trained on {len(X_distance)} samples")
        
        # Train stationary model
        if (~distance_mask).any():
            X_stationary = X[~distance_mask]
            y_stationary = y[~distance_mask]
            
            # Stationary model might need different hyperparameters
            self.stationary_model = xgb.XGBRegressor(
                n_estimators=80,
                max_depth=4,
                learning_rate=0.15,
                random_state=42
            )
            self.stationary_model.fit(X_stationary, y_stationary)
            print(f"Stationary model trained on {len(X_stationary)} samples")
    
    def predict(self, X, activity_type):
        """Route predictions to appropriate model"""
        predictions = np.zeros(len(X))
        
        distance_mask = activity_type.isin(self.activity_types['distance'])
        
        if distance_mask.any() and self.distance_model:
            predictions[distance_mask] = self.distance_model.predict(X[distance_mask])
        
        if (~distance_mask).any() and self.stationary_model:
            predictions[~distance_mask] = self.stationary_model.predict(X[~distance_mask])
        
        return predictions


class WeightedEnsemble:
    """
    Weighted average of multiple models.
    Learns optimal weights for combining predictions from different models.
    """
    
    def __init__(self, models=None):
        if models is None:
            # Default ensemble of diverse models
            self.models = {
                'xgboost': xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42),
                'lightgbm': lgb.LGBMRegressor(n_estimators=100, max_depth=5, random_state=42, verbose=-1),
                'random_forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
                'gradient_boost': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
            }
        else:
            self.models = models
        
        self.weights = None
        self.fitted_models = {}
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train all models and learn optimal weights.
        If validation set provided, uses it to determine weights.
        Otherwise uses out-of-fold predictions.
        """
        
        # Train each model
        print("Training ensemble models...")
        for name, model in self.models.items():
            print(f"  Training {name}...")
            model.fit(X_train, y_train)
            self.fitted_models[name] = model
        
        # Get predictions for weight optimization
        if X_val is not None and y_val is not None:
            # Use validation set
            predictions = self._get_predictions(X_val)
            self.weights = self._optimize_weights(predictions, y_val)
        else:
            # Use cross-validation on training set
            predictions = self._get_oof_predictions(X_train, y_train)
            self.weights = self._optimize_weights(predictions, y_train)
        
        print(f"\nOptimized weights:")
        for name, weight in zip(self.models.keys(), self.weights):
            print(f"  {name}: {weight:.3f}")
    
    def _get_predictions(self, X):
        """Get predictions from all models"""
        predictions = []
        for name, model in self.fitted_models.items():
            pred = model.predict(X)
            predictions.append(pred)
        return np.array(predictions).T
    
    def _get_oof_predictions(self, X, y):
        """Get out-of-fold predictions for weight optimization"""
        from sklearn.model_selection import KFold
        
        n_models = len(self.models)
        oof_predictions = np.zeros((len(X), n_models))
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for train_idx, val_idx in kf.split(X):
            X_train_fold = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
            y_train_fold = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
            X_val_fold = X.iloc[val_idx] if hasattr(X, 'iloc') else X[val_idx]
            
            for i, (name, model) in enumerate(self.models.items()):
                # Clone model to avoid overwriting
                model_clone = model.__class__(**model.get_params())
                model_clone.fit(X_train_fold, y_train_fold)
                oof_predictions[val_idx, i] = model_clone.predict(X_val_fold)
        
        return oof_predictions
    
    def _optimize_weights(self, predictions, y_true):
        """Find optimal weights using scipy optimization"""
        from scipy.optimize import minimize
        
        def loss(weights):
            # Ensure weights sum to 1
            weights = weights / weights.sum()
            weighted_pred = np.dot(predictions, weights)
            return mean_squared_error(y_true, weighted_pred)
        
        # Initial equal weights
        init_weights = np.ones(predictions.shape[1]) / predictions.shape[1]
        
        # Optimize
        bounds = [(0, 1) for _ in range(predictions.shape[1])]
        result = minimize(loss, init_weights, bounds=bounds, method='SLSQP')
        
        # Normalize to sum to 1
        optimal_weights = result.x / result.x.sum()
        return optimal_weights
    
    def predict(self, X):
        """Make weighted predictions"""
        predictions = self._get_predictions(X)
        weighted_pred = np.dot(predictions, self.weights)
        return weighted_pred


class StackingEnsemble:
    """
    Stacking ensemble (meta-learning).
    Uses predictions from base models as features for a meta-model.
    """
    
    def __init__(self, base_models=None, meta_model=None):
        if base_models is None:
            # Default base models
            self.base_models = {
                'xgboost': xgb.XGBRegressor(n_estimators=80, max_depth=4, random_state=42),
                'lightgbm': lgb.LGBMRegressor(n_estimators=80, max_depth=4, random_state=42, verbose=-1),
                'random_forest': RandomForestRegressor(n_estimators=80, max_depth=8, random_state=42)
            }
        else:
            self.base_models = base_models
        
        if meta_model is None:
            # Default meta-model (simpler to avoid overfitting)
            self.meta_model = xgb.XGBRegressor(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                random_state=42
            )
        else:
            self.meta_model = meta_model
        
        self.fitted_base_models = {}
        self.include_original_features = True  # Whether to include original features in meta-model
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Two-stage training:
        1. Train base models
        2. Train meta-model on base model predictions
        """
        
        # Stage 1: Train base models and get out-of-fold predictions
        print("Stage 1: Training base models...")
        oof_predictions = self._train_base_models(X_train, y_train)
        
        # Stage 2: Train meta-model
        print("\nStage 2: Training meta-model...")
        if self.include_original_features:
            # Include both original features and base model predictions
            meta_features = np.hstack([X_train, oof_predictions])
        else:
            # Use only base model predictions
            meta_features = oof_predictions
        
        self.meta_model.fit(meta_features, y_train)
        
        # Evaluate if validation set provided
        if X_val is not None and y_val is not None:
            val_predictions = self.predict(X_val)
            val_r2 = r2_score(y_val, val_predictions)
            val_mae = mean_absolute_error(y_val, val_predictions)
            print(f"\nValidation Performance:")
            print(f"  R² Score: {val_r2:.4f}")
            print(f"  MAE: {val_mae:.2f} calories")
    
    def _train_base_models(self, X, y):
        """Train base models and get out-of-fold predictions"""
        from sklearn.model_selection import KFold
        
        n_models = len(self.base_models)
        oof_predictions = np.zeros((len(X), n_models))
        
        # Use K-Fold to get out-of-fold predictions
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for i, (name, model) in enumerate(self.base_models.items()):
            print(f"  Training {name} with cross-validation...")
            model_oof = np.zeros(len(X))
            
            for train_idx, val_idx in kf.split(X):
                X_train_fold = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
                y_train_fold = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
                X_val_fold = X.iloc[val_idx] if hasattr(X, 'iloc') else X[val_idx]
                
                # Clone and train model
                model_clone = model.__class__(**model.get_params())
                model_clone.fit(X_train_fold, y_train_fold)
                
                # Predict on validation fold
                model_oof[val_idx] = model_clone.predict(X_val_fold)
            
            oof_predictions[:, i] = model_oof
            
            # Train final model on all data
            model.fit(X, y)
            self.fitted_base_models[name] = model
            
            # Report OOF performance
            oof_r2 = r2_score(y, model_oof)
            print(f"    {name} OOF R²: {oof_r2:.4f}")
        
        return oof_predictions
    
    def predict(self, X):
        """Make predictions using stacking ensemble"""
        # Get base model predictions
        base_predictions = []
        for name, model in self.fitted_base_models.items():
            pred = model.predict(X)
            base_predictions.append(pred)
        
        base_predictions = np.array(base_predictions).T
        
        # Create meta-features
        if self.include_original_features:
            meta_features = np.hstack([X, base_predictions])
        else:
            meta_features = base_predictions
        
        # Meta-model prediction
        return self.meta_model.predict(meta_features)


class HybridEnsemble:
    """
    Combines activity routing with ensemble methods.
    Routes to different ensembles based on activity type.
    """
    
    def __init__(self):
        # Use different ensemble strategies for different activity types
        self.distance_ensemble = WeightedEnsemble()
        self.stationary_ensemble = StackingEnsemble()
        self.activity_types = {
            'distance': ['Run', 'Ride', 'Walk', 'Hike', 'Swim', 'VirtualRide'],
            'stationary': ['WeightTraining', 'Yoga', 'Workout', 'StairStepper', 'Elliptical']
        }
    
    def fit(self, X, y, activity_type):
        """Train separate ensembles for each activity type"""
        
        # Split by activity type
        distance_mask = activity_type.isin(self.activity_types['distance'])
        
        # Train distance ensemble
        if distance_mask.any():
            X_distance = X[distance_mask]
            y_distance = y[distance_mask]
            print(f"\nTraining distance activity ensemble ({len(X_distance)} samples)...")
            self.distance_ensemble.fit(X_distance, y_distance)
        
        # Train stationary ensemble
        if (~distance_mask).any():
            X_stationary = X[~distance_mask]
            y_stationary = y[~distance_mask]
            print(f"\nTraining stationary activity ensemble ({len(X_stationary)} samples)...")
            self.stationary_ensemble.fit(X_stationary, y_stationary)
    
    def predict(self, X, activity_type):
        """Route to appropriate ensemble for predictions"""
        predictions = np.zeros(len(X))
        
        distance_mask = activity_type.isin(self.activity_types['distance'])
        
        if distance_mask.any():
            predictions[distance_mask] = self.distance_ensemble.predict(X[distance_mask])
        
        if (~distance_mask).any():
            predictions[~distance_mask] = self.stationary_ensemble.predict(X[~distance_mask])
        
        return predictions


def compare_ensemble_approaches(X, y, activity_type):
    """
    Compare different ensemble approaches on your data
    """
    from sklearn.model_selection import train_test_split
    
    # Split data
    X_train, X_test, y_train, y_test, act_train, act_test = train_test_split(
        X, y, activity_type, test_size=0.2, random_state=42
    )
    
    results = {}
    
    # 1. Single model baseline
    print("\n" + "="*60)
    print("BASELINE: Single XGBoost Model")
    print("="*60)
    baseline = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)
    results['baseline'] = {
        'r2': r2_score(y_test, baseline_pred),
        'mae': mean_absolute_error(y_test, baseline_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, baseline_pred))
    }
    print(f"R² Score: {results['baseline']['r2']:.4f}")
    print(f"MAE: {results['baseline']['mae']:.2f} calories")
    print(f"RMSE: {results['baseline']['rmse']:.2f} calories")
    
    # 2. Activity Router
    print("\n" + "="*60)
    print("APPROACH 1: Activity Router")
    print("="*60)
    router = ActivityRouter()
    router.fit(X_train, y_train, act_train)
    router_pred = router.predict(X_test, act_test)
    results['router'] = {
        'r2': r2_score(y_test, router_pred),
        'mae': mean_absolute_error(y_test, router_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, router_pred))
    }
    print(f"R² Score: {results['router']['r2']:.4f}")
    print(f"MAE: {results['router']['mae']:.2f} calories")
    print(f"RMSE: {results['router']['rmse']:.2f} calories")
    
    # 3. Weighted Ensemble
    print("\n" + "="*60)
    print("APPROACH 2: Weighted Ensemble")
    print("="*60)
    weighted = WeightedEnsemble()
    weighted.fit(X_train, y_train)
    weighted_pred = weighted.predict(X_test)
    results['weighted'] = {
        'r2': r2_score(y_test, weighted_pred),
        'mae': mean_absolute_error(y_test, weighted_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, weighted_pred))
    }
    print(f"R² Score: {results['weighted']['r2']:.4f}")
    print(f"MAE: {results['weighted']['mae']:.2f} calories")
    print(f"RMSE: {results['weighted']['rmse']:.2f} calories")
    
    # 4. Stacking Ensemble
    print("\n" + "="*60)
    print("APPROACH 3: Stacking Ensemble")
    print("="*60)
    stacking = StackingEnsemble()
    stacking.fit(X_train, y_train, X_test, y_test)
    stacking_pred = stacking.predict(X_test)
    results['stacking'] = {
        'r2': r2_score(y_test, stacking_pred),
        'mae': mean_absolute_error(y_test, stacking_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, stacking_pred))
    }
    print(f"R² Score: {results['stacking']['r2']:.4f}")
    print(f"MAE: {results['stacking']['mae']:.2f} calories")
    print(f"RMSE: {results['stacking']['rmse']:.2f} calories")
    
    # 5. Hybrid Ensemble
    print("\n" + "="*60)
    print("APPROACH 4: Hybrid Ensemble (Router + Ensembles)")
    print("="*60)
    hybrid = HybridEnsemble()
    hybrid.fit(X_train, y_train, act_train)
    hybrid_pred = hybrid.predict(X_test, act_test)
    results['hybrid'] = {
        'r2': r2_score(y_test, hybrid_pred),
        'mae': mean_absolute_error(y_test, hybrid_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, hybrid_pred))
    }
    print(f"R² Score: {results['hybrid']['r2']:.4f}")
    print(f"MAE: {results['hybrid']['mae']:.2f} calories")
    print(f"RMSE: {results['hybrid']['rmse']:.2f} calories")
    
    # Summary
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    print(f"{'Model':<20} {'R²':>8} {'MAE':>10} {'RMSE':>10}")
    print("-" * 50)
    for name, metrics in results.items():
        print(f"{name:<20} {metrics['r2']:>8.4f} {metrics['mae']:>10.2f} {metrics['rmse']:>10.2f}")
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['r2'])
    print(f"\nBest model: {best_model[0]} with R² = {best_model[1]['r2']:.4f}")
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Ensemble Model Training Script")
    print("This script provides multiple ensemble approaches for mixed activity types")
    print("\nAvailable approaches:")
    print("1. ActivityRouter - Separate models for distance vs stationary activities")
    print("2. WeightedEnsemble - Optimal weighted average of multiple models")
    print("3. StackingEnsemble - Meta-learning with base models as features")
    print("4. HybridEnsemble - Combines routing with ensemble methods")
    print("\nTo use, import the classes and call compare_ensemble_approaches(X, y, activity_type)")