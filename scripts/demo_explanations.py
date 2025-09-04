#!/usr/bin/env python3
# Demo script to show model selection explanations

from model_explainer import ModelExplainer
import numpy as np

def demo_explanations():
    """
    Demonstrate how the model explainer generates detailed explanations.
    """
    explainer = ModelExplainer()
    
    print("="*80)
    print(" CALORIE PREDICTOR - MODEL SELECTION EXPLANATION DEMO ".center(80))
    print("="*80)
    
    # Example 1: XGBoost wins for distance activities
    print("\n" + "="*70)
    print(" EXAMPLE 1: DISTANCE ACTIVITIES (Running, Cycling, etc.) ".center(70))
    print("="*70)
    
    # Simulated results from testing all models
    distance_results = {
        'linear': {'r2': 0.712, 'mae': 62.3, 'rmse': 85.4, 'cv_mean': 0.698, 'cv_std': 0.032},
        'ridge': {'r2': 0.718, 'mae': 60.8, 'rmse': 84.1, 'cv_mean': 0.705, 'cv_std': 0.029},
        'lasso': {'r2': 0.695, 'mae': 65.2, 'rmse': 88.3, 'cv_mean': 0.682, 'cv_std': 0.035},
        'elastic_net': {'r2': 0.709, 'mae': 62.9, 'rmse': 86.1, 'cv_mean': 0.696, 'cv_std': 0.031},
        'random_forest': {'r2': 0.832, 'mae': 48.5, 'rmse': 65.2, 'cv_mean': 0.818, 'cv_std': 0.024},
        'gradient_boost': {'r2': 0.845, 'mae': 46.2, 'rmse': 62.7, 'cv_mean': 0.831, 'cv_std': 0.022},
        'xgboost': {'r2': 0.856, 'mae': 44.8, 'rmse': 60.3, 'cv_mean': 0.842, 'cv_std': 0.021},
        'lightgbm': {'r2': 0.851, 'mae': 45.3, 'rmse': 61.2, 'cv_mean': 0.837, 'cv_std': 0.023},
        'svr': {'r2': 0.798, 'mae': 52.1, 'rmse': 71.4, 'cv_mean': 0.782, 'cv_std': 0.028}
    }
    
    # Winner: XGBoost
    winner = 'xgboost'
    winner_perf = distance_results[winner]
    
    data_characteristics = {
        'activity_type': 'distance',
        'n_samples': 542,  # Example: 542 distance activities
        'n_features': 11,  # pace, distance, elevation, HR, etc.
        'has_nonlinear': True,  # Non-linear calorie burn patterns
        'has_interactions': True,  # Pace-elevation interactions
        'has_outliers': True,  # GPS errors, unusual workouts
        'comparison': distance_results
    }
    
    explanation = explainer.explain_model_selection(
        winner,
        winner_perf,
        data_characteristics
    )
    print(explanation)
    
    # Example 2: Random Forest wins for non-distance activities
    print("\n" + "="*70)
    print(" EXAMPLE 2: NON-DISTANCE ACTIVITIES (Yoga, Weights, etc.) ".center(70))
    print("="*70)
    
    # Simulated results for non-distance activities
    non_distance_results = {
        'linear': {'r2': 0.685, 'mae': 38.2, 'rmse': 52.1, 'cv_mean': 0.671, 'cv_std': 0.041},
        'ridge': {'r2': 0.691, 'mae': 37.5, 'rmse': 51.3, 'cv_mean': 0.678, 'cv_std': 0.038},
        'lasso': {'r2': 0.672, 'mae': 39.8, 'rmse': 53.9, 'cv_mean': 0.658, 'cv_std': 0.044},
        'elastic_net': {'r2': 0.683, 'mae': 38.5, 'rmse': 52.5, 'cv_mean': 0.669, 'cv_std': 0.040},
        'random_forest': {'r2': 0.823, 'mae': 28.9, 'rmse': 39.2, 'cv_mean': 0.808, 'cv_std': 0.031},
        'gradient_boost': {'r2': 0.815, 'mae': 29.7, 'rmse': 40.1, 'cv_mean': 0.799, 'cv_std': 0.033},
        'xgboost': {'r2': 0.818, 'mae': 29.3, 'rmse': 39.8, 'cv_mean': 0.802, 'cv_std': 0.032},
        'lightgbm': {'r2': 0.812, 'mae': 30.1, 'rmse': 40.5, 'cv_mean': 0.796, 'cv_std': 0.034},
        'svr': {'r2': 0.765, 'mae': 33.2, 'rmse': 45.2, 'cv_mean': 0.748, 'cv_std': 0.038}
    }
    
    # Winner: Random Forest
    winner_nd = 'random_forest'
    winner_nd_perf = non_distance_results[winner_nd]
    
    data_characteristics_nd = {
        'activity_type': 'non_distance',
        'n_samples': 200,  # Example: 200 non-distance activities
        'n_features': 7,  # HR, duration, intensity (no distance features)
        'has_nonlinear': True,
        'has_interactions': True,
        'comparison': non_distance_results
    }
    
    explanation_nd = explainer.explain_model_selection(
        winner_nd,
        winner_nd_perf,
        data_characteristics_nd
    )
    print(explanation_nd)
    
    # Summary comparison
    print("\n" + "="*70)
    print(" KEY INSIGHTS ".center(70))
    print("="*70)
    
    print("""
📊 Model Selection Summary:

DISTANCE ACTIVITIES (Running, Cycling, Swimming):
• Winner: XGBoost (R² = 0.856)
• Why: Captures complex pace-distance-elevation interactions
• Key advantage: Handles non-linear fatigue effects over distance
• Runner-up: LightGBM (R² = 0.851) - faster but slightly less accurate

NON-DISTANCE ACTIVITIES (Yoga, Weights, Gym):
• Winner: Random Forest (R² = 0.823)
• Why: Robust to diverse activity patterns without distance confounding
• Key advantage: Handles mixed activity types without assumptions
• Runner-up: XGBoost (R² = 0.818) - marginally lower but still viable

🎯 Why Different Models for Different Activities?

1. FEATURE IMPORTANCE DIFFERS:
   - Distance activities: Pace and elevation crucial
   - Non-distance: Heart rate intensity and duration dominate

2. PATTERN COMPLEXITY:
   - Distance: Non-linear fatigue, elevation impact
   - Non-distance: Activity-specific burn rates

3. DATA CHARACTERISTICS:
   - Distance: More samples, GPS noise
   - Non-distance: Fewer samples, cleaner HR data

💡 Combined Model Advantage:
By using XGBoost for distance and Random Forest for non-distance,
the system achieves ~5-8% better accuracy than a single model approach.
    """)
    
    # Show how the comparison would look for all models
    print("\n" + "="*70)
    print(" FULL MODEL COMPARISON ".center(70))
    print("="*70)
    
    print("\nDISTANCE ACTIVITIES - All Models Ranked:")
    print("-"*60)
    print(f"{'Rank':<6} {'Model':<20} {'R²':<10} {'MAE':<10} {'Status'}")
    print("-"*60)
    
    # Sort distance results by R²
    sorted_distance = sorted(distance_results.items(), key=lambda x: x[1]['r2'], reverse=True)
    for i, (model, perf) in enumerate(sorted_distance, 1):
        status = "✓ SELECTED" if model == 'xgboost' else ""
        print(f"{i:<6} {explainer.model_characteristics[model]['name']:<20} "
              f"{perf['r2']:<10.4f} {perf['mae']:<10.1f} {status}")
    
    print("\nNON-DISTANCE ACTIVITIES - All Models Ranked:")
    print("-"*60)
    print(f"{'Rank':<6} {'Model':<20} {'R²':<10} {'MAE':<10} {'Status'}")
    print("-"*60)
    
    # Sort non-distance results by R²
    sorted_non_distance = sorted(non_distance_results.items(), key=lambda x: x[1]['r2'], reverse=True)
    for i, (model, perf) in enumerate(sorted_non_distance, 1):
        status = "✓ SELECTED" if model == 'random_forest' else ""
        print(f"{i:<6} {explainer.model_characteristics[model]['name']:<20} "
              f"{perf['r2']:<10.4f} {perf['mae']:<10.1f} {status}")


if __name__ == "__main__":
    demo_explanations()