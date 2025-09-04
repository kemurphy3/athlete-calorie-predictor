#!/usr/bin/env python3
# Complete comparison script - runs all approaches and generates comprehensive report

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.split_model_trainer import ActivitySplitTrainer, load_and_prepare_data
from scripts.model_combiner import SmartCaloriePredictor
from scripts.ensemble_model import compare_ensemble_approaches
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


def run_complete_comparison():
    """
    Run all modeling approaches and compare results.
    """
    print("="*80)
    print(" COMPREHENSIVE MODEL COMPARISON FOR CALORIE PREDICTION ".center(80))
    print("="*80)
    print("\nThis will compare:")
    print("  1. Single model (baseline)")
    print("  2. Split models (distance vs non-distance with 9 model types each)")
    print("  3. Ensemble approaches (weighted, stacking, hybrid)")
    print("\n" + "="*80)
    
    # Load and prepare data
    print("\nSTEP 1: LOADING DATA")
    print("-"*40)
    df = load_and_prepare_data()
    
    if 'ACTIVITY_TYPE' not in df.columns:
        print("ERROR: No ACTIVITY_TYPE column found!")
        return None
    
    # Show data distribution
    print("\nActivity Distribution:")
    activity_counts = df['ACTIVITY_TYPE'].value_counts()
    for activity, count in activity_counts.head(10).items():
        print(f"  {activity}: {count}")
    if len(activity_counts) > 10:
        print(f"  ... and {len(activity_counts)-10} more activity types")
    
    # Categorize activities
    distance_activities = ['Run', 'Ride', 'Walk', 'Hike', 'Swim', 'VirtualRide']
    non_distance_activities = ['WeightTraining', 'Yoga', 'Workout', 'StairStepper', 'Elliptical']
    
    df['IS_DISTANCE'] = df['ACTIVITY_TYPE'].isin(distance_activities)
    df['IS_NON_DISTANCE'] = df['ACTIVITY_TYPE'].isin(non_distance_activities)
    
    print(f"\nCategorization:")
    print(f"  Distance activities: {df['IS_DISTANCE'].sum()} samples")
    print(f"  Non-distance activities: {df['IS_NON_DISTANCE'].sum()} samples")
    print(f"  Other/Unknown: {(~df['IS_DISTANCE'] & ~df['IS_NON_DISTANCE']).sum()} samples")
    
    # Split data for testing
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['IS_DISTANCE'])
    
    print(f"\nTrain/Test Split:")
    print(f"  Training: {len(train_df)} samples")
    print(f"  Testing: {len(test_df)} samples")
    
    results = {}
    
    # ========== APPROACH 1: BASELINE SINGLE MODEL ==========
    print("\n" + "="*80)
    print(" APPROACH 1: BASELINE SINGLE MODEL ".center(80))
    print("="*80)
    
    try:
        import xgboost as xgb
        
        # Prepare features (use all available)
        feature_cols = ['DURATION_ACTUAL', 'HRAVG', 'HRMAX', 'AGE']
        if 'DISTANCE_ACTUAL' in train_df.columns:
            feature_cols.append('DISTANCE_ACTUAL')
        if 'ELEVATIONGAIN' in train_df.columns:
            feature_cols.append('ELEVATIONGAIN')
        
        # Remove any with missing values
        feature_cols = [f for f in feature_cols if f in train_df.columns]
        
        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df['CALORIES']
        X_test = test_df[feature_cols].fillna(0)
        y_test = test_df['CALORIES']
        
        # Train baseline model
        baseline_model = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
        baseline_model.fit(X_train, y_train)
        
        # Evaluate
        baseline_pred = baseline_model.predict(X_test)
        baseline_r2 = r2_score(y_test, baseline_pred)
        baseline_mae = mean_absolute_error(y_test, baseline_pred)
        
        print(f"\nBaseline Single Model (XGBoost):")
        print(f"  R² Score: {baseline_r2:.4f}")
        print(f"  MAE: {baseline_mae:.1f} calories")
        
        results['baseline'] = {
            'r2': baseline_r2,
            'mae': baseline_mae,
            'model': 'XGBoost',
            'features': len(feature_cols)
        }
        
    except Exception as e:
        print(f"Baseline model failed: {e}")
        results['baseline'] = None
    
    # ========== APPROACH 2: SPLIT MODELS ==========
    print("\n" + "="*80)
    print(" APPROACH 2: SPLIT MODELS (DISTANCE VS NON-DISTANCE) ".center(80))
    print("="*80)
    
    try:
        # Train split models
        split_trainer = ActivitySplitTrainer()
        split_trainer.fit(train_df)
        
        # Count samples for explanations
        distance_mask = train_df['ACTIVITY_TYPE'].isin(split_trainer.distance_activities)
        non_distance_mask = train_df['ACTIVITY_TYPE'].isin(split_trainer.non_distance_activities)
        n_distance = distance_mask.sum()
        n_non_distance = non_distance_mask.sum()
        
        # Generate detailed explanations for model selection
        split_trainer.explain_selection(n_distance, n_non_distance)
        
        # Evaluate on test set
        split_pred = split_trainer.predict(test_df)
        
        # Filter out zeros (unpredicted activities)
        valid_mask = split_pred > 0
        if valid_mask.sum() > 0:
            split_r2 = r2_score(y_test[valid_mask], split_pred[valid_mask])
            split_mae = mean_absolute_error(y_test[valid_mask], split_pred[valid_mask])
            
            print(f"\nSplit Model Performance:")
            print(f"  R² Score: {split_r2:.4f}")
            print(f"  MAE: {split_mae:.1f} calories")
            print(f"  Coverage: {valid_mask.sum()}/{len(y_test)} samples")
            
            results['split'] = {
                'r2': split_r2,
                'mae': split_mae,
                'distance_model': getattr(split_trainer, 'best_distance_name', 'N/A'),
                'non_distance_model': getattr(split_trainer, 'best_non_distance_name', 'N/A'),
                'coverage': f"{valid_mask.sum()}/{len(y_test)}"
            }
            
            # Save split model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            split_model_path = f'../model_outputs/best_split_model_{timestamp}.pkl'
            os.makedirs('../model_outputs', exist_ok=True)
            split_trainer.save(split_model_path)
            
        else:
            print("Split model failed - no valid predictions")
            results['split'] = None
            
    except Exception as e:
        print(f"Split model failed: {e}")
        results['split'] = None
    
    # ========== APPROACH 3: ENSEMBLE MODELS ==========
    print("\n" + "="*80)
    print(" APPROACH 3: ENSEMBLE APPROACHES ".center(80))
    print("="*80)
    
    try:
        # Prepare features for ensemble
        ensemble_features = ['DURATION_ACTUAL', 'HRAVG', 'HRMAX']
        if 'DISTANCE_ACTUAL' in train_df.columns:
            ensemble_features.append('DISTANCE_ACTUAL')
        if 'ELEVATIONGAIN' in train_df.columns:
            ensemble_features.append('ELEVATIONGAIN')
        if 'AGE' in train_df.columns and train_df['AGE'].std() > 0:
            ensemble_features.append('AGE')
        
        X_train_ens = train_df[ensemble_features].fillna(0)
        y_train_ens = train_df['CALORIES']
        activity_train = train_df['ACTIVITY_TYPE']
        
        X_test_ens = test_df[ensemble_features].fillna(0)
        y_test_ens = test_df['CALORIES']
        activity_test = test_df['ACTIVITY_TYPE']
        
        # Run ensemble comparison
        ensemble_results = compare_ensemble_approaches(
            pd.concat([X_train_ens, X_test_ens]),
            pd.concat([y_train_ens, y_test_ens]),
            pd.concat([activity_train, activity_test])
        )
        
        # Store best ensemble
        best_ensemble = max(ensemble_results.items(), key=lambda x: x[1]['r2'] if x[1] else 0)
        results['ensemble'] = {
            'best_type': best_ensemble[0],
            'r2': best_ensemble[1]['r2'],
            'mae': best_ensemble[1]['mae'],
            'all_results': ensemble_results
        }
        
    except Exception as e:
        print(f"Ensemble comparison failed: {e}")
        results['ensemble'] = None
    
    # ========== FINAL COMPARISON ==========
    print("\n" + "="*80)
    print(" FINAL COMPARISON ".center(80))
    print("="*80)
    
    print("\nSummary of All Approaches:")
    print("-"*60)
    print(f"{'Approach':<25} {'R² Score':>12} {'MAE':>15}")
    print("-"*60)
    
    if results['baseline']:
        print(f"{'1. Baseline (Single)':<25} {results['baseline']['r2']:>12.4f} "
              f"{results['baseline']['mae']:>15.1f}")
    
    if results['split']:
        print(f"{'2. Split Models':<25} {results['split']['r2']:>12.4f} "
              f"{results['split']['mae']:>15.1f}")
        print(f"   Distance: {results['split']['distance_model']}")
        print(f"   Non-distance: {results['split']['non_distance_model']}")
    
    if results['ensemble']:
        print(f"{'3. Ensemble':<25} {results['ensemble']['r2']:>12.4f} "
              f"{results['ensemble']['mae']:>15.1f}")
        print(f"   Best: {results['ensemble']['best_type']}")
    
    # Find overall best
    valid_results = {k: v for k, v in results.items() if v and 'r2' in v}
    if valid_results:
        best_approach = max(valid_results.items(), key=lambda x: x[1]['r2'])
        print("\n" + "="*60)
        print(f"WINNER: {best_approach[0].upper()} approach")
        print(f"R² Score: {best_approach[1]['r2']:.4f}")
        print(f"MAE: {best_approach[1]['mae']:.1f} calories")
        
        # Calculate improvement over baseline
        if results['baseline'] and best_approach[0] != 'baseline':
            improvement = ((best_approach[1]['r2'] - results['baseline']['r2']) / 
                          results['baseline']['r2'] * 100)
            print(f"Improvement over baseline: {improvement:.1f}%")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f'../model_outputs/comparison_results_{timestamp}.pkl'
    with open(results_path, 'wb') as f:
        import pickle
        pickle.dump(results, f)
    
    print(f"\nResults saved to: {results_path}")
    
    # Generate visualization if possible
    try:
        generate_comparison_plot(results, timestamp)
    except Exception as e:
        print(f"Could not generate plot: {e}")
    
    return results


def generate_comparison_plot(results, timestamp):
    """
    Generate a visual comparison of model performances.
    """
    import matplotlib.pyplot as plt
    
    # Prepare data for plotting
    approaches = []
    r2_scores = []
    mae_scores = []
    
    if results['baseline']:
        approaches.append('Baseline')
        r2_scores.append(results['baseline']['r2'])
        mae_scores.append(results['baseline']['mae'])
    
    if results['split']:
        approaches.append('Split Models')
        r2_scores.append(results['split']['r2'])
        mae_scores.append(results['split']['mae'])
    
    if results['ensemble']:
        approaches.append(f"Ensemble\n({results['ensemble']['best_type']})")
        r2_scores.append(results['ensemble']['r2'])
        mae_scores.append(results['ensemble']['mae'])
    
    if not approaches:
        return
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # R² Score comparison
    colors = ['#ff7f0e', '#2ca02c', '#d62728']
    bars1 = ax1.bar(approaches, r2_scores, color=colors[:len(approaches)])
    ax1.set_ylabel('R² Score', fontsize=12)
    ax1.set_title('Model Performance - R² Score', fontsize=14)
    ax1.set_ylim([min(r2_scores) * 0.95, min(1.0, max(r2_scores) * 1.05)])
    
    # Add value labels on bars
    for bar, score in zip(bars1, r2_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.4f}', ha='center', va='bottom')
    
    # MAE comparison
    bars2 = ax2.bar(approaches, mae_scores, color=colors[:len(approaches)])
    ax2.set_ylabel('Mean Absolute Error (calories)', fontsize=12)
    ax2.set_title('Model Performance - MAE', fontsize=14)
    ax2.set_ylim([0, max(mae_scores) * 1.1])
    
    # Add value labels on bars
    for bar, score in zip(bars2, mae_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.1f}', ha='center', va='bottom')
    
    plt.suptitle('Calorie Prediction Model Comparison', fontsize=16)
    plt.tight_layout()
    
    # Save plot
    plot_path = f'../model_outputs/comparison_plot_{timestamp}.png'
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    print(f"Comparison plot saved to: {plot_path}")
    plt.close()


if __name__ == "__main__":
    print("\nCALORIE PREDICTOR - COMPLETE MODEL COMPARISON")
    print("="*80)
    print("\nThis script will:")
    print("  1. Train a baseline single model")
    print("  2. Train split models (best of 9 types for each activity category)")
    print("  3. Compare 4 ensemble approaches")
    print("  4. Determine the overall best approach")
    print("\nThis may take several minutes...\n")
    
    results = run_complete_comparison()
    
    if results:
        print("\n" + "="*80)
        print(" COMPARISON COMPLETE ".center(80))
        print("="*80)
        print("\nThe best model configuration has been identified and saved.")
        print("You can now use the split_model_trainer.py with the optimal models.")
    else:
        print("\nComparison failed. Please check your data.")