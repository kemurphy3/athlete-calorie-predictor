# Model Explainer - Provides detailed explanations for model selection
# Explains why certain models perform better for specific activity types

import numpy as np
from typing import Dict, List, Tuple


class ModelExplainer:
    """
    Provides detailed explanations for why specific models were selected
    and their characteristics for different data types.
    """
    
    def __init__(self):
        self.model_characteristics = self._define_model_characteristics()
        self.performance_thresholds = self._define_performance_thresholds()
    
    def _define_model_characteristics(self):
        """
        Define strengths, weaknesses, and ideal use cases for each model type.
        """
        return {
            'linear': {
                'name': 'Linear Regression',
                'strengths': [
                    'Simple and interpretable',
                    'Fast training and prediction',
                    'Works well with linear relationships',
                    'No hyperparameter tuning needed',
                    'Resistant to overfitting with many features'
                ],
                'weaknesses': [
                    'Cannot capture non-linear patterns',
                    'Assumes linear relationship between features and target',
                    'Sensitive to outliers',
                    'Poor performance with complex interactions'
                ],
                'ideal_for': [
                    'Small datasets (<500 samples)',
                    'Linear relationships',
                    'When interpretability is crucial',
                    'Baseline model for comparison'
                ],
                'complexity': 'Low',
                'interpretability': 'High'
            },
            
            'ridge': {
                'name': 'Ridge Regression',
                'strengths': [
                    'Handles multicollinearity well',
                    'Reduces overfitting through L2 regularization',
                    'Stable with many features',
                    'Works well when all features are relevant'
                ],
                'weaknesses': [
                    'Still assumes linear relationships',
                    'Cannot perform feature selection',
                    'All features remain in the model'
                ],
                'ideal_for': [
                    'Datasets with correlated features',
                    'When preventing overfitting is important',
                    'High-dimensional data with all relevant features'
                ],
                'complexity': 'Low',
                'interpretability': 'High'
            },
            
            'lasso': {
                'name': 'Lasso Regression',
                'strengths': [
                    'Automatic feature selection',
                    'Produces sparse models',
                    'Good for high-dimensional data',
                    'Reduces model complexity'
                ],
                'weaknesses': [
                    'Can be unstable with correlated features',
                    'May arbitrarily select one feature from correlated group',
                    'Still limited to linear relationships'
                ],
                'ideal_for': [
                    'Feature selection needed',
                    'Sparse data with many irrelevant features',
                    'When model simplicity is important'
                ],
                'complexity': 'Low',
                'interpretability': 'High'
            },
            
            'elastic_net': {
                'name': 'Elastic Net',
                'strengths': [
                    'Combines Ridge and Lasso benefits',
                    'Handles correlated features better than Lasso',
                    'Can select groups of correlated features',
                    'Balanced regularization'
                ],
                'weaknesses': [
                    'More hyperparameters to tune',
                    'Still assumes linear relationships',
                    'Computational cost higher than simple linear models'
                ],
                'ideal_for': [
                    'Correlated features with need for selection',
                    'High-dimensional data',
                    'When both Ridge and Lasso have limitations'
                ],
                'complexity': 'Low-Medium',
                'interpretability': 'High'
            },
            
            'random_forest': {
                'name': 'Random Forest',
                'strengths': [
                    'Captures non-linear relationships',
                    'Handles interactions automatically',
                    'Robust to outliers',
                    'No feature scaling needed',
                    'Can handle mixed data types',
                    'Provides feature importance'
                ],
                'weaknesses': [
                    'Can overfit with small datasets',
                    'Slower prediction than linear models',
                    'Less interpretable (black box)',
                    'Memory intensive with many trees'
                ],
                'ideal_for': [
                    'Non-linear patterns',
                    'Mixed feature types',
                    'When feature interactions are important',
                    'Moderate to large datasets (>1000 samples)'
                ],
                'complexity': 'Medium-High',
                'interpretability': 'Medium'
            },
            
            'gradient_boost': {
                'name': 'Gradient Boosting',
                'strengths': [
                    'Often best performance on structured data',
                    'Handles non-linear patterns well',
                    'Sequential learning corrects errors',
                    'Good with heterogeneous features'
                ],
                'weaknesses': [
                    'Prone to overfitting',
                    'Sensitive to hyperparameters',
                    'Slower training than Random Forest',
                    'Sequential nature prevents parallelization'
                ],
                'ideal_for': [
                    'Competition-level accuracy needed',
                    'Complex non-linear patterns',
                    'When you have time for tuning'
                ],
                'complexity': 'High',
                'interpretability': 'Low-Medium'
            },
            
            'xgboost': {
                'name': 'XGBoost',
                'strengths': [
                    'State-of-the-art performance',
                    'Handles missing values automatically',
                    'Built-in regularization',
                    'Parallel processing',
                    'Excellent with non-linear patterns',
                    'Feature importance available'
                ],
                'weaknesses': [
                    'Many hyperparameters to tune',
                    'Can overfit on small datasets',
                    'Less interpretable',
                    'Memory intensive'
                ],
                'ideal_for': [
                    'Maximum predictive accuracy',
                    'Large datasets (>1000 samples)',
                    'Complex feature interactions',
                    'Mixed data types with missing values'
                ],
                'complexity': 'High',
                'interpretability': 'Low-Medium'
            },
            
            'lightgbm': {
                'name': 'LightGBM',
                'strengths': [
                    'Faster than XGBoost',
                    'Lower memory usage',
                    'Handles categorical features directly',
                    'Good with high-dimensional data',
                    'Leaf-wise growth for better accuracy'
                ],
                'weaknesses': [
                    'Can overfit on small datasets',
                    'Sensitive to hyperparameters',
                    'Less robust to outliers than XGBoost'
                ],
                'ideal_for': [
                    'Large datasets with many features',
                    'When training speed is important',
                    'Categorical features present',
                    'High-cardinality features'
                ],
                'complexity': 'High',
                'interpretability': 'Low-Medium'
            },
            
            'svr': {
                'name': 'Support Vector Regression',
                'strengths': [
                    'Effective in high-dimensional spaces',
                    'Robust to outliers',
                    'Flexible with kernel choice',
                    'Good generalization'
                ],
                'weaknesses': [
                    'Computationally expensive for large datasets',
                    'Sensitive to hyperparameters',
                    'Not suitable for large datasets',
                    'Black box model'
                ],
                'ideal_for': [
                    'Small to medium datasets (<5000 samples)',
                    'High-dimensional data',
                    'Non-linear patterns with RBF kernel',
                    'When outlier robustness is important'
                ],
                'complexity': 'Medium-High',
                'interpretability': 'Low'
            }
        }
    
    def _define_performance_thresholds(self):
        """
        Define what constitutes good/excellent performance.
        """
        return {
            'excellent': {'r2': 0.90, 'mae_ratio': 0.05},  # MAE < 5% of target range
            'very_good': {'r2': 0.85, 'mae_ratio': 0.08},
            'good': {'r2': 0.80, 'mae_ratio': 0.10},
            'acceptable': {'r2': 0.70, 'mae_ratio': 0.15},
            'poor': {'r2': 0.60, 'mae_ratio': 0.20}
        }
    
    def explain_model_selection(self, model_name: str, performance: Dict, 
                               data_characteristics: Dict) -> str:
        """
        Generate detailed explanation for why a model was selected.
        
        Args:
            model_name: Name of the selected model
            performance: Dict with 'r2', 'mae', 'rmse', 'cv_mean', 'cv_std'
            data_characteristics: Dict with dataset properties
        """
        model_info = self.model_characteristics.get(model_name, {})
        if not model_info:
            return f"Model {model_name} not recognized."
        
        explanation = []
        explanation.append("="*60)
        explanation.append(f"MODEL SELECTION EXPLANATION")
        explanation.append("="*60)
        
        # Model name and performance
        explanation.append(f"\n✓ Selected Model: {model_info['name']}")
        explanation.append(f"  R² Score: {performance['r2']:.4f}")
        explanation.append(f"  MAE: {performance['mae']:.1f} calories")
        explanation.append(f"  RMSE: {performance['rmse']:.1f} calories")
        if 'cv_mean' in performance:
            explanation.append(f"  Cross-validation R²: {performance['cv_mean']:.4f} ± {performance['cv_std']:.4f}")
        
        # Performance evaluation
        perf_level = self._evaluate_performance_level(performance['r2'])
        explanation.append(f"\n  Performance Level: {perf_level.upper()}")
        
        # Why this model excels
        explanation.append(f"\n📊 Why {model_info['name']} Excels Here:")
        
        # Analyze based on data characteristics
        reasons = self._analyze_model_fit(model_name, data_characteristics)
        for i, reason in enumerate(reasons, 1):
            explanation.append(f"  {i}. {reason}")
        
        # Model strengths relevant to this data
        explanation.append(f"\n💪 Key Strengths for This Dataset:")
        relevant_strengths = self._get_relevant_strengths(model_name, data_characteristics)
        for strength in relevant_strengths[:3]:  # Top 3 strengths
            explanation.append(f"  • {strength}")
        
        # Comparison with other models
        if 'comparison' in data_characteristics:
            explanation.append(f"\n📈 Performance Comparison:")
            explanation.append(self._generate_comparison(model_name, data_characteristics['comparison']))
        
        # Technical details
        explanation.append(f"\n🔧 Technical Details:")
        explanation.append(f"  • Complexity: {model_info['complexity']}")
        explanation.append(f"  • Interpretability: {model_info['interpretability']}")
        
        # Recommendations
        explanation.append(f"\n💡 Recommendations:")
        recommendations = self._generate_recommendations(model_name, performance, data_characteristics)
        for rec in recommendations:
            explanation.append(f"  • {rec}")
        
        return "\n".join(explanation)
    
    def _evaluate_performance_level(self, r2: float) -> str:
        """Evaluate performance level based on R² score."""
        for level, thresholds in self.performance_thresholds.items():
            if r2 >= thresholds['r2']:
                return level
        return 'poor'
    
    def _analyze_model_fit(self, model_name: str, data_chars: Dict) -> List[str]:
        """Analyze why this model fits the data well."""
        reasons = []
        
        n_samples = data_chars.get('n_samples', 0)
        n_features = data_chars.get('n_features', 0)
        activity_type = data_chars.get('activity_type', 'general')
        
        # Sample size analysis
        if model_name in ['xgboost', 'lightgbm', 'random_forest']:
            if n_samples > 500:
                reasons.append(f"Dataset size ({n_samples} samples) is ideal for tree-based ensemble methods")
            else:
                reasons.append(f"Despite smaller dataset ({n_samples} samples), regularization prevents overfitting")
        
        elif model_name in ['linear', 'ridge', 'lasso']:
            if n_samples < 500:
                reasons.append(f"Small dataset ({n_samples} samples) benefits from simple linear model")
            else:
                reasons.append(f"Linear relationship in data makes simple models competitive")
        
        # Feature analysis
        if activity_type == 'distance':
            if model_name in ['xgboost', 'gradient_boost']:
                reasons.append("Captures complex pace-distance-elevation interactions in endurance activities")
                reasons.append("Handles non-linear fatigue effects over distance")
            elif model_name == 'random_forest':
                reasons.append("Robust to outliers from GPS errors or unusual workouts")
        
        elif activity_type == 'non-distance':
            if model_name == 'random_forest':
                reasons.append("Handles diverse non-distance activities without assuming patterns")
                reasons.append("Captures intensity-duration interactions without distance confounding")
            elif model_name in ['xgboost', 'lightgbm']:
                reasons.append("Learns activity-specific calorie patterns from heart rate data")
        
        # Feature relationships
        if data_chars.get('has_nonlinear', False):
            if model_name in ['xgboost', 'lightgbm', 'random_forest', 'gradient_boost']:
                reasons.append("Automatically captures non-linear heart rate to calorie relationships")
        
        if data_chars.get('has_interactions', False):
            if model_name in ['xgboost', 'random_forest']:
                reasons.append("Models feature interactions (e.g., pace × elevation effects)")
        
        return reasons if reasons else ["Well-suited for this data structure"]
    
    def _get_relevant_strengths(self, model_name: str, data_chars: Dict) -> List[str]:
        """Get model strengths relevant to the dataset."""
        model_info = self.model_characteristics[model_name]
        strengths = model_info['strengths']
        
        # Prioritize strengths based on data characteristics
        relevant = []
        
        if data_chars.get('has_missing', False) and model_name == 'xgboost':
            relevant.append("Handles missing values automatically")
        
        if data_chars.get('has_outliers', False) and model_name == 'random_forest':
            relevant.append("Robust to outliers")
        
        if data_chars.get('activity_type') == 'distance' and model_name in ['xgboost', 'lightgbm']:
            relevant.append("Excellent with non-linear patterns")
        
        # Add remaining strengths
        for strength in strengths:
            if strength not in relevant:
                relevant.append(strength)
        
        return relevant
    
    def _generate_comparison(self, winner: str, comparison_data: Dict) -> str:
        """Generate comparison with other models."""
        lines = []
        
        # Sort by performance
        sorted_models = sorted(comparison_data.items(), 
                             key=lambda x: x[1].get('r2', 0) if x[1] else 0, 
                             reverse=True)
        
        for rank, (model, perf) in enumerate(sorted_models[:5], 1):
            if perf:
                indicator = "→" if model == winner else " "
                lines.append(f"  {rank}. {indicator} {model:15} R²={perf['r2']:.4f} MAE={perf['mae']:.1f}")
        
        return "\n".join(lines)
    
    def _generate_recommendations(self, model_name: str, performance: Dict, 
                                 data_chars: Dict) -> List[str]:
        """Generate recommendations for improving the model."""
        recs = []
        
        # Performance-based recommendations
        if performance['r2'] < 0.80:
            recs.append("Consider collecting more training data for better accuracy")
            if model_name in ['linear', 'ridge', 'lasso']:
                recs.append("Try ensemble methods (XGBoost/Random Forest) for non-linear patterns")
        
        if performance.get('cv_std', 0) > 0.05:
            recs.append("High variance in cross-validation suggests potential instability")
        
        # Model-specific recommendations
        if model_name in ['xgboost', 'lightgbm']:
            recs.append("Fine-tune hyperparameters for potential 2-3% improvement")
            if data_chars.get('n_samples', 0) < 500:
                recs.append("Monitor for overfitting with small dataset")
        
        elif model_name == 'random_forest':
            recs.append("Increase n_estimators if computational resources allow")
        
        elif model_name in ['linear', 'ridge']:
            recs.append("Consider polynomial features for non-linear relationships")
        
        # Data-specific recommendations
        if data_chars.get('activity_type') == 'distance':
            recs.append("Weather data could improve predictions for outdoor activities")
        elif data_chars.get('activity_type') == 'non-distance':
            recs.append("Activity-specific features (sets, reps) could improve accuracy")
        
        return recs if recs else ["Model is well-optimized for current data"]
    
    def compare_models(self, results: Dict, activity_type: str) -> str:
        """
        Generate comprehensive comparison and explanation for multiple models.
        """
        explanation = []
        explanation.append("\n" + "="*70)
        explanation.append(f" MODEL SELECTION ANALYSIS - {activity_type.upper()} ACTIVITIES ".center(70))
        explanation.append("="*70)
        
        # Find winner
        valid_results = {k: v for k, v in results.items() if v is not None}
        if not valid_results:
            return "No valid models to compare."
        
        winner = max(valid_results.items(), key=lambda x: x[1]['r2'])
        winner_name, winner_perf = winner
        
        # Generate detailed explanation for winner
        data_chars = {
            'activity_type': activity_type.lower().replace(' ', '_'),
            'n_samples': 742,  # Update with actual
            'n_features': 9,   # Update with actual
            'has_nonlinear': True,
            'has_interactions': True,
            'comparison': valid_results
        }
        
        winner_explanation = self.explain_model_selection(
            winner_name, winner_perf, data_chars
        )
        explanation.append(winner_explanation)
        
        # Explain why other models didn't win
        explanation.append("\n" + "="*60)
        explanation.append("WHY OTHER MODELS PERFORMED WORSE")
        explanation.append("="*60)
        
        for model_name, perf in sorted(valid_results.items(), 
                                      key=lambda x: x[1]['r2'], 
                                      reverse=True):
            if model_name != winner_name and perf:
                diff = winner_perf['r2'] - perf['r2']
                explanation.append(f"\n{self.model_characteristics[model_name]['name']}:")
                explanation.append(f"  R² = {perf['r2']:.4f} (−{diff:.4f} from best)")
                
                # Explain why it underperformed
                if model_name in ['linear', 'ridge', 'lasso', 'elastic_net']:
                    explanation.append("  • Limited by linear assumptions")
                    explanation.append("  • Cannot capture complex calorie burn patterns")
                elif model_name == 'svr':
                    explanation.append("  • Computational constraints with this dataset size")
                    explanation.append("  • Hyperparameter sensitivity")
                elif diff < 0.02:
                    explanation.append("  • Competitive performance, could be viable alternative")
                else:
                    explanation.append("  • Less suitable for this data structure")
        
        return "\n".join(explanation)


# Integrate into split_model_trainer
def enhance_split_trainer_with_explanations():
    """
    Enhancement to add explanation capability to split_model_trainer.
    """
    code = '''
    def explain_selection(self):
        """
        Explain why specific models were selected for each activity type.
        """
        from model_explainer import ModelExplainer
        explainer = ModelExplainer()
        
        print("\\n" + "="*70)
        print(" MODEL SELECTION EXPLANATIONS ".center(70))
        print("="*70)
        
        # Explain distance model selection
        if self.results.get('distance') and hasattr(self, 'best_distance_name'):
            distance_explanation = explainer.compare_models(
                self.results['distance'],
                'distance'
            )
            print(distance_explanation)
        
        # Explain non-distance model selection
        if self.results.get('non_distance') and hasattr(self, 'best_non_distance_name'):
            non_distance_explanation = explainer.compare_models(
                self.results['non_distance'],
                'non-distance'
            )
            print(non_distance_explanation)
        
        return explainer
    '''
    return code


if __name__ == "__main__":
    # Demo the explainer
    explainer = ModelExplainer()
    
    # Example explanation
    sample_performance = {
        'r2': 0.856,
        'mae': 45.2,
        'rmse': 62.3,
        'cv_mean': 0.842,
        'cv_std': 0.023
    }
    
    sample_data_chars = {
        'activity_type': 'distance',
        'n_samples': 742,
        'n_features': 9,
        'has_nonlinear': True,
        'has_interactions': True
    }
    
    explanation = explainer.explain_model_selection(
        'xgboost',
        sample_performance,
        sample_data_chars
    )
    
    print(explanation)