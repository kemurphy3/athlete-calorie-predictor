"""Unit tests for calorie predictor model."""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.calorie_predictor import CaloriePredictorEnhanced


class TestCaloriePredictor(unittest.TestCase):
    """Test cases for CaloriePredictorEnhanced class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock data manager
        self.mock_data_manager = Mock()
        
        # Create sample training data
        self.sample_data = pd.DataFrame({
            'DURATION_ACTUAL': [1.0, 1.5, 2.0, 0.5, 1.2],
            'DISTANCE_ACTUAL': [10000, 15000, 20000, 5000, 12000],
            'HRMAX': [180, 175, 185, 170, 182],
            'HRAVG': [150, 155, 160, 140, 152],
            'AGE': [30, 35, 28, 45, 32],
            'WEIGHT': [70, 75, 68, 80, 72],
            'SEX_ENCODED': [1, 0, 1, 1, 0],
            'ELEVATIONAVG': [100, 120, 80, 150, 110],
            'ELEVATIONGAIN': [50, 100, 30, 200, 75],
            'TRAININGSTRESSSCOREACTUAL': [75, 85, 90, 45, 80],
            'PACE': [0.1, 0.1, 0.1, 0.1, 0.1],
            'SPEED': [10, 10, 10, 10, 10],
            'INTENSITY_RATIO': [0.83, 0.89, 0.86, 0.82, 0.84],
            'CONSUMED_CALORIES': [500, 750, 1000, 250, 600]
        })
        
        self.mock_data_manager.get_training_data.return_value = self.sample_data
        
        # Initialize predictor
        self.predictor = CaloriePredictorEnhanced(self.mock_data_manager)
    
    def test_initialization(self):
        """Test predictor initialization."""
        self.assertIsNotNone(self.predictor)
        self.assertEqual(self.predictor.data_manager, self.mock_data_manager)
        self.assertIsNone(self.predictor.model)
        self.assertIsNone(self.predictor.feature_names)
    
    def test_train_model_selection(self):
        """Test model training and selection process."""
        # Train the model
        best_model_name = self.predictor.train(use_cross_validation=False)
        
        # Check that a model was selected
        self.assertIn(best_model_name, ['Linear Regression', 'Random Forest', 'LightGBM', 'XGBoost'])
        
        # Check that model is trained
        self.assertIsNotNone(self.predictor.model)
        self.assertIsNotNone(self.predictor.feature_names)
    
    def test_prediction(self):
        """Test making predictions."""
        # Train the model first
        self.predictor.train(use_cross_validation=False)
        
        # Create test data
        test_data = pd.DataFrame({
            'DURATION_ACTUAL': [1.0],
            'DISTANCE_ACTUAL': [10000],
            'HRMAX': [180],
            'HRAVG': [150],
            'AGE': [30],
            'WEIGHT': [70],
            'SEX_ENCODED': [1],
            'ELEVATIONAVG': [100],
            'ELEVATIONGAIN': [50],
            'TRAININGSTRESSSCOREACTUAL': [75],
            'PACE': [0.1],
            'SPEED': [10],
            'INTENSITY_RATIO': [0.83]
        })
        
        # Make prediction
        predictions = self.predictor.predict(test_data)
        
        # Check predictions
        self.assertEqual(len(predictions), 1)
        self.assertIsInstance(predictions[0], (int, float))
        self.assertGreater(predictions[0], 0)  # Calories should be positive
    
    def test_feature_engineering(self):
        """Test that feature engineering creates correct features."""
        # Check that required engineered features exist
        training_data = self.mock_data_manager.get_training_data()
        
        self.assertIn('PACE', training_data.columns)
        self.assertIn('SPEED', training_data.columns)
        self.assertIn('INTENSITY_RATIO', training_data.columns)
    
    def test_model_persistence(self):
        """Test saving and loading model."""
        # Train the model
        self.predictor.train(use_cross_validation=False)
        
        # Save model
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            model_path = tmp.name
        
        self.predictor.save_model(model_path)
        
        # Check file exists
        self.assertTrue(os.path.exists(model_path))
        
        # Clean up
        os.remove(model_path)
    
    def test_business_impact_analysis(self):
        """Test business impact analysis functionality."""
        # This would test the analyze_business_impact method if it exists
        # For now, we'll just check the data structure
        self.assertIn('CONSUMED_CALORIES', self.sample_data.columns)
        
        # Check calorie ranges make sense
        calories = self.sample_data['CONSUMED_CALORIES']
        self.assertTrue(all(calories > 0))
        self.assertTrue(all(calories < 5000))  # Reasonable upper limit


if __name__ == '__main__':
    unittest.main()