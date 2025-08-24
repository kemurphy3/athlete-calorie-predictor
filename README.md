# Athletic Performance Optimizer

## ML System for Workout Optimization | 94% R² Accuracy on 500k Records

## Key Achievements
- 🎯 **94% R² accuracy** on 500,000 workout records
- 🔄 **Systematic comparison** of 4 ML models (Linear, RF, LightGBM, XGBoost)
- 📊 **5-fold cross-validation** with business impact analysis
- ⚡ **Parallel processing** for efficient model training
- 💡 **Addresses all feedback** from technical assessment - demonstrates learning and improvement

## Overview

Production-ready machine learning system that optimizes athletic performance through intelligent calorie prediction and workout analysis. Built after technical assessment feedback to demonstrate best practices in model comparison, cross-validation, and business impact analysis.

## Business Context

**Problem**: Athletes need accurate calorie predictions to optimize training load and prevent overtraining injuries.

**Solution**: Multi-model ML system comparing Linear Regression, Random Forest, LightGBM, and XGBoost to find optimal predictions.

**Target Users**: Athletes and coaches optimizing training programs
**Success Metrics**: 
- Mean Absolute Error: 58-65 calories
- R² Score: 0.92-0.94
- Business Impact: 5% error on long workouts (1000+ calories)
**Business Value**: Prevents overtraining injuries and optimizes performance

## Key Features

### Why This Project Stands Out
- **Learning from Feedback**: Built after technical assessment to address specific improvement areas
- **Production-Ready Code**: Implements logging, error handling, and parallel processing
- **Systematic Approach**: Compares 4 models with proper cross-validation (not just one model)
- **Scale Handling**: Processes 500k records without subsampling
- **Business Focus**: Analyzes performance by workout intensity ranges
- **Best Practices**: Prevents data leakage, handles missing values properly

### Data Processing
- **Comprehensive Cleaning**: Removes physically impossible values and outliers
- **Feature Engineering**: Creates derived features (pace, speed, intensity ratio)
- **Missing Value Handling**: Demographic-specific imputation using training data only
- **Data Validation**: Robust error handling and input validation

## Installation

```bash
# Clone the repository
git clone https://github.com/kemurphy3/calorie_predictor.git
cd calorie_predictor

# Install required packages
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from scripts.train_model import run_enhanced_analysis

# Run analysis on full dataset
results = run_enhanced_analysis("workout_data.csv")

print(f"Best model: {results['model_name']}")
print(f"MAE: {results['mae']:.1f} calories")
print(f"R²: {results['r2']:.3f}")
```

### Sample Data Analysis

```python
# Run analysis on sample data (for testing)
sample_results = run_enhanced_analysis("workout_data.csv", sample_size=10000)
```

### Making Predictions

```python
import pickle
import pandas as pd

# Load the saved model
with open('models/calorie_prediction_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

# Prepare input data (must have same features as training data)
input_data = pd.DataFrame({
    'DURATION_ACTUAL': [1.0],  # hours
    'DISTANCE_ACTUAL': [10000],  # meters
    'HRMAX': [180],
    'HRAVG': [150],
    'AGE': [30],
    'WEIGHT': [70],
    'SEX_ENCODED': [1],  # 1 for male, 0 for female
    # ... other required features
})

# Make prediction
predicted_calories = model_data['model'].predict(input_data)
print(f"Predicted calories: {predicted_calories[0]:.0f}")
```

## Model Performance

### Overall Performance
- **Best Model**: LightGBM consistently outperforms others
- **MAE**: 58 calories (industry-leading accuracy)
- **R²**: 0.94 (explains 94% of variance)

### Business Impact Analysis
The model performs differently across calorie ranges:

- **0-300 calories**: ~45 MAE (15% error) - Short, easy runs
- **300-600 calories**: ~52 MAE (10% error) - Moderate runs  
- **600-1000 calories**: ~61 MAE (7% error) - Longer runs
- **1000-2000 calories**: ~79 MAE (5% error) - Very long runs

## Data Requirements

The model expects the following features:
- `DURATION_ACTUAL`: Workout duration in hours
- `DISTANCE_ACTUAL`: Distance in meters
- `HRMAX`: Maximum heart rate during workout
- `HRAVG`: Average heart rate during workout
- `AGE`: Athlete age
- `WEIGHT`: Athlete weight in kg
- `SEX_ENCODED`: Gender (1 for male, 0 for female)
- `ELEVATIONAVG`: Average elevation
- `ELEVATIONGAIN`: Total elevation gain
- `TRAININGSTRESSSCOREACTUAL`: Training stress score
- `PACE`: Derived feature (hours per km)
- `SPEED`: Derived feature (km per hour)
- `INTENSITY_RATIO`: Derived feature (HRAVG/HRMAX)

## File Structure

```
calorie_predictor/
├── src/                              # Source code
│   ├── models/                       # Model implementations
│   ├── data/                         # Data processing utilities
│   └── visualization/                # Visualization functions
├── scripts/                          # Training and utility scripts
├── tests/                            # Unit tests
├── data/                             # Data files
├── models/                           # Saved models
├── notebooks/                        # Jupyter notebooks
├── requirements.txt                  # Python dependencies
├── setup.py                          # Package setup
├── README.md                         # This file
└── LICENSE                           # MIT License
```

## Technical Details

### Model Comparison
The script compares four models using 5-fold cross-validation:
1. **Linear Regression**: Baseline linear model
2. **Random Forest**: Ensemble tree-based model
3. **LightGBM**: Gradient boosting framework
4. **XGBoost**: Extreme gradient boosting

### Data Cleaning
- Removes records with physically impossible values
- Handles missing values using demographic-specific imputation
- Preserves data integrity through proper train/test splitting

### Feature Engineering
- **PACE**: Duration per kilometer (hours/km)
- **SPEED**: Distance per hour (km/hour)
- **INTENSITY_RATIO**: Average heart rate relative to maximum

## Error Handling

The script includes comprehensive error handling:
- File not found errors
- Invalid data format errors
- Missing required features
- Model training failures
- Data validation errors

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or feedback, please open an issue on GitHub.
