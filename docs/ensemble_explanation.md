# Ensemble Models for Calorie Prediction

## Overview
Ensemble methods combine multiple models to create a stronger predictor. For your calorie predictor with mixed activity types (running vs yoga), ensembles can significantly improve accuracy.

## Why Ensemble for Mixed Activities?

Your dataset has two fundamentally different activity types:
- **Distance-based**: Running, cycling, swimming (use distance, pace, elevation)
- **Stationary**: Yoga, weights, elliptical (no meaningful distance data)

A single model struggles because:
- Distance features are meaningless for yoga
- Stationary activities have different calorie burn patterns
- Feature importance varies by activity type

## Ensemble Approaches Implemented

### 1. Activity Router (Simplest)
```python
# Trains separate models for each activity type
distance_model = XGBoost()  # For runs, rides
stationary_model = XGBoost()  # For yoga, weights
```

**Pros:**
- Simple to understand and debug
- Each model optimized for its activity type
- No feature pollution between types

**Cons:**
- Requires sufficient data for each type
- No knowledge sharing between models

**When to use:** When you have clear activity categories and enough data for each

### 2. Weighted Ensemble
```python
# Combines predictions from multiple models with optimal weights
prediction = w1*xgboost + w2*lightgbm + w3*random_forest
```

**Pros:**
- Reduces overfitting through model diversity
- Automatically finds optimal combination
- Works well with limited data

**Cons:**
- All models see all data (including irrelevant features)
- More computational overhead

**When to use:** When you want robust predictions across all activities

### 3. Stacking Ensemble (Meta-Learning)
```python
# Uses base model predictions as features for a meta-model
base_predictions = [xgb.predict(), lgb.predict(), rf.predict()]
final_prediction = meta_model.predict(base_predictions + original_features)
```

**Pros:**
- Can learn complex interactions between models
- Often gives best performance
- Meta-model learns when to trust each base model

**Cons:**
- Risk of overfitting with small datasets
- More complex to tune
- Requires careful cross-validation

**When to use:** When you have sufficient data and want maximum accuracy

### 4. Hybrid Ensemble
```python
# Routes to different ensembles based on activity
if activity == 'distance':
    use WeightedEnsemble()
else:
    use StackingEnsemble()
```

**Pros:**
- Best of both worlds
- Optimized for each activity type
- Most flexible approach

**Cons:**
- Most complex to implement
- Requires more hyperparameter tuning

**When to use:** When different activity types benefit from different ensemble strategies

## Implementation Details

### Feature Handling

For mixed activities, features are handled differently:

```python
# Universal features (all activities)
universal = ['duration', 'hr_avg', 'hr_max', 'age']

# Distance-specific features
distance_only = ['distance', 'pace', 'elevation_gain']

# For stationary activities, distance features = 0 or excluded
```

### Cross-Validation Strategy

Critical for ensemble training:
1. **Out-of-Fold (OOF) Predictions**: Prevents overfitting when training meta-models
2. **Stratified by Activity**: Ensures each fold has representative activities
3. **Time-based splits**: For temporal data patterns

### Weight Optimization

For weighted ensemble, uses scipy optimization:
```python
def optimize_weights(predictions, y_true):
    # Minimize MSE with constraint: sum(weights) = 1
    return optimal_weights
```

## Expected Performance Improvements

Based on typical results:
- **Single Model**: R² = 0.806 (your current)
- **Activity Router**: R² = 0.82-0.84 (2-4% improvement)
- **Weighted Ensemble**: R² = 0.83-0.85 (3-5% improvement)
- **Stacking**: R² = 0.84-0.87 (4-8% improvement)
- **Hybrid**: R² = 0.85-0.88 (5-9% improvement)

## Practical Recommendations

### For Your Dataset (~1000 samples)

1. **Start with Activity Router**
   - Simplest to implement and understand
   - Clear separation of activity types
   - Easy to debug and explain

2. **Then try Weighted Ensemble**
   - Good balance of complexity and performance
   - Works well with your sample size
   - Robust to outliers

3. **Consider Stacking if:**
   - You need maximum accuracy
   - You have time for tuning
   - Model interpretability isn't critical

### Feature Engineering Tips

1. **Activity-Specific Features**:
   ```python
   if activity == 'Run':
       features += ['cadence', 'stride_length']
   elif activity == 'WeightTraining':
       features += ['rest_periods', 'set_count']
   ```

2. **Smart Null Handling**:
   ```python
   # Don't penalize yoga for having 0 distance
   df['pace'] = np.where(is_distance_activity, 
                         duration/distance, 
                         0)
   ```

3. **Interaction Features**:
   ```python
   # Only for relevant activities
   df['intensity_duration'] = df['hr_avg'] * df['duration']
   ```

## Running the Code

```bash
# Train and compare all ensemble approaches
cd /home/nerds/Murph/calorie_predictor
python3 scripts/train_ensemble.py

# This will:
# 1. Load data with activity types
# 2. Train all 4 ensemble approaches
# 3. Compare performance
# 4. Save the best model
```

## Next Steps

1. **Immediate**: Run ensemble comparison to see which works best
2. **Data Collection**: Add more stationary activities for better routing
3. **Feature Engineering**: Create activity-specific features
4. **Production**: Implement the best approach in your Streamlit app

## Key Takeaways

- Ensemble methods are powerful for mixed data types
- Activity routing is simple and effective
- Stacking gives best performance but needs more data
- Start simple (router) and increase complexity as needed
- Always validate with proper cross-validation

## Questions to Consider

1. Do you have enough samples per activity type? (Minimum ~50 each)
2. Is interpretability important? (Router is most interpretable)
3. What's your accuracy vs complexity tradeoff?
4. Will you add new activity types later? (Router is most extensible)

The ensemble approach will help your model understand that calories burned during 30 minutes of running follows different patterns than 30 minutes of yoga, leading to more accurate predictions across all activity types.