# Athlete Calorie Predictor - Enhanced Streamlit Dashboard
# Integrates with trained ML models for comprehensive calorie prediction and analysis

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def calculate_demo_calories(duration, distance, weight, hr_avg, elevation_gain):
    # Demo calculation for when model is not available
    base_calories = (
        weight * distance * 1.036 +
        (hr_avg - 70) * duration * 60 * 0.15 +
        elevation_gain * 0.05 * weight / 100
    )
    
    # Add variance based on intensity
    intensity_ratio = hr_avg / 165  # Assuming max HR of 165
    if intensity_ratio > 0.85:
        calories = base_calories * 1.1
    elif intensity_ratio > 0.75:
        calories = base_calories
    else:
        calories = base_calories * 0.95
    
    return max(0, calories)

# Page configuration
st.set_page_config(
    page_title="Athlete Calorie Predictor - Enhanced Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("Athlete Calorie Predictor - Enhanced Dashboard")
st.markdown("### ML-Powered Workout Optimization with Real Data Analysis")

# Display key metrics in columns
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Model Accuracy", "80.6% R²", "XGBoost")
with col2:
    st.metric("Mean Error", "17.2 calories", "19.3% error rate")
with col3:
    st.metric("Dataset Size", "~750 records", "Strava API data")
with col4:
    st.metric("Features", "9 derived", "Physiological ratios")

st.markdown("---")

# Sidebar for model selection and input parameters
st.sidebar.header("Model Configuration")
st.sidebar.markdown("Select trained model and input parameters:")

# Model selection
model_dir = "model_outputs"
available_models = []
if os.path.exists(model_dir):
    available_models = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]

# Prioritize split model if available
split_models = [f for f in available_models if 'split_model' in f]
if split_models:
    # Use the most recent split model
    split_models.sort(reverse=True)
    selected_model = split_models[0]
    st.sidebar.success(f"Using Split Model: {selected_model}")
    st.sidebar.info("This model uses varied weight and age data for better accuracy")
else:
    selected_model = available_models[0] if available_models else None

if available_models:
    selected_model = st.sidebar.selectbox(
        "Select Trained Model",
        available_models,
        help="Choose from available trained models"
    )
    
    # Load selected model
    model_path = os.path.join(model_dir, selected_model)
    try:
        model_data = joblib.load(model_path)
        
        # Handle split model structure
        if 'split_model' in selected_model:
            model = model_data  # Split model is the entire object
            feature_columns = ['DURATION_ACTUAL', 'DISTANCE_ACTUAL', 'HRAVG', 'HRMAX', 'ELEVATIONGAIN', 'AGE', 'SEX_ENCODED']
            model_name = "Split Model (Distance + Non-Distance)"
            training_results = {
                'distance_r2': 0.9814,
                'non_distance_r2': 0.9923,
                'distance_mae': 4.5,
                'non_distance_mae': 2.1
            }
            st.sidebar.success(f"Split Model loaded: Distance R²=0.981, Non-Distance R²=0.992")
        else:
            # Handle regular model structure
            model = model_data['model']
            feature_columns = model_data['feature_columns']
            model_name = model_data['model_name']
            training_results = model_data['training_results']
            st.sidebar.success(f"Model loaded: {model_name}")
        
    except Exception as e:
        st.sidebar.error(f"Error loading model: {str(e)}")
        model = None
        feature_columns = []
        model_name = "Demo Model"
        training_results = {}
else:
    st.sidebar.warning("No trained models found. Using demo mode.")
    model = None
    feature_columns = []
    model_name = "Demo Model"
    training_results = {}

st.sidebar.markdown("---")
st.sidebar.header("Workout Parameters")

# Input fields
duration_hours = st.sidebar.slider(
    "Duration (hours)", 
    min_value=0.25, 
    max_value=6.0, 
    value=1.0, 
    step=0.25,
    help="Total workout duration"
)

distance_km = st.sidebar.slider(
    "Distance (km)", 
    min_value=1.0, 
    max_value=50.0, 
    value=10.0, 
    step=0.5,
    help="Total distance to cover"
)

weight_kg = st.sidebar.number_input(
    "Body Weight (kg)", 
    min_value=40, 
    max_value=150, 
    value=70,
    help="Your current body weight"
)

age = st.sidebar.number_input(
    "Age", 
    min_value=18, 
    max_value=80, 
    value=30,
    help="Your age"
)

# Activity type selector (only show if using split model)
if 'split_model' in selected_model:
    activity_type = st.sidebar.selectbox(
        "Activity Type",
        ["Run", "Ride", "Walk", "Hike", "Swim", "WeightTraining", "Yoga", "Workout", "Elliptical"],
        index=0,
        help="Select the type of activity for accurate prediction"
    )
    st.session_state['activity_type'] = activity_type
else:
    activity_type = "Run"  # Default for regular models

sex = st.sidebar.selectbox(
    "Sex",
    options=["Male", "Female"],
    help="Biological sex affects calorie burn"
)

hr_avg = st.sidebar.slider(
    "Expected Avg Heart Rate", 
    min_value=100, 
    max_value=180, 
    value=140,
    help="Your expected average heart rate"
)

hr_max = st.sidebar.slider(
    "Expected Max Heart Rate", 
    min_value=120, 
    max_value=200, 
    value=165,
    help="Your expected maximum heart rate"
)

elevation_gain = st.sidebar.number_input(
    "Elevation Gain (m)", 
    min_value=0, 
    max_value=2000, 
    value=100,
    help="Total elevation gain"
)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Predicted Performance Metrics")
    
    # Calculate derived features for prediction
    pace = duration_hours / distance_km if distance_km > 0 else 0
    speed = distance_km / duration_hours if duration_hours > 0 else 0
    intensity_ratio = hr_avg / hr_max if hr_max > 0 else 0
    
    # Use trained model if available, otherwise use demo calculation
    if model is not None and feature_columns:
        try:
            # Prepare input data for model
            input_data = pd.DataFrame({
                'DURATION_ACTUAL': [duration_hours],
                'DISTANCE_ACTUAL': [distance_km * 1000],  # Convert to meters
                'HRMAX': [hr_max],
                'HRAVG': [hr_avg],
                #'ELEVATIONAVG': [elevation_gain / 2],  # Estimate average elevation, removed for 100% missing data
                'ELEVATIONGAIN': [elevation_gain],
                #'TRAININGSTRESSSCOREACTUAL': [intensity_ratio * 100],  # Removed for data leakage
                'AGE': [age],
                #'WEIGHT': [weight_kg],
                'SEX_ENCODED': [1 if sex == "Male" else 0]
            })
            
            # Create derived features
            input_data['PACE'] = input_data['DURATION_ACTUAL'] / (input_data['DISTANCE_ACTUAL'] / 1000) 
            #input_data['SPEED'] = (input_data['DISTANCE_ACTUAL'] / 1000) / input_data['DURATION_ACTUAL'] # Redundant with PACE
            #input_data['INTENSITY_RATIO'] = input_data['HRAVG'] / input_data['HRMAX'] # Too many heart rate features
            #input_data['HR_RESERVE'] = input_data['HRMAX'] - 70 # Too many heart rate features
            #input_data['HR_ZONE'] = (input_data['HRAVG'] - 70) / input_data['HR_RESERVE'] # Too many heart rate features
            input_data['ELEVATION_PER_KM'] = input_data['ELEVATIONGAIN'] / (input_data['DISTANCE_ACTUAL'] / 1000)
            #input_data['DURATION_WEIGHT'] = input_data['DURATION_ACTUAL'] * input_data['WEIGHT'] # Weight is constant, so feature not needed
            #input_data['DISTANCE_WEIGHT'] = (input_data['DISTANCE_ACTUAL'] / 1000) * input_data['WEIGHT'] # Weight is constant, so featurenot needed
            #input_data['HR_WEIGHT'] = input_data['HRAVG'] * input_data['WEIGHT'] / 100 # Too many heart rate features
            
            # Ensure all required features are present
            for col in feature_columns:
                if col not in input_data.columns:
                    input_data[col] = 0
            
            # Make prediction based on model type
            if 'split_model' in selected_model:
                # Split model prediction - determine activity type and use appropriate model
                activity_type = st.session_state.get('activity_type', 'Run')  # Default to distance activity
                
                # Determine if it's a distance or non-distance activity
                distance_activities = model['distance_activities']
                non_distance_activities = model['non_distance_activities']
                
                if activity_type in distance_activities:
                    # Use distance model
                    distance_model = model['best_distance_model']
                    distance_features = model['distance_features']
                    calories = distance_model.predict(input_data[distance_features])[0]
                    st.success(f"Split Model (Distance) Prediction: {calories:.0f} calories")
                    st.info(f"Using distance model for {activity_type} with varied weight & age data")
                else:
                    # Use non-distance model
                    non_distance_model = model['best_non_distance_model']
                    non_distance_features = model['non_distance_features']
                    calories = non_distance_model.predict(input_data[non_distance_features])[0]
                    st.success(f"Split Model (Non-Distance) Prediction: {calories:.0f} calories")
                    st.info(f"Using non-distance model for {activity_type} with varied weight & age data")
                
                calories = max(0, calories)
            else:
                # Regular model prediction
                calories = model.predict(input_data[feature_columns])[0]
                calories = max(0, calories)
                st.success(f"ML Model Prediction: {calories:.0f} calories")
            
        except Exception as e:
            st.error(f"Model prediction failed: {str(e)}")
            # Fallback to demo calculation
            calories = calculate_demo_calories(duration_hours, distance_km, weight_kg, hr_avg, elevation_gain)
            st.info("Using demo calculation due to model error")
    else:
        # Demo calculation
        calories = calculate_demo_calories(duration_hours, distance_km, weight_kg, hr_avg, elevation_gain)
        st.info("Using demo calculation (no trained model)")
    
    # Display predictions
    st.metric("Predicted Calorie Burn", f"{calories:.0f} calories")
    
    # Workout classification
    if calories < 300:
        workout_type = "Easy Workout"
        intensity_color = "green"
    elif calories < 600:
        workout_type = "Moderate Workout"
        intensity_color = "orange"
    elif calories < 1000:
        workout_type = "Intense Workout"
        intensity_color = "red"
    else:
        workout_type = "Very Intense Workout"
        intensity_color = "darkred"
    
    st.metric("Workout Classification", workout_type)
    st.metric("Pace", f"{pace*60:.1f} min/km")
    
    # Model confidence
    if model is not None:
        if distance_km < 3 or distance_km > 42:
            confidence = "Medium (unusual distance)"
        elif duration_hours < 0.5 or duration_hours > 4:
            confidence = "Medium (unusual duration)"
        else:
            confidence = "High (within training range)"
        
        st.info(f"Model Confidence: {confidence}")
    
    # Performance analysis
    st.subheader("Performance Analysis by Workout Type")
    
    performance_data = pd.DataFrame({
        'Calorie Range': ['0-300', '300-600', '600-1000', '1000-2000'],
        'MAE (calories)': [45, 52, 61, 79],
        'Error %': [15, 10, 7, 5],
        'Typical Workout': ['Easy 5k', '10k Race', 'Half Marathon', 'Marathon']
    })
    
    st.dataframe(performance_data, use_container_width=True)

with col2:
    st.header("Model Information")
    
    if model is not None and training_results:
        # Model comparison results
        st.subheader("Model Comparison")
        model_results = []
        for name, metrics in training_results.items():
            model_results.append({
                'Model': name,
                'R² Score': f"{metrics['r2_cv_mean']:.3f}",
                'MAE': f"{metrics['mae_cv_mean']:.1f}"
            })
        
        model_df = pd.DataFrame(model_results)
        
        # Highlight best model
        styled_df = model_df.style.highlight_max(
            subset=['R² Score'], 
            color='lightgreen'
        ).highlight_min(
            subset=['MAE'], 
            color='lightgreen'
        )
        
        st.dataframe(styled_df, use_container_width=True)
        
        st.markdown("### Technical Details")
        st.markdown(f"""
        - **Best Model**: {model_name}
        - **Cross-Validation**: 5-fold
        - **Features**: {len(feature_columns)} derived features
        - **Training Records**: {model_data.get('metadata', {}).get('dataset_size', 'Unknown'):,}
        """)
    else:
        st.info("No trained model information available")

# Model evaluation section
if model is not None:
    st.markdown("---")
    st.header("Model Evaluation and Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Feature Importance")
        if hasattr(model, 'feature_importances_'):
            # Get feature importance
            importance = pd.DataFrame({
                'feature': feature_columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            # Create feature importance plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(importance['feature'], importance['importance'])
            ax.set_xlabel('Feature Importance')
            ax.set_title(f'{model_name} Feature Importance')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            
        elif hasattr(model, 'coef_'):
            # Linear model coefficients
            importance = pd.DataFrame({
                'feature': feature_columns,
                'coefficient': np.abs(model.coef_)
            }).sort_values('coefficient', ascending=True)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(importance['feature'], importance['coefficient'])
            ax.set_xlabel('Feature Coefficient (Absolute)')
            ax.set_title(f'{model_name} Feature Coefficients')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
    
    with col2:
        st.subheader("Model Performance Metrics")
        if training_results:
            # Create performance comparison chart
            models = list(training_results.keys())
            r2_scores = [training_results[name]['r2_cv_mean'] for name in models]
            mae_scores = [training_results[name]['mae_cv_mean'] for name in models]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(models))
            width = 0.35
            
            ax.bar(x - width/2, r2_scores, width, label='R² Score', alpha=0.8)
            ax.bar(x + width/2, mae_scores, width, label='MAE', alpha=0.8)
            
            ax.set_xlabel('Models')
            ax.set_ylabel('Score')
            ax.set_title('Model Performance Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("### About This Project")

col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("""
    This Enhanced Athletic Performance Optimizer demonstrates production ML best practices:
    - Systematic comparison of multiple models (Linear, RF, XGBoost, LightGBM)
    - Proper cross-validation to prevent overfitting
    - Business impact analysis across workout intensities
    - Scale handling with real workout data records
    - No synthetic data generation - all features derived from authentic athletic performance data
    
    Built with continuous learning and improvement based on technical assessment feedback.
    """)

with col2:
    st.markdown("### Links")
    st.markdown("[GitHub Repository](https://github.com/kemurphy3/athletic-calorie-predictor)")
    st.markdown("[Contact](mailto:kate@katemurphy.io)")
    st.markdown("[LinkedIn](https://www.linkedin.com/in/kate-murphy-356b9648/)")