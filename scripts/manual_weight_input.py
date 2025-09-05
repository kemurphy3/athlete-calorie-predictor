#!/usr/bin/env python3
"""
Manual Weight Input System
Allows you to input your 15 weigh-ins in pounds and apply them to workout data
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Tuple, Dict

def create_weight_dataset() -> pd.DataFrame:
    """
    Create a weight dataset from your manual inputs.
    Replace the sample data with your actual weigh-ins!
    """
    # TODO: Replace these with your actual Garmin weigh-ins
    # Format: (date_str, weight_lbs)
    weigh_ins = [
        ('2020-01-15', 133.5),  # Replace with your actual data
        ('2020-03-20', 132.0),
        ('2020-06-15', 130.5),
        ('2020-09-10', 129.8),
        ('2020-12-05', 129.2),
        ('2021-02-20', 128.5),
        ('2021-05-15', 128.0),
        ('2021-08-10', 127.5),
        ('2021-11-05', 127.0),
        ('2022-01-20', 126.5),
        ('2022-04-15', 126.0),
        ('2022-07-10', 125.5),
        ('2022-10-05', 125.0),
        ('2023-01-20', 124.5),
        ('2023-04-15', 124.0),
        ('2023-07-10', 123.5),
        ('2023-10-05', 123.0),
        ('2024-01-20', 122.5),
        ('2024-04-15', 122.0),
        ('2024-07-10', 121.5),
        ('2024-10-05', 121.0),
        ('2025-01-20', 120.5),
        ('2025-04-15', 120.0),
        ('2025-07-10', 119.5),
        ('2025-09-04', 119.0)
    ]
    
    # Convert to DataFrame
    weight_df = pd.DataFrame(weigh_ins, columns=['date', 'weight_lbs'])
    weight_df['date'] = pd.to_datetime(weight_df['date'])
    
    # Convert pounds to kg (1 lb = 0.453592 kg)
    weight_df['weight_kg'] = weight_df['weight_lbs'] * 0.453592
    weight_df['weight_kg'] = weight_df['weight_kg'].round(2)
    
    return weight_df

def interpolate_weight_for_workouts(workout_df: pd.DataFrame, weight_df: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolate weight for each workout based on the weight dataset.
    """
    workout_df = workout_df.copy()
    workout_df['date'] = pd.to_datetime(workout_df['date'])
    
    # Sort weight data by date
    weight_df = weight_df.sort_values('date')
    
    def get_weight_for_date(target_date):
        """Get interpolated weight for a given date."""
        # If before first weigh-in, use first weight
        if target_date < weight_df['date'].min():
            return weight_df['weight_kg'].iloc[0]
            
        # If after last weigh-in, use last weight
        if target_date > weight_df['date'].max():
            return weight_df['weight_kg'].iloc[-1]
            
        # Find surrounding weigh-ins
        for i in range(len(weight_df) - 1):
            date1 = weight_df['date'].iloc[i]
            date2 = weight_df['date'].iloc[i + 1]
            weight1 = weight_df['weight_kg'].iloc[i]
            weight2 = weight_df['weight_kg'].iloc[i + 1]
            
            if date1 <= target_date <= date2:
                # Linear interpolation
                days_between = (date2 - date1).days
                days_to_target = (target_date - date1).days
                
                if days_between == 0:
                    return weight1
                    
                weight_diff = weight2 - weight1
                interpolated_weight = weight1 + (weight_diff * days_to_target / days_between)
                return round(interpolated_weight, 2)
                
        return weight_df['weight_kg'].iloc[-1]  # Fallback
    
    # Apply interpolation to each workout
    workout_df['weight_kg'] = workout_df['date'].apply(get_weight_for_date)
    
    return workout_df

def main():
    """Main function to process weight data."""
    print("Manual Weight Input System")
    print("=" * 50)
    
    # Step 1: Create weight dataset
    print("Creating weight dataset...")
    weight_df = create_weight_dataset()
    
    print(f"Weight data points: {len(weight_df)}")
    print(f"Weight range: {weight_df['weight_lbs'].min():.1f} to {weight_df['weight_lbs'].max():.1f} lbs")
    print(f"Weight range: {weight_df['weight_kg'].min():.2f} to {weight_df['weight_kg'].max():.2f} kg")
    
    # Show sample of weight data
    print("\nWeight data sample:")
    print(weight_df.head(10))
    
    # Step 2: Load workout data
    print("\nLoading workout data...")
    workout_df = pd.read_csv('data/workouts.csv')
    
    print(f"Original workouts: {len(workout_df)}")
    print(f"Original weight range: {workout_df['weight_kg'].min():.2f} to {workout_df['weight_kg'].max():.2f} kg")
    print(f"Original weight unique values: {workout_df['weight_kg'].nunique()}")
    
    # Step 3: Apply interpolated weights
    print("\nApplying interpolated weights...")
    workout_df_with_weight = interpolate_weight_for_workouts(workout_df, weight_df)
    
    print(f"New weight range: {workout_df_with_weight['weight_kg'].min():.2f} to {workout_df_with_weight['weight_kg'].max():.2f} kg")
    print(f"New weight unique values: {workout_df_with_weight['weight_kg'].nunique()}")
    
    # Step 4: Show weight over time
    print("\nWeight over time (sample):")
    weight_over_time = workout_df_with_weight[['date', 'weight_kg']].drop_duplicates().sort_values('date')
    print(weight_over_time.head(10))
    print("...")
    print(weight_over_time.tail(10))
    
    # Step 5: Save updated data
    output_file = 'data/workouts_with_varied_weight.csv'
    workout_df_with_weight.to_csv(output_file, index=False)
    print(f"\nSaved updated data to '{output_file}'")
    
    # Step 6: Show statistics
    print("\nWeight variation statistics:")
    print(f"  Weight change over time: {workout_df_with_weight['weight_kg'].max() - workout_df_with_weight['weight_kg'].min():.2f} kg")
    print(f"  Average weight: {workout_df_with_weight['weight_kg'].mean():.2f} kg")
    print(f"  Weight standard deviation: {workout_df_with_weight['weight_kg'].std():.2f} kg")
    
    return workout_df_with_weight

if __name__ == "__main__":
    main()
