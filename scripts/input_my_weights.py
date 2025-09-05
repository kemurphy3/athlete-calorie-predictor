#!/usr/bin/env python3
"""
Interactive Weight Input
Allows you to input your actual 15 weigh-ins from Garmin
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Tuple

def input_weigh_ins() -> List[Tuple[str, float]]:
    """
    Interactive function to input your actual weigh-ins.
    """
    print("Enter your actual Garmin weigh-ins:")
    print("Format: YYYY-MM-DD,weight_in_pounds")
    print("Example: 2020-01-15,133.5")
    print("Press Enter with empty input when done")
    print("-" * 50)
    
    weigh_ins = []
    
    while True:
        try:
            user_input = input("Enter weigh-in (date,weight_lbs) or press Enter to finish: ").strip()
            
            if not user_input:
                break
                
            if ',' not in user_input:
                print("Please use format: YYYY-MM-DD,weight_in_pounds")
                continue
                
            date_str, weight_str = user_input.split(',', 1)
            weight_lbs = float(weight_str.strip())
            
            # Validate date format
            try:
                pd.to_datetime(date_str)
            except:
                print("Invalid date format. Use YYYY-MM-DD")
                continue
                
            weigh_ins.append((date_str, weight_lbs))
            print(f"Added: {date_str} = {weight_lbs} lbs")
            
        except ValueError:
            print("Invalid weight. Please enter a number.")
            continue
        except KeyboardInterrupt:
            print("\nExiting...")
            break
    
    return weigh_ins

def create_weight_dataset_from_input(weigh_ins: List[Tuple[str, float]]) -> pd.DataFrame:
    """
    Create weight dataset from user input.
    """
    if not weigh_ins:
        print("No weigh-ins provided. Using sample data.")
        return create_sample_weight_dataset()
    
    # Convert to DataFrame
    weight_df = pd.DataFrame(weigh_ins, columns=['date', 'weight_lbs'])
    weight_df['date'] = pd.to_datetime(weight_df['date'])
    
    # Convert pounds to kg (1 lb = 0.453592 kg)
    weight_df['weight_kg'] = weight_df['weight_lbs'] * 0.453592
    weight_df['weight_kg'] = weight_df['weight_kg'].round(2)
    
    # Sort by date
    weight_df = weight_df.sort_values('date')
    
    return weight_df

def create_sample_weight_dataset() -> pd.DataFrame:
    """
    Create sample weight dataset if no input provided.
    """
    sample_weigh_ins = [
        ('2020-01-15', 133.5),
        ('2020-06-15', 130.5),
        ('2020-12-05', 129.2),
        ('2021-06-15', 128.0),
        ('2021-12-05', 127.0),
        ('2022-06-15', 126.0),
        ('2022-12-05', 125.0),
        ('2023-06-15', 124.0),
        ('2023-12-05', 123.0),
        ('2024-06-15', 122.0),
        ('2024-12-05', 121.0),
        ('2025-06-15', 120.0),
        ('2025-09-04', 119.0)
    ]
    
    weight_df = pd.DataFrame(sample_weigh_ins, columns=['date', 'weight_lbs'])
    weight_df['date'] = pd.to_datetime(weight_df['date'])
    weight_df['weight_kg'] = weight_df['weight_lbs'] * 0.453592
    weight_df['weight_kg'] = weight_df['weight_kg'].round(2)
    
    return weight_df

def interpolate_weight_for_workouts(workout_df: pd.DataFrame, weight_df: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolate weight for each workout based on the weight dataset.
    """
    workout_df = workout_df.copy()
    workout_df['date'] = pd.to_datetime(workout_df['date'])
    
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
    print("Interactive Weight Input System")
    print("=" * 50)
    
    # Step 1: Get user input
    weigh_ins = input_weigh_ins()
    
    # Step 2: Create weight dataset
    print("\nCreating weight dataset...")
    weight_df = create_weight_dataset_from_input(weigh_ins)
    
    print(f"\nWeight data points: {len(weight_df)}")
    print(f"Weight range: {weight_df['weight_lbs'].min():.1f} to {weight_df['weight_lbs'].max():.1f} lbs")
    print(f"Weight range: {weight_df['weight_kg'].min():.2f} to {weight_df['weight_kg'].max():.2f} kg")
    
    # Show weight data
    print("\nWeight data:")
    print(weight_df[['date', 'weight_lbs', 'weight_kg']])
    
    # Step 3: Load workout data
    print("\nLoading workout data...")
    workout_df = pd.read_csv('data/workouts.csv')
    
    print(f"Original workouts: {len(workout_df)}")
    print(f"Original weight: {workout_df['weight_kg'].iloc[0]:.2f} kg (constant)")
    
    # Step 4: Apply interpolated weights
    print("\nApplying interpolated weights...")
    workout_df_with_weight = interpolate_weight_for_workouts(workout_df, weight_df)
    
    print(f"New weight range: {workout_df_with_weight['weight_kg'].min():.2f} to {workout_df_with_weight['weight_kg'].max():.2f} kg")
    print(f"New weight unique values: {workout_df_with_weight['weight_kg'].nunique()}")
    
    # Step 5: Save updated data
    output_file = 'data/workouts_with_varied_weight.csv'
    workout_df_with_weight.to_csv(output_file, index=False)
    print(f"\nSaved updated data to '{output_file}'")
    
    # Step 6: Show weight over time
    print("\nWeight over time (sample):")
    weight_over_time = workout_df_with_weight[['date', 'weight_kg']].drop_duplicates().sort_values('date')
    print(weight_over_time.head(10))
    if len(weight_over_time) > 10:
        print("...")
        print(weight_over_time.tail(10))
    
    return workout_df_with_weight

if __name__ == "__main__":
    main()
