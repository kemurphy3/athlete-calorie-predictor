#!/usr/bin/env python3
"""
Weight Interpolation System
Creates realistic weight variation over time based on actual weigh-ins
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Dict

class WeightInterpolator:
    """Interpolates weight over time based on actual weigh-ins."""
    
    def __init__(self, weigh_ins: List[Tuple[str, float]]):
        """
        Initialize with actual weigh-ins.
        
        Args:
            weigh_ins: List of (date_str, weight_kg) tuples
                      e.g., [('2020-01-15', 60.5), ('2020-06-15', 59.2), ...]
        """
        self.weigh_ins = [(pd.to_datetime(date), weight) for date, weight in weigh_ins]
        self.weigh_ins.sort(key=lambda x: x[0])  # Sort by date
        
    def interpolate_weight(self, target_date: pd.Timestamp) -> float:
        """
        Interpolate weight for a given date.
        
        Args:
            target_date: Date to get weight for
            
        Returns:
            Interpolated weight in kg
        """
        if not self.weigh_ins:
            return 58.5  # Default fallback
            
        # If before first weigh-in, use first weight
        if target_date < self.weigh_ins[0][0]:
            return self.weigh_ins[0][1]
            
        # If after last weigh-in, use last weight
        if target_date > self.weigh_ins[-1][0]:
            return self.weigh_ins[-1][1]
            
        # Find surrounding weigh-ins
        for i in range(len(self.weigh_ins) - 1):
            date1, weight1 = self.weigh_ins[i]
            date2, weight2 = self.weigh_ins[i + 1]
            
            if date1 <= target_date <= date2:
                # Linear interpolation between the two weigh-ins
                days_between = (date2 - date1).days
                days_to_target = (target_date - date1).days
                
                if days_between == 0:
                    return weight1
                    
                # Linear interpolation
                weight_diff = weight2 - weight1
                interpolated_weight = weight1 + (weight_diff * days_to_target / days_between)
                return round(interpolated_weight, 2)
                
        return self.weigh_ins[-1][1]  # Fallback to last weight
    
    def add_weight_to_workouts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add interpolated weight to workout dataframe.
        
        Args:
            df: DataFrame with 'date' column
            
        Returns:
            DataFrame with 'weight_kg' column added/updated
        """
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # Interpolate weight for each workout
        df['weight_kg'] = df['date'].apply(self.interpolate_weight)
        
        return df

def create_sample_weight_data() -> List[Tuple[str, float]]:
    """
    Create sample weight data based on your 15 weigh-ins.
    Replace these with your actual Garmin weigh-ins!
    """
    # TODO: Replace with your actual Garmin weigh-ins
    # Format: (date_str, weight_kg)
    sample_weigh_ins = [
        ('2020-01-15', 60.5),
        ('2020-03-20', 59.8),
        ('2020-06-15', 59.2),
        ('2020-09-10', 58.9),
        ('2020-12-05', 58.6),
        ('2021-02-20', 58.3),
        ('2021-05-15', 58.1),
        ('2021-08-10', 57.9),
        ('2021-11-05', 57.7),
        ('2022-01-20', 57.5),
        ('2022-04-15', 57.3),
        ('2022-07-10', 57.1),
        ('2022-10-05', 56.9),
        ('2023-01-20', 56.7),
        ('2023-04-15', 56.5),
        ('2023-07-10', 56.3),
        ('2023-10-05', 56.1),
        ('2024-01-20', 55.9),
        ('2024-04-15', 55.7),
        ('2024-07-10', 55.5),
        ('2024-10-05', 55.3),
        ('2025-01-20', 55.1),
        ('2025-04-15', 54.9),
        ('2025-07-10', 54.7),
        ('2025-09-04', 54.5)
    ]
    
    return sample_weigh_ins

def main():
    """Example usage of weight interpolation."""
    print("Weight Interpolation System")
    print("=" * 50)
    
    # Create sample weight data (replace with your actual weigh-ins)
    weigh_ins = create_sample_weight_data()
    
    # Initialize interpolator
    interpolator = WeightInterpolator(weigh_ins)
    
    # Test interpolation
    test_dates = [
        '2020-01-01',  # Before first weigh-in
        '2020-02-15',  # Between weigh-ins
        '2022-06-15',  # Between weigh-ins
        '2025-09-04',  # Last weigh-in
        '2025-12-01'   # After last weigh-in
    ]
    
    print("Weight interpolation examples:")
    for date_str in test_dates:
        date = pd.to_datetime(date_str)
        weight = interpolator.interpolate_weight(date)
        print(f"  {date_str}: {weight} kg")
    
    # Load workout data and add interpolated weights
    print("\nLoading workout data...")
    df = pd.read_csv('data/workouts.csv')
    
    print(f"Original weight range: {df['weight_kg'].min():.2f} to {df['weight_kg'].max():.2f} kg")
    print(f"Original weight unique values: {df['weight_kg'].nunique()}")
    
    # Add interpolated weights
    df_with_weight = interpolator.add_weight_to_workouts(df)
    
    print(f"New weight range: {df_with_weight['weight_kg'].min():.2f} to {df_with_weight['weight_kg'].max():.2f} kg")
    print(f"New weight unique values: {df_with_weight['weight_kg'].nunique()}")
    
    # Save updated data
    df_with_weight.to_csv('data/workouts_with_varied_weight.csv', index=False)
    print("\nSaved updated data to 'data/workouts_with_varied_weight.csv'")
    
    # Show weight over time
    print("\nWeight over time (sample):")
    sample = df_with_weight[['date', 'weight_kg']].drop_duplicates().sort_values('date')
    print(sample.head(10))
    print("...")
    print(sample.tail(10))

if __name__ == "__main__":
    main()
