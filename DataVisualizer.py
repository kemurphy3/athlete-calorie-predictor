import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DataVisualizer:
    # Handles visualization before and after cleaning
    # Used Cursor to replicate plots after given one model
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.df = data_manager.df 
    
    def visualize_raw_data(self):
        # Plot raw data before cleaning
        print("Creating pre-cleaning visualizations...")
        
        # Update dataframe reference
        #self.df = self.data_manager.df
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Data Distribution Before Cleaning', fontsize=16)
        
        # Calories distribution
        axes[0,0].hist(self.df['CALORIES'].dropna(), bins=50, alpha=0.7, edgecolor='black')
        axes[0,0].set_title('Calories Distribution')
        axes[0,0].set_xlabel('Calories')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].axvline(x=2000, color='red', linestyle='--', label='Upper limit')
        axes[0,0].legend()
        
        # Duration distribution
        axes[0,1].hist(self.df['DURATION_ACTUAL'].dropna(), bins=50, alpha=0.7, edgecolor='black')
        axes[0,1].set_title('Duration Distribution (hours)')
        axes[0,1].set_xlabel('Duration (hours)')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].axvline(x=5, color='red', linestyle='--', label='Upper limit')
        axes[0,1].legend()
        
        # Distance distribution
        axes[0,2].hist(self.df['DISTANCE_ACTUAL'].dropna(), bins=50, alpha=0.7, edgecolor='black')
        axes[0,2].set_title('Distance Distribution (meters)')
        axes[0,2].set_xlabel('Distance (meters)')
        axes[0,2].set_ylabel('Frequency')
        axes[0,2].axvline(x=100000, color='red', linestyle='--', label='Upper limit')
        axes[0,2].legend()
        
        # Weight distribution
        if 'WEIGHT' in self.df.columns:
            axes[1,0].hist(self.df['WEIGHT'].dropna(), bins=30, alpha=0.7, edgecolor='black')
            axes[1,0].set_title('Weight Distribution (kg)')
            axes[1,0].set_xlabel('Weight (kg)')
            axes[1,0].set_ylabel('Frequency')
        
        # Age distribution
        if 'AGE' in self.df.columns:
            axes[1,1].hist(self.df['AGE'].dropna(), bins=30, alpha=0.7, edgecolor='black')
            axes[1,1].set_title('Age Distribution')
            axes[1,1].set_xlabel('Age (years)')
            axes[1,1].set_ylabel('Frequency')
        
        # Heart rate distribution
        if 'HRAVG' in self.df.columns:
            axes[1,2].hist(self.df['HRAVG'].dropna(), bins=30, alpha=0.7, edgecolor='black')
            axes[1,2].set_title('Average Heart Rate Distribution')
            axes[1,2].set_xlabel('Heart Rate (bpm)')
            axes[1,2].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
        
        # Activity ranges analysis
        print("\n++ Activity Metadata Before Cleaning ++")
        print(f"Calories range: {self.df['CALORIES'].min():.0f} - {self.df['CALORIES'].max():.0f}")
        print(f"Duration range: {self.df['DURATION_ACTUAL'].min():.3f} - {self.df['DURATION_ACTUAL'].max():.3f} hours")
        print(f"Distance range: {self.df['DISTANCE_ACTUAL'].min():.0f} - {self.df['DISTANCE_ACTUAL'].max():.0f} meters")
    
    def visualize_cleaned_data(self):
        # Plot cleaned data
        print("\nCreating post-cleaning visualizations...")
        
        # Update dataframe reference
        self.df = self.data_manager.df
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Data Distribution After Cleaning', fontsize=16)
        
        # Calories distribution
        axes[0,0].hist(self.df['CALORIES'], bins=50, alpha=0.7, edgecolor='black', color='green')
        axes[0,0].set_title('Calories Distribution (Cleaned)')
        axes[0,0].set_xlabel('Calories')
        axes[0,0].set_ylabel('Frequency')
        
        # Duration distribution
        axes[0,1].hist(self.df['DURATION_ACTUAL'], bins=50, alpha=0.7, edgecolor='black', color='green')
        axes[0,1].set_title('Duration Distribution (Cleaned)')
        axes[0,1].set_xlabel('Duration (hours)')
        axes[0,1].set_ylabel('Frequency')
        
        # Distance distribution
        axes[0,2].hist(self.df['DISTANCE_ACTUAL'], bins=50, alpha=0.7, edgecolor='black', color='green')
        axes[0,2].set_title('Distance Distribution (Cleaned)')
        axes[0,2].set_xlabel('Distance (meters)')
        axes[0,2].set_ylabel('Frequency')
        
        # Weight distribution
        if 'WEIGHT' in self.df.columns:
            axes[1,0].hist(self.df['WEIGHT'].dropna(), bins=30, alpha=0.7, edgecolor='black', color='green')
            axes[1,0].set_title('Weight Distribution (Cleaned)')
            axes[1,0].set_xlabel('Weight (kg)')
            axes[1,0].set_ylabel('Frequency')
        
        # Age distribution
        if 'AGE' in self.df.columns:
            axes[1,1].hist(self.df['AGE'].dropna(), bins=30, alpha=0.7, edgecolor='black', color='green')
            axes[1,1].set_title('Age Distribution (Cleaned)')
            axes[1,1].set_xlabel('Age (years)')
            axes[1,1].set_ylabel('Frequency')
        
        # Heart rate distribution
        if 'HRAVG' in self.df.columns:
            axes[1,2].hist(self.df['HRAVG'].dropna(), bins=30, alpha=0.7, edgecolor='black', color='green')
            axes[1,2].set_title('Average Heart Rate Distribution (Cleaned)')
            axes[1,2].set_xlabel('Heart Rate (bpm)')
            axes[1,2].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
        
        # Activity ranges after cleaning
        print("\n++ Activity Metadata After Cleaning ++")
        print(f"Calories range: {self.df['CALORIES'].min():.0f} - {self.df['CALORIES'].max():.0f}")
        print(f"Duration range: {self.df['DURATION_ACTUAL'].min():.3f} - {self.df['DURATION_ACTUAL'].max():.3f} hours")
        print(f"Distance range: {self.df['DISTANCE_ACTUAL'].min():.0f} - {self.df['DISTANCE_ACTUAL'].max():.0f} meters")
    
    def update_data(self):
        # Update the dataframe reference after changes
        self.df = self.data_manager.df