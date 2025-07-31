class DataManager:
    
    def __init__(self, filepath, sample_size=None):
        self.filepath = filepath
        self.sample_size = sample_size
        self.df = None
        self.df_cleaned = None
        
    def load_data(self):
        # Load and subset data
        print("Loading data...")
        self.df = pd.read_csv(self.filepath)
        self.df.columns = self.df.columns.str.upper()
        
        if self.sample_size:
            # Shuffle the data first to ensure random sampling
            # Set the random_state for reproducibility
            self.df = self.df.sample(frac=1.0, random_state=42).reset_index(drop=True)
            self.df = self.df.sample(n=min(self.sample_size, len(self.df)), random_state=42).reset_index(drop=True)
            print(f"Randomly sampled {len(self.df):,} records")
        else:
            print(f"Using all {len(self.df):,} records")
            
        return self.df
    
    def analyze_missing_data(self):
        # Identify and return the percentage of missing data for each of the key features in the dataset
        print("\n++ Missing Data Analysis ++")
        
        key_features = ['CALORIES', 'DURATION_ACTUAL', 'DISTANCE_ACTUAL', 'HRMAX', 'HRAVG', 
                       'ELEVATIONAVG', 'ELEVATIONGAIN', 'TRAININGSTRESSSCOREACTUAL', 'AGE', 'WEIGHT', 'SEX']

        # Loop through each of the key features and store their missing count and percentage in a dictionary for easy reference
        missing_stats = {}
        for col in key_features:
            if col in self.df.columns:
                missing_count = self.df[col].isnull().sum() # count the number of nulls in each column
                missing_pct = (missing_count / len(self.df)) * 100 # divide missing count by total entries to find percent
                missing_stats[col] = {'count': missing_count, 'percentage': missing_pct}
                print(f"{col:<25}: {missing_count:>6,} missing ({missing_pct:>5.1f}%)")
        
        return missing_stats
    
    def clean_data(self):
        # Clean data by removing invalid and outlier data
        print("\nCleaning data...")
        # Count the number of rows in the initial dataset, if subset, should be that value
        initial_rows = len(self.df)
        
        # Identify extreme values or typos before removal
        extreme_calories = (self.df['CALORIES'] < 5) | (self.df['CALORIES'] > 2000) | (self.df['CALORIES'].isnull())
        extreme_duration = (self.df['DURATION_ACTUAL'] < 0.0167) | (self.df['DURATION_ACTUAL'] > 5)
        extreme_distance = (self.df['DISTANCE_ACTUAL'] < 100) | (self.df['DISTANCE_ACTUAL'] > 100000)
        extreme_hravg = (self.df['HRAVG'] < 30) | (self.df['HRAVG'] > 250)
        extreme_hrmax = (self.df['HRMAX'] < 30) | (self.df['HRMAX'] > 250)
        extreme_age = (self.df['AGE'] < 18) | (self.df['AGE'] > 100)
        extreme_weight = (self.df['WEIGHT'] < 20) | (self.df['WEIGHT'] > 200)

        # Count the number of extreme values so a total can be presented to the user
        calories_removed = extreme_calories.sum()
        duration_removed = extreme_duration.sum()
        distance_removed = extreme_distance.sum()
        hravg_removed = extreme_hravg.sum()
        hrmax_removed = extreme_hrmax.sum()
        age_removed = extreme_age.sum()
        weight_removed = extreme_weight.sum()
        
        # Remove invalid records by preserving all the data that doesn't violate an acceptable limit rather than removing all that don't fall in the acceptable range
        # This nuance allows for null values to remain which helps with overall data retention
        self.df = self.df[~(extreme_calories | extreme_duration | extreme_distance | extreme_hravg | extreme_hrmax | extreme_age | extreme_weight)]
        
        final_rows = len(self.df)
        total_removed = initial_rows - final_rows

        # Deliver a summarized output of invalid data and how many row remain to quantify data retention
        print(f"\n++ Summary of Cleaned Dataset ++")
        print(f"Records flagged for invalid calories (< 5 or > 2000): {calories_removed:,}")
        print(f"Records flagged for invalid duration (< 1 minute or > 5 hours): {duration_removed:,}")
        print(f"Records flagged for invalid distance (< 100 m or > 100 km): {distance_removed:,}")
        print(f"Records flagged for invalid average heart rate (< 30 or > 250): {hravg_removed:,}")
        print(f"Records flagged for invalid heart rate max (< 30 or > 250): {hrmax_removed:,}")
        print(f"Records flagged for invalid age (< 18 or > 100): {age_removed:,}")
        print(f"Records flagged for invalid weight (< 20 kgs or > 200 kgs): {weight_removed:,}")
        print(f"Total rows removed: {total_removed:,}")
        print(f"Final rows after cleaning: {final_rows:,}")
        print(f"Data retention: {((final_rows/initial_rows)*100):.1f}%")
        
        self.df_cleaned = self.df.copy()
        return self.df
    
    def analyze_athlete_distribution(self):
        # Determine the distribution of activities per athlete
        # Prevents the script from throwing an error by trying to find a column that doesn't exist
        if 'ATHLETE_ID' not in self.df.columns:
            print("No ATHLETE_ID column found")
            return
        
        print("\n++ Activities by Athlete ++")
        
        athlete_counts = self.df['ATHLETE_ID'].value_counts() # calculates how many unique athlete ID numbers there are
        total_athletes = len(athlete_counts)
        total_activities = len(self.df) # each row is an activity

        # Prints the counts of activities and athletes and their relationship 
        print(f"Total athletes: {total_athletes:,}")
        print(f"Total activities: {total_activities:,}")
        print(f"Average activities per athlete: {total_activities/total_athletes:.1f}")
        
        # Activity count distribution, which will inform the type of analysis done
        activity_ranges = [
            (1, 5, "1-5 activities"),
            (6, 15, "6-15 activities"),
            (16, 30, "16-30 activities"),
            (31, 50, "31-50 activities"),
            (51, float('inf'), "50+ activities")
        ]

        # Had Cursor determine how to assign and display activity counts to each range
        print("\n++ Athletes by Activity Count ++")
        for min_act, max_act, label in activity_ranges:
            if max_act == float('inf'):
                count = (athlete_counts >= min_act).sum()
            else:
                count = ((athlete_counts >= min_act) & (athlete_counts <= max_act)).sum()
            percentage = (count / total_athletes) * 100
            print(f"  {label:<15}: {count:>6,} athletes ({percentage:>5.1f}%)")
    
    def get_data(self):
        # Return the current dataset
        return self.df