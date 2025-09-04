#!/usr/bin/env python3
"""
Athlete Calorie Predictor - Unified Dataset Pipeline

Builds a unified workout dataset from multiple sources:
- Strava API (personal workout data)
- UCI PAMAP2 (aggregated to workout-level)
- Kaggle Fitbit-style dataset (aggregated to workout-level)

Features:
- Automatic archival and backups
- Schema validation and standardization
- Optional synthetic data expansion
- Idempotent behavior
- Comprehensive logging
- CSV-only output format (optimized for small datasets)
"""

import os
import sys
import json
import logging
import shutil
import time
import hashlib
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
import warnings

import pandas as pd
import numpy as np
import requests

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger = logging.getLogger(__name__)
    logger.info("Loaded environment variables from .env file")
except ImportError:
    # python-dotenv not installed, continue without it
    pass

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuration class with environment variable defaults."""
    
    # Paths
    DATA_DIR: str = "data"
    SCRIPTS_DIR: str = "scripts"
    ARCHIVE_DIR: str = "archived_scripts"
    PAMAP2_PATH: str = "external/uci_pamap2"
    KAGGLE_FITBIT_PATH: str = "external/kaggle_fitbit"
    
    # Strava API
    STRAVA_ACCESS_TOKEN: Optional[str] = None
    STRAVA_CLIENT_ID: Optional[str] = None
    STRAVA_CLIENT_SECRET: Optional[str] = None
    STRAVA_REFRESH_TOKEN: Optional[str] = None
    
    # Identity defaults
    DEFAULT_SEX: str = "f"
    DEFAULT_AGE: float = 33.0
    DEFAULT_WEIGHT_KG: float = 57.6
    
    # Output behavior
    SYNTHETIC_TARGET_ROWS: int = 0
    
    def __post_init__(self):
        """Load configuration from environment variables."""
        # Paths
        self.DATA_DIR = os.getenv('DATA_DIR', self.DATA_DIR)
        self.SCRIPTS_DIR = os.getenv('SCRIPTS_DIR', self.SCRIPTS_DIR)
        self.ARCHIVE_DIR = os.getenv('ARCHIVE_DIR', self.ARCHIVE_DIR)
        self.PAMAP2_PATH = os.getenv('PAMAP2_PATH', self.PAMAP2_PATH)
        self.KAGGLE_FITBIT_PATH = os.getenv('KAGGLE_FITBIT_PATH', self.KAGGLE_FITBIT_PATH)
        
        # Strava
        self.STRAVA_ACCESS_TOKEN = os.getenv('STRAVA_ACCESS_TOKEN')
        self.STRAVA_CLIENT_ID = os.getenv('STRAVA_CLIENT_ID')
        self.STRAVA_CLIENT_SECRET = os.getenv('STRAVA_CLIENT_SECRET')
        self.STRAVA_REFRESH_TOKEN = os.getenv('STRAVA_REFRESH_TOKEN')
        
        # Identity defaults
        self.DEFAULT_SEX = os.getenv('DEFAULT_SEX', self.DEFAULT_SEX)
        self.DEFAULT_AGE = float(os.getenv('DEFAULT_AGE', self.DEFAULT_AGE))
        self.DEFAULT_WEIGHT_KG = float(os.getenv('DEFAULT_WEIGHT_KG', self.DEFAULT_WEIGHT_KG))
        
        # Output behavior
        self.SYNTHETIC_TARGET_ROWS = int(os.getenv('SYNTHETIC_TARGET_ROWS', self.SYNTHETIC_TARGET_ROWS))


class DataArchiver:
    """Handles archival and backup operations."""
    
    def __init__(self, config: Config):
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def archive_scripts(self) -> None:
        """Archive the scripts directory if it exists."""
        scripts_path = Path(self.config.SCRIPTS_DIR)
        if scripts_path.exists():
            archive_path = Path(self.config.ARCHIVE_DIR) / f"scripts_{self.timestamp}"
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Archiving {scripts_path} to {archive_path}")
            shutil.copytree(scripts_path, archive_path)
            logger.info(f"Scripts archived successfully")
        else:
            logger.info("No scripts directory found to archive")
    
    def backup_existing_data(self) -> None:
        """Backup existing workout data files."""
        data_path = Path(self.config.DATA_DIR)
        data_path.mkdir(parents=True, exist_ok=True)
        
        # Backup workout_data files
        for pattern in ["workout_data.*", "workouts.*"]:
            for file_path in data_path.glob(pattern):
                if file_path.exists():
                    backup_file = data_path / f"{file_path.stem}.bkp_{self.timestamp}{file_path.suffix}"
                    shutil.copy2(file_path, backup_file)
                    logger.info(f"Backed up {file_path} to {backup_file}")


class StravaLoader:
    """Loads workout data from Strava API."""
    
    def __init__(self, config: Config):
        self.config = config
        self.base_url = "https://www.strava.com/api/v3"
        self.session = requests.Session()
    
    def refresh_token(self) -> Optional[str]:
        """Refresh Strava access token if refresh credentials are available."""
        if not all([self.config.STRAVA_CLIENT_ID, self.config.STRAVA_CLIENT_SECRET, self.config.STRAVA_REFRESH_TOKEN]):
            logger.warning("Missing refresh credentials (client_id, client_secret, or refresh_token)")
            return None
        
        try:
            logger.info("Attempting to refresh Strava access token...")
            response = self.session.post(
                "https://www.strava.com/oauth/token",
                data={
                    "client_id": self.config.STRAVA_CLIENT_ID,
                    "client_secret": self.config.STRAVA_CLIENT_SECRET,
                    "grant_type": "refresh_token",
                    "refresh_token": self.config.STRAVA_REFRESH_TOKEN
                }
            )
            
            if response.status_code == 200:
                token_data = response.json()
                new_access_token = token_data.get("access_token")
                if new_access_token:
                    logger.info("Successfully obtained new access token")
                    return new_access_token
                else:
                    logger.warning("Refresh response missing access_token")
                    return None
            else:
                logger.warning(f"Token refresh failed with status {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            logger.warning(f"Failed to refresh Strava token: {e}")
            return None
    
    def get_athlete_profile(self) -> Dict:
        """Fetch athlete profile including demographics."""
        try:
            response = self.session.get(f"{self.base_url}/athlete")
            response.raise_for_status()
            athlete = response.json()
            
            logger.info(f"Fetched athlete profile: {athlete.get('firstname', '')} {athlete.get('lastname', '')}")
            logger.info(f"Demographics: Sex={athlete.get('sex', 'Unknown')}, Weight={athlete.get('weight', 'Unknown')}kg")
            
            return athlete
        except Exception as e:
            logger.warning(f"Failed to fetch athlete profile: {e}")
            return {}
    
    def get_activities(self) -> pd.DataFrame:
        """Fetch activities from Strava API with pagination and rate limiting."""
        access_token = None
        
        # First try to use existing access token if available
        if self.config.STRAVA_ACCESS_TOKEN:
            access_token = self.config.STRAVA_ACCESS_TOKEN
            logger.info("Using existing Strava access token")
        
        # If no access token, try to refresh using refresh credentials
        if not access_token and all([
            self.config.STRAVA_CLIENT_ID,
            self.config.STRAVA_CLIENT_SECRET,
            self.config.STRAVA_REFRESH_TOKEN
        ]):
            logger.info("No access token found, attempting to refresh...")
            access_token = self.refresh_token()
            if access_token:
                logger.info("Successfully refreshed Strava access token")
            else:
                logger.warning("Failed to refresh Strava access token")
        
        # If still no access token, we can't proceed
        if not access_token:
            logger.warning("No valid Strava access token available, skipping Strava data")
            return pd.DataFrame()
        
        self.session.headers.update({"Authorization": f"Bearer {access_token}"})
        
        # Fetch athlete profile first
        athlete_profile = self.get_athlete_profile()
        
        all_activities = []
        page = 1
        per_page = 200
        
        while True:
            try:
                # Fetch activities with explicit parameters for all-time data
                response = self.session.get(
                    f"{self.base_url}/athlete/activities",
                    params={
                        "page": page, 
                        "per_page": per_page,
                        "after": 0,  # Start from beginning of time
                        "before": int(time.time())  # Up to now
                    }
                )
                
                if response.status_code == 429:  # Rate limited
                    retry_after = int(response.headers.get('Retry-After', 60))
                    logger.info(f"Rate limited, waiting {retry_after} seconds")
                    time.sleep(retry_after)
                    continue
                
                response.raise_for_status()
                activities = response.json()
                
                if not activities:
                    logger.info(f"No more activities found on page {page}")
                    break
                
                all_activities.extend(activities)
                logger.info(f"Fetched page {page} with {len(activities)} activities")
                page += 1
                
                # Small delay to be respectful
                time.sleep(0.1)
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching Strava activities: {e}")
                break
        
        if not all_activities:
            logger.warning("No activities fetched from Strava")
            return pd.DataFrame()
        
        logger.info(f"Total Strava activities fetched: {len(all_activities)}")
        return self._normalize_activities(all_activities, athlete_profile)
    
    def _normalize_activities(self, activities: List[Dict], athlete_profile: Dict = None) -> pd.DataFrame:
        """Normalize Strava activities to unified schema."""
        normalized_data = []
        
        for activity in activities:
            # Skip activities with invalid duration
            if not activity.get('moving_time') or activity['moving_time'] <= 0:
                continue
            
            # Calculate calories (use provided or estimate)
            calories = activity.get('calories')
            if not calories or calories <= 0:
                # Simple calorie estimation based on HR and duration
                # Note: Strava may not return calories for all API scopes, so we use a fallback
                avg_hr = activity.get('average_heartrate', 0)
                duration_hours = activity['moving_time'] / 3600.0
                if avg_hr > 0:
                    # Rough estimate: higher HR = higher calorie burn
                    calories = int(avg_hr * duration_hours * 0.8)  # Simplified formula
                else:
                    calories = int(duration_hours * 400)  # Fallback: 400 cal/hour
            
            # Skip negative calories
            if calories <= 0:
                continue
            
            # Generate stable workout_id
            workout_id = self._generate_workout_id(activity)
            
            # Parse date
            date_str = activity.get('start_date_local', '')
            try:
                date = pd.to_datetime(date_str).date() if date_str else None
            except:
                date = None
            
            # Use real Strava demographics if available, fallback to config defaults
            sex = athlete_profile.get('sex', self.config.DEFAULT_SEX) if athlete_profile else self.config.DEFAULT_SEX
            weight_kg = athlete_profile.get('weight', self.config.DEFAULT_WEIGHT_KG) if athlete_profile else self.config.DEFAULT_WEIGHT_KG
            age = athlete_profile.get('age', self.config.DEFAULT_AGE) if athlete_profile else self.config.DEFAULT_AGE
            
            normalized_data.append({
                'workout_id': workout_id,
                'source': 'STRAVA',
                'athlete_id': str(activity.get('athlete', {}).get('id', '')),
                'activity_type': activity.get('type', 'Other'),
                'date': date,
                'duration_hours': activity['moving_time'] / 3600.0,
                'distance_m': activity.get('distance', np.nan),
                'calories': calories,
                'hr_avg': activity.get('average_heartrate', np.nan),
                'hr_max': activity.get('max_heartrate', np.nan),
                'elev_gain_m': activity.get('total_elevation_gain', np.nan),
                'elev_avg_m': np.nan,  # Strava doesn't provide average elevation
                'training_stress': np.nan,  # Strava doesn't provide TSS
                'sex': sex,
                'age': age,
                'weight_kg': weight_kg
            })
        
        df = pd.DataFrame(normalized_data)
        
        # Filter out invalid rows
        initial_count = len(df)
        df = df[
            (df['duration_hours'] > 0) & 
            (df['calories'] > 0)
        ]
        filtered_count = initial_count - len(df)
        
        if filtered_count > 0:
            logger.info(f"Filtered out {filtered_count} invalid Strava activities")
        
        return df
    
    def _generate_workout_id(self, activity: Dict) -> str:
        """Generate a stable workout ID for Strava activities."""
        activity_id = activity.get('id', '')
        athlete_id = activity.get('athlete', {}).get('id', '')
        date_str = activity.get('start_date_local', '')
        
        # Create a hash for stability
        id_string = f"strava_{athlete_id}_{activity_id}_{date_str}"
        return hashlib.md5(id_string.encode()).hexdigest()[:16]


class UCI_PAMAP2Loader:
    """Loads and aggregates UCI PAMAP2 dataset to workout-level."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def load_data(self) -> pd.DataFrame:
        """Load and aggregate UCI PAMAP2 data to workout-level."""
        pamap2_path = Path(self.config.PAMAP2_PATH)
        
        if not pamap2_path.exists():
            logger.warning(f"UCI PAMAP2 path not found: {pamap2_path}")
            return pd.DataFrame()
        
        # Find all CSV files recursively
        csv_files = list(pamap2_path.rglob("*.csv"))
        if not csv_files:
            logger.warning(f"No CSV files found in {pamap2_path}")
            return pd.DataFrame()
        
        logger.info(f"Found {len(csv_files)} CSV files in UCI PAMAP2 directory")
        
        all_workouts = []
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                # Check if required columns exist
                required_cols = ['heart_rate', 'EE', 'activity_id', 'subject_id']
                if not all(col in df.columns for col in required_cols):
                    logger.warning(f"Missing required columns in {csv_file.name}")
                    continue
                
                # Aggregate by activity_id and subject_id
                workouts = self._aggregate_workouts(df)
                all_workouts.extend(workouts)
                
            except Exception as e:
                logger.warning(f"Error processing {csv_file.name}: {e}")
                continue
        
        if not all_workouts:
            logger.warning("No valid workout data found in UCI PAMAP2 files")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_workouts)
        
        # Filter out tiny segments
        initial_count = len(df)
        df = df[df['duration_hours'] > 0.02]  # Keep only > 1.2 minutes
        filtered_count = initial_count - len(df)
        
        if filtered_count > 0:
            logger.info(f"Filtered out {filtered_count} tiny UCI PAMAP2 segments")
        
        logger.info(f"UCI PAMAP2: {len(df)} workouts loaded")
        return df
    
    def _aggregate_workouts(self, df: pd.DataFrame) -> List[Dict]:
        """Aggregate second-by-second data into workout-level records."""
        workouts = []
        
        # Group by activity_id and subject_id
        for (activity_id, subject_id), group in df.groupby(['activity_id', 'subject_id']):
            if len(group) == 0:
                continue
            
            # Calculate workout metrics
            heart_rates = group['heart_rate'].dropna()
            if len(heart_rates) == 0:
                continue
            
            ee_values = group['EE'].dropna()
            if len(ee_values) == 0:
                continue
            
            # Generate stable workout_id
            workout_id = f"pamap2_{subject_id}_{activity_id}_{len(workouts)}"
            
            # Map activity type if possible
            activity_type = self._map_activity_type(activity_id)
            
            workout = {
                'workout_id': workout_id,
                'source': 'UCI_PAMAP2',
                'athlete_id': str(subject_id),
                'activity_type': activity_type,
                'date': None,  # UCI PAMAP2 doesn't provide dates
                'duration_hours': len(group) / 3600.0,  # seconds to hours
                'distance_m': np.nan,  # UCI PAMAP2 doesn't provide distance
                'calories': ee_values.sum(),
                'hr_avg': heart_rates.mean(),
                'hr_max': heart_rates.max(),
                'elev_gain_m': np.nan,
                'elev_avg_m': np.nan,
                'training_stress': np.nan,
                'sex': self.config.DEFAULT_SEX,
                'age': self.config.DEFAULT_AGE,
                'weight_kg': self.config.DEFAULT_WEIGHT_KG
            }
            
            workouts.append(workout)
        
        return workouts
    
    def _map_activity_type(self, activity_id) -> str:
        """Map UCI PAMAP2 activity IDs to human-readable types."""
        # Basic mapping - can be expanded based on actual dataset
        activity_map = {
            1: 'Other',
            2: 'Other',
            3: 'Other',
            4: 'Other',
            5: 'Other',
            6: 'Other',
            7: 'Other',
            8: 'Other',
            9: 'Other',
            10: 'Other',
            11: 'Other',
            12: 'Other',
            13: 'Other',
            14: 'Other',
            15: 'Other',
            16: 'Other',
            17: 'Other',
            18: 'Other',
            19: 'Other',
            20: 'Other',
            21: 'Other',
            22: 'Other',
            23: 'Other',
            24: 'Other'
        }
        return activity_map.get(activity_id, 'Other')


class KaggleFitbitLoader:
    """Loads and aggregates Kaggle Fitbit-style dataset to workout-level."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def load_data(self) -> pd.DataFrame:
        """Load and aggregate Kaggle Fitbit data to workout-level."""
        fitbit_path = Path(self.config.KAGGLE_FITBIT_PATH)
        
        if not fitbit_path.exists():
            logger.warning(f"Kaggle Fitbit path not found: {fitbit_path}")
            return pd.DataFrame()
        
        # Find all CSV files recursively
        csv_files = list(fitbit_path.rglob("*.csv"))
        if not csv_files:
            logger.warning(f"No CSV files found in {fitbit_path}")
            return pd.DataFrame()
        
        logger.info(f"Found {len(csv_files)} CSV files in Kaggle Fitbit directory")
        
        all_workouts = []
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                # Map columns using heuristics
                column_mapping = self._map_columns(df.columns)
                if not column_mapping:
                    logger.warning(f"Could not map columns in {csv_file.name}")
                    continue
                
                # Rename columns for consistency
                df = df.rename(columns=column_mapping)
                
                # Aggregate workouts
                workouts = self._aggregate_workouts(df)
                all_workouts.extend(workouts)
                
            except Exception as e:
                logger.warning(f"Error processing {csv_file.name}: {e}")
                continue
        
        if not all_workouts:
            logger.warning("No valid workout data found in Kaggle Fitbit files")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_workouts)
        
        # Filter out invalid durations
        initial_count = len(df)
        df = df[df['duration_hours'] > 0]
        filtered_count = initial_count - len(df)
        
        if filtered_count > 0:
            logger.info(f"Filtered out {filtered_count} invalid Kaggle Fitbit workouts")
        
        logger.info(f"Kaggle Fitbit: {len(df)} workouts loaded")
        return df
    
    def _map_columns(self, columns: pd.Index) -> Dict[str, str]:
        """Map column names using heuristics for different schemas."""
        column_mapping = {}
        
        # Calories columns
        for col in columns:
            if 'calorie' in col.lower():
                column_mapping[col] = 'calories'
                break
        
        # Distance columns
        for col in columns:
            if 'distance' in col.lower():
                column_mapping[col] = 'distance'
                break
        
        # Duration columns (convert to minutes)
        for col in columns:
            if any(term in col.lower() for term in ['minute', 'duration', 'time']):
                column_mapping[col] = 'duration_minutes'
                break
        
        # Heart rate columns
        for col in columns:
            if any(term in col.lower() for term in ['avg_hr', 'heartrateavg', 'average_heartrate']):
                column_mapping[col] = 'avg_hr'
                break
        
        for col in columns:
            if any(term in col.lower() for term in ['max_hr', 'heartratemax', 'maxheartrate']):
                column_mapping[col] = 'max_hr'
                break
        
        # Grouping key columns
        for col in columns:
            if any(term in col.lower() for term in ['activityid', 'sessionid', 'workoutid']):
                column_mapping[col] = 'group_key'
                break
        
        # Fallback to date-based grouping
        if 'group_key' not in column_mapping.values():
            for col in columns:
                if any(term in col.lower() for term in ['date', 'activitydate']):
                    column_mapping[col] = 'group_key'
                    break
        
        return column_mapping
    
    def _aggregate_workouts(self, df: pd.DataFrame) -> List[Dict]:
        """Aggregate data into workout-level records."""
        workouts = []
        
        # Group by the identified key
        if 'group_key' not in df.columns:
            logger.warning("No grouping key found for workout aggregation")
            return []
        
        for group_key, group in df.groupby('group_key'):
            if len(group) == 0:
                continue
            
            # Calculate workout metrics
            calories = group.get('calories', pd.Series([0])).sum()
            distance = group.get('distance', pd.Series([np.nan])).sum()
            duration_minutes = group.get('duration_minutes', pd.Series([0])).sum()
            
            # Skip if no duration
            if duration_minutes <= 0:
                continue
            
            # Heart rate metrics
            avg_hr = group.get('avg_hr', pd.Series([np.nan])).mean()
            max_hr = group.get('max_hr', pd.Series([np.nan])).max()
            
            # Generate stable workout_id
            workout_id = f"kaggle_{hash(str(group_key)) % 1000000}"
            
            # Try to get date if available
            date = None
            if 'date' in group.columns:
                try:
                    date = pd.to_datetime(group['date'].iloc[0]).date()
                except:
                    pass
            
            workout = {
                'workout_id': workout_id,
                'source': 'KAGGLE_FITBIT',
                'athlete_id': 'kaggle_generic',
                'activity_type': 'Other',  # Default for Kaggle data
                'date': date,
                'duration_hours': duration_minutes / 60.0,  # minutes to hours
                'distance_m': distance if not pd.isna(distance) else np.nan,
                'calories': calories,
                'hr_avg': avg_hr if not pd.isna(avg_hr) else np.nan,
                'hr_max': max_hr if not pd.isna(max_hr) else np.nan,
                'elev_gain_m': np.nan,
                'elev_avg_m': np.nan,
                'training_stress': np.nan,
                'sex': self.config.DEFAULT_SEX,
                'age': self.config.DEFAULT_AGE,
                'weight_kg': self.config.DEFAULT_WEIGHT_KG
            }
            
            workouts.append(workout)
        
        return workouts


class DatasetValidator:
    """Validates and standardizes the unified dataset."""
    
    REQUIRED_COLUMNS = [
        'workout_id', 'source', 'athlete_id', 'activity_type', 'date',
        'duration_hours', 'distance_m', 'calories', 'hr_avg', 'hr_max',
        'elev_gain_m', 'elev_avg_m', 'training_stress', 'sex', 'age', 'weight_kg'
    ]
    
    @staticmethod
    def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
        """Validate schema and clean the dataset."""
        logger.info("Validating and cleaning dataset...")
        
        # Ensure all required columns exist
        missing_cols = set(DatasetValidator.REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            logger.warning(f"Adding missing columns: {missing_cols}")
            for col in missing_cols:
                df[col] = np.nan
        
        # Reorder columns to match required schema
        df = df[DatasetValidator.REQUIRED_COLUMNS]
        
        # Coerce numeric types
        numeric_cols = ['duration_hours', 'distance_m', 'calories', 'hr_avg', 'hr_max',
                       'elev_gain_m', 'elev_avg_m', 'training_stress', 'age', 'weight_kg']
        
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Normalize SEX column
        df['sex'] = df['sex'].str.lower().map({'m': 'm', 'f': 'f'}).fillna('f')
        
        # Validate and filter rows
        initial_count = len(df)
        
        # Drop invalid rows
        df = df[
            (df['duration_hours'] > 0) & 
            (df['calories'] > 0)
        ]
        
        filtered_count = initial_count - len(df)
        if filtered_count > 0:
            logger.info(f"Filtered out {filtered_count} invalid rows")
        
        # Log summary statistics
        logger.info(f"Final dataset: {len(df)} rows")
        logger.info(f"Source breakdown: {df['source'].value_counts().to_dict()}")
        
        return df


class SyntheticExpander:
    """Expands dataset with synthetic data if needed."""
    
    def __init__(self, config: Config):
        self.config = config
        # Set fixed seed for reproducibility
        np.random.seed(42)
    
    def expand_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Expand dataset to target size with synthetic data."""
        if self.config.SYNTHETIC_TARGET_ROWS <= 0:
            return df
        
        current_rows = len(df)
        if current_rows >= self.config.SYNTHETIC_TARGET_ROWS:
            logger.info(f"Dataset already has {current_rows} rows, no expansion needed")
            return df
        
        target_rows = self.config.SYNTHETIC_TARGET_ROWS
        additional_rows = target_rows - current_rows
        
        logger.info(f"Expanding dataset from {current_rows} to {target_rows} rows with synthetic data")
        
        # Bootstrap sample with replacement
        synthetic_indices = np.random.choice(current_rows, size=additional_rows, replace=True)
        synthetic_df = df.iloc[synthetic_indices].copy()
        
        # Apply small jitter to numeric columns
        jitter_cols = ['calories', 'distance_m', 'duration_hours', 'hr_avg', 'hr_max']
        
        for col in jitter_cols:
            if col in synthetic_df.columns:
                # Add ±5% jitter
                jitter_factor = np.random.uniform(0.95, 1.05, size=len(synthetic_df))
                synthetic_df[col] = synthetic_df[col] * jitter_factor
        
        # Mark as synthetic and ensure unique workout_ids
        synthetic_df['source'] = 'SYNTHETIC'
        for i, row in synthetic_df.iterrows():
            synthetic_df.at[i, 'workout_id'] = f"synthetic_{i}_{hash(str(row['workout_id'])) % 1000000}"
        
        # Combine original and synthetic data
        expanded_df = pd.concat([df, synthetic_df], ignore_index=True)
        
        logger.info(f"Dataset expanded to {len(expanded_df)} rows")
        return expanded_df


class DatasetBuilder:
    """Main orchestrator for building the unified dataset."""
    
    def __init__(self, config: Config):
        self.config = config
        self.archiver = DataArchiver(config)
        self.strava_loader = StravaLoader(config)
        self.validator = DatasetValidator()
        self.expander = SyntheticExpander(config)
    
    def build(self) -> pd.DataFrame:
        """Build the unified dataset from all sources."""
        logger.info("Starting dataset build process...")
        
        # Step 1: Archive and backup
        self.archiver.archive_scripts()
        self.archiver.backup_existing_data()
        
        # Step 2: Load data from all sources
        datasets = []
        
        # Load Strava data
        strava_df = self.strava_loader.get_activities()
        if not strava_df.empty:
            datasets.append(strava_df)
            logger.info(f"Strava: {len(strava_df)} workouts loaded")
        
        # Note: UCI PAMAP2 and Kaggle Fitbit loaders removed - focusing on Strava data only
        
        # Step 3: Check if any data was loaded
        if not datasets:
            logger.error("No data sources were successfully loaded!")
            logger.error("Please check your configuration and ensure at least one data source is available.")
            sys.exit(1)
        
        # Step 4: Combine datasets
        logger.info("Combining datasets...")
        combined_df = pd.concat(datasets, ignore_index=True)
        logger.info(f"Combined dataset: {len(combined_df)} rows")
        
        # Step 5: Validate and clean
        validated_df = self.validator.validate_and_clean(combined_df)
        
        # Step 6: Optional synthetic expansion
        if self.config.SYNTHETIC_TARGET_ROWS > 0:
            validated_df = self.expander.expand_dataset(validated_df)
        
        # Step 7: Save outputs
        self._save_outputs(validated_df)
        
        logger.info("Dataset build completed successfully!")
        return validated_df
    
    def _save_outputs(self, df: pd.DataFrame) -> None:
        """Save the dataset to CSV format."""
        data_path = Path(self.config.DATA_DIR)
        data_path.mkdir(parents=True, exist_ok=True)
        
        # Save CSV (primary format)
        csv_file = data_path / "workouts.csv"
        df.to_csv(csv_file, index=False)
        logger.info(f"Dataset saved to {csv_file}")
        
        # Log final summary
        logger.info("=" * 50)
        logger.info("DATASET BUILD SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total rows: {len(df):,}")
        logger.info(f"Source breakdown:")
        for source, count in df['source'].value_counts().items():
            logger.info(f"  {source}: {count:,}")
        logger.info("=" * 50)


def main():
    """Main entry point."""
    try:
        # Load configuration
        config = Config()
        
        # Create and run dataset builder
        builder = DatasetBuilder(config)
        final_df = builder.build()
        
        logger.info("Dataset build completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Dataset build interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Dataset build failed: {e}")
        logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
