#!/usr/bin/env python3
"""
Environment Setup Helper for Athlete Calorie Predictor

This script helps you set up your .env file with the necessary configuration.
It will create a .env file based on config.env.example and prompt you for values.
"""

import os
import shutil
from pathlib import Path

def main():
    print("🏃‍♂️ Athlete Calorie Predictor - Environment Setup")
    print("=" * 50)
    
    # Check if .env already exists
    if Path('.env').exists():
        print("⚠️  .env file already exists!")
        response = input("Do you want to overwrite it? (y/N): ").lower()
        if response != 'y':
            print("Setup cancelled. Your existing .env file is preserved.")
            return
    
    # Check if config.env.example exists
    if not Path('config.env.example').exists():
        print("❌ config.env.example not found!")
        print("Please ensure you're running this from the project root directory.")
        return
    
    print("📝 Setting up your .env file...")
    
    # Copy the template
    try:
        shutil.copy('config.env.example', '.env')
        print("✅ Created .env from template")
    except Exception as e:
        print(f"❌ Failed to create .env: {e}")
        return
    
    print("\n🔐 Now you need to edit the .env file with your actual values:")
    print("1. Open .env in your text editor")
    print("2. Replace placeholder values with your real credentials")
    print("3. Save the file")
    
    print("\n📋 Key things to configure:")
    print("- STRAVA_ACCESS_TOKEN: Your Strava API access token")
    print("- STRAVA_CLIENT_ID: Your Strava app client ID")
    print("- STRAVA_CLIENT_SECRET: Your Strava app client secret")
    print("- STRAVA_REFRESH_TOKEN: Your Strava refresh token")
    
    print("\n🔗 Get Strava credentials from: https://www.strava.com/settings/api")
    
    print("\n✅ Setup complete! Your .env file is ready to use.")
    print("💡 Remember: .env is in .gitignore and will never be committed to git.")

if __name__ == "__main__":
    main()
