import os
from pathlib import Path

class APIConfig:
    BASE_DIR = Path("data/raw/race_data")
    CACHE_DIR = BASE_DIR / "cache"
    
    @classmethod
    def ensure_directories_exist(cls):
        """Ensure all required directories exist"""
        cls.BASE_DIR.mkdir(parents=True, exist_ok=True)
        cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        # For backward compatibility
        cls.FASTF1_CACHE_DIR = str(cls.CACHE_DIR)

class DataConfig:
    SESSIONS = ['FP1', 'FP2', 'FP3', 'Q', 'R']  # Free Practice, Qualifying, Race