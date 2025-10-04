"""
FastF1 Data Client
Handles all interactions with the FastF1 API for race data collection
"""

import fastf1
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from loguru import logger
import warnings
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.config.config import APIConfig, DataConfig

# Suppress FastF1 warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class FastF1Client:
    def __init__(self):
        """Initialize the FastF1 client with cache configuration"""
        # Set up FastF1 cache
        fastf1.Cache.enable_cache(APIConfig.FASTF1_CACHE_DIR)
        logger.info(f"FastF1 cache enabled at: {APIConfig.FASTF1_CACHE_DIR}")
        
        # Ensure required directories exist
        APIConfig.ensure_directories_exist()
    
    def get_race_weekend(self, year: int, race_name: str) -> Dict[str, fastf1.core.Session]:
        """
        Get all sessions for a race weekend
        
        Args:
        Args:
            year: Season year
            
        Returns:
            List of race names
        """
        try:
            schedule = fastf1.get_event_schedule(year)
            return schedule['EventName'].tolist()
        except Exception as e:
            logger.error(f"Error getting race schedule for {year}: {e}")
            return []
    
    def validate_driver_abbreviation(self, session: fastf1.core.Session, 
                                   driver: str) -> bool:
        """
        Check if a driver abbreviation is valid for the session
        
        Args:
            session: FastF1 session object
            driver: Driver abbreviation
            
        Returns:
            True if valid, False otherwise
        """
        try:
            drivers = session.drivers
            return driver in drivers
        except:
            return False