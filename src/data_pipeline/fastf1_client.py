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

from src.config.config import APIConfig, DataConfig

# Suppress FastF1 warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class FastF1Client:
    """Client for collecting F1 data using FastF1 API"""
    
    def __init__(self):
        """Initialize the FastF1 client with cache configuration"""
        # Set up FastF1 cache
        fastf1.Cache.enable_cache(APIConfig.FASTF1_CACHE_DIR)
        logger.info(f"FastF1 cache enabled at: {APIConfig.FASTF1_CACHE_DIR}")
        
        self.sessions_cache = {}
        
    def get_race_weekend(self, year: int, race_name: str) -> Dict[str, fastf1.core.Session]:
        """
        Get all sessions for a race weekend
        
        Args:
            year: Season year
            race_name: Race name (e.g., 'Monaco', 'British Grand Prix')
            
        Returns:
            Dictionary of session objects keyed by session type
        """
        sessions = {}
        
        try:
            # Get the event
            event = fastf1.get_event(year, race_name)
            logger.info(f"Loading race weekend: {year} {race_name}")
            
            # Load each session type
            for session_type in DataConfig.SESSIONS:
                try:
                    session = fastf1.get_session(year, race_name, session_type)
                    session.load()
                    sessions[session_type] = session
                    logger.info(f"Loaded {session_type} session")
                except Exception as e:
                    logger.warning(f"Could not load {session_type} session: {e}")
                    
        except Exception as e:
            logger.error(f"Error loading race weekend {year} {race_name}: {e}")
            
        return sessions
    
    def get_session_results(self, session: fastf1.core.Session) -> pd.DataFrame:
        """
        Extract results from a session
        
        Args:
            session: FastF1 session object
            
        Returns:
            DataFrame with session results
        """
        try:
            results = session.results
            
            # Add session metadata
            results['session_type'] = session.name
            results['event_name'] = session.event.EventName
            results['year'] = session.event.year
            results['round'] = session.event.RoundNumber
            results['date'] = session.date
            
            return results
            
        except Exception as e:
            logger.error(f"Error extracting session results: {e}")
            return pd.DataFrame()
    
    def get_lap_data(self, session: fastf1.core.Session, driver: Optional[str] = None) -> pd.DataFrame:
        """
        Get lap-by-lap data for a session
        
        Args:
            session: FastF1 session object
            driver: Specific driver (optional, gets all if None)
            
        Returns:
            DataFrame with lap data
        """
        try:
            if driver:
                laps = session.laps.pick_driver(driver)
            else:
                laps = session.laps
            
            # Add session metadata
            laps_df = laps.copy()
            laps_df['session_type'] = session.name
            laps_df['event_name'] = session.event.EventName
            laps_df['year'] = session.event.year
            laps_df['round'] = session.event.RoundNumber
            
            return laps_df
            
        except Exception as e:
            logger.error(f"Error extracting lap data: {e}")
            return pd.DataFrame()
    
    def get_telemetry_data(self, session: fastf1.core.Session, driver: str, 
                          lap_number: int) -> pd.DataFrame:
        """
        Get telemetry data for a specific lap
        
        Args:
            session: FastF1 session object
            driver: Driver abbreviation (e.g., 'VER', 'HAM')
            lap_number: Lap number
            
        Returns:
            DataFrame with telemetry data
        """
        try:
            lap = session.laps.pick_driver(driver).pick_lap(lap_number)
            telemetry = lap.get_telemetry()
            
            # Add metadata
            telemetry['driver'] = driver
            telemetry['lap_number'] = lap_number
            telemetry['session_type'] = session.name
            telemetry['event_name'] = session.event.EventName
            telemetry['year'] = session.event.year
            
            return telemetry
            
        except Exception as e:
            logger.error(f"Error extracting telemetry for {driver} lap {lap_number}: {e}")
            return pd.DataFrame()
    
    def get_fastest_lap_comparison(self, session: fastf1.core.Session, 
                                 drivers: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Compare fastest laps between drivers
        
        Args:
            session: FastF1 session object
            drivers: List of driver abbreviations
            
        Returns:
            Dictionary with telemetry data for each driver's fastest lap
        """
        comparison_data = {}
        
        for driver in drivers:
            try:
                # Get fastest lap
                fastest_lap = session.laps.pick_driver(driver).pick_fastest()
                telemetry = fastest_lap.get_telemetry()
                
                # Add driver info
                telemetry['driver'] = driver
                telemetry['lap_time'] = fastest_lap.LapTime.total_seconds()
                
                comparison_data[driver] = telemetry
                
            except Exception as e:
                logger.warning(f"Could not get fastest lap for {driver}: {e}")
        
        return comparison_data
    
    def get_tire_strategy_data(self, session: fastf1.core.Session) -> pd.DataFrame:
        """
        Extract tire strategy information for all drivers
        
        Args:
            session: FastF1 session object
            
        Returns:
            DataFrame with tire strategy data
        """
        try:
            laps = session.laps
            
            # Get tire compound information
            tire_data = laps[['Driver', 'LapNumber', 'Compound', 'TyreLife', 
                            'FreshTyre', 'LapTime', 'Stint']].copy()
            
            # Add session metadata
            tire_data['session_type'] = session.name
            tire_data['event_name'] = session.event.EventName
            tire_data['year'] = session.event.year
            
            return tire_data
            
        except Exception as e:
            logger.error(f"Error extracting tire strategy data: {e}")
            return pd.DataFrame()
    
    def get_position_data(self, session: fastf1.core.Session) -> pd.DataFrame:
        """
        Get position changes throughout the race
        
        Args:
            session: FastF1 session object
            
        Returns:
            DataFrame with position data over time
        """
        try:
            laps = session.laps
            
            # Get position data
            position_data = laps[['Driver', 'LapNumber', 'Position', 
                                'LapTime', 'Stint', 'Compound']].copy()
            
            # Calculate gaps (simplified)
            position_data = position_data.sort_values(['LapNumber', 'Position'])
            
            # Add session metadata
            position_data['session_type'] = session.name
            position_data['event_name'] = session.event.EventName
            position_data['year'] = session.event.year
            
            return position_data
            
        except Exception as e:
            logger.error(f"Error extracting position data: {e}")
            return pd.DataFrame()
    
    def get_weather_data(self, session: fastf1.core.Session) -> pd.DataFrame:
        """
        Get weather information for the session
        
        Args:
            session: FastF1 session object
            
        Returns:
            DataFrame with weather data
        """
        try:
            weather = session.weather_data
            
            # Add session metadata
            weather['session_type'] = session.name
            weather['event_name'] = session.event.EventName
            weather['year'] = session.event.year
            
            return weather
            
        except Exception as e:
            logger.error(f"Error extracting weather data: {e}")
            return pd.DataFrame()
    
    def get_available_races(self, year: int) -> List[str]:
        """
        Get list of available races for a season
        
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