"""
Enhanced FastF1 Data Client
Handles comprehensive F1 data collection including telemetry, weather, and tire data
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import fastf1
from fastf1 import plotting
from fastf1.core import Laps, Session
from loguru import logger
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
plotting.setup_mpl()

class APIConfig:
    """Configuration for FastF1 API and data storage"""
    BASE_DIR = Path("data/raw/race_data")
    CACHE_DIR = BASE_DIR / "cache"
    
    @classmethod
    def setup_directories(cls):
        """Ensure all required directories exist"""
        # Create base and cache directories if they don't exist
        os.makedirs(cls.BASE_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        
        # Enable cache
        fastf1.Cache.enable_cache(str(cls.CACHE_DIR))
        
        # Only try to clear cache if there are files to clear
        if os.path.exists(cls.CACHE_DIR) and os.listdir(cls.CACHE_DIR):
            try:
                fastf1.Cache.clear_cache(os.path.join(str(cls.CACHE_DIR), '*.pkl'))
            except Exception as e:
                logger.warning(f"Could not clear cache: {e}")

class FastF1Enhanced:
    """Enhanced FastF1 client with comprehensive data collection"""
    
    def __init__(self):
        """Initialize the enhanced FastF1 client"""
        APIConfig.setup_directories()
        self.sessions = {}
        logger.info(f"FastF1 cache enabled at: {APIConfig.CACHE_DIR}")
    
    def get_race_weekend(self, year: int, race_name: str) -> Dict[str, Session]:
        """
        Get all sessions for a race weekend with enhanced data loading
        
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
            
            # Define session types to load
            session_types = ['FP1', 'FP2', 'FP3', 'Q', 'R']
            
            for session_type in session_types:
                try:
                    session = fastf1.get_session(year, race_name, session_type)
                    
                    # Load comprehensive data
                    session.load(
                        telemetry=True,
                        weather=True,
                        messages=True,
                        livedata=None  # Disable live data for reliability
                    )
                    
                    sessions[session_type] = session
                    logger.info(f"Loaded {session_type} session with telemetry")
                    
                except Exception as e:
                    logger.warning(f"Could not load {session_type} session: {e}")
            
            self.sessions = sessions
            return sessions
            
        except Exception as e:
            logger.error(f"Error loading race weekend {year} {race_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}
    
    def get_session_results(self, session: Session) -> pd.DataFrame:
        """
        Get enhanced session results with additional metrics
        
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
    
    def get_lap_data(self, session: Session) -> pd.DataFrame:
        """
        Get comprehensive lap data with enhanced metrics
        
        Args:
            session: FastF1 session object
            
        Returns:
            DataFrame with lap data
        """
        try:
            laps = session.laps
            
            # Convert timedelta to seconds for calculations
            laps['LapTime_sec'] = laps['LapTime'].dt.total_seconds()
            
            # Add stint information
            laps['StintNumber'] = laps['Stint'].rank(method='dense')
            
            # Calculate gaps to leader and car ahead
            if 'Position' in laps.columns:
                leader_times = laps[laps['Position'] == 1].set_index('LapNumber')['LapTime_sec']
                if not leader_times.empty:
                    laps['GapToLeader'] = laps.apply(
                        lambda x: x['LapTime_sec'] - leader_times.get(x['LapNumber'], np.nan),
                        axis=1
                    )
            
            return laps
            
        except Exception as e:
            logger.error(f"Error getting lap data: {e}")
            return pd.DataFrame()
    
    def get_telemetry_data(self, session: Session, driver: str) -> pd.DataFrame:
        """
        Get telemetry data for a specific driver
        
        Args:
            session: FastF1 session object
            driver: Driver abbreviation (e.g., 'VER', 'HAM')
            
        Returns:
            DataFrame with telemetry data
        """
        try:
            # Get the driver's lap
            driver_laps = session.laps.pick_driver(driver)
            if driver_laps.empty:
                return pd.DataFrame()
                
            # Get telemetry for the fastest lap
            fastest_lap = driver_laps.pick_fastest()
            telemetry = fastest_lap.get_telemetry()
            
            # Add driver info
            telemetry['Driver'] = driver
            telemetry['Session'] = session.name
            
            return telemetry
            
        except Exception as e:
            logger.error(f"Error getting telemetry for {driver}: {e}")
            return pd.DataFrame()
    
    def get_tire_strategy_data(self, session: Session) -> pd.DataFrame:
        """
        Get comprehensive tire strategy data
        
        Args:
            session: FastF1 session object
            
        Returns:
            DataFrame with tire strategy information
        """
        try:
            laps = session.laps
            if laps.empty:
                return pd.DataFrame()
                
            # Get pit stops if available
            pit_stops = getattr(session, 'pitstops', pd.DataFrame())
            
            # Prepare tire data
            tire_data = []
            
            # Group by driver and stint
            for (driver, stint), stint_laps in laps.groupby(['Driver', 'Stint']):
                if stint_laps.empty:
                    continue
                    
                stint_data = {
                    'Driver': driver,
                    'Stint': stint,
                    'StartLap': stint_laps['LapNumber'].min(),
                    'EndLap': stint_laps['LapNumber'].max(),
                    'Laps': len(stint_laps),
                    'Compound': stint_laps['Compound'].iloc[0],
                    'TireAgeAtStart': stint_laps['TyreLife'].iloc[0],
                    'TireAgeAtEnd': stint_laps['TyreLife'].iloc[-1],
                }
                
                # Add pit stop information if available
                if not pit_stops.empty:
                    driver_pits = pit_stops[pit_stops['Driver'] == driver]
                    if not driver_pits.empty:
                        pit = driver_pits[driver_pits['LapNumber'] == stint_data['EndLap']]
                        if not pit.empty:
                            stint_data.update({
                                'PitDuration': pit['Time'].iloc[0].total_seconds(),
                                'PitLap': pit['LapNumber'].iloc[0],
                                'TireChange': pit['NewTyre'].iloc[0],
                            })
                
                tire_data.append(stint_data)
            
            return pd.DataFrame(tire_data)
            
        except Exception as e:
            logger.error(f"Error getting tire strategy data: {e}")
            return pd.DataFrame()
    
    def get_position_data(self, session: Session) -> pd.DataFrame:
        """
        Get position data for all drivers
        
        Args:
            session: FastF1 session object
            
        Returns:
            DataFrame with position data over time
        """
        try:
            position_data = []
            
            for driver in session.drivers:
                try:
                    driver_laps = session.laps.pick_driver(driver)
                    if not driver_laps.empty:
                        positions = driver_laps[['LapNumber', 'Position', 'Time']].copy()
                        positions['Driver'] = driver
                        position_data.append(positions)
                except Exception as e:
                    logger.warning(f"Could not get position data for {driver}: {e}")
            
            if not position_data:
                return pd.DataFrame()
                
            return pd.concat(position_data).sort_values(['LapNumber', 'Position'])
            
        except Exception as e:
            logger.error(f"Error getting position data: {e}")
            return pd.DataFrame()

# Alias for backward compatibility
FastF1Client = FastF1Enhanced
