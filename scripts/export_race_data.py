import os
import sys
import traceback
from pathlib import Path
from datetime import datetime
import pandas as pd
from loguru import logger

# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from src.data_pipeline.fastf1_enhanced import FastF1Client

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
)

def ensure_dir(directory):
    """Ensure directory exists"""
    os.makedirs(directory, exist_ok=True)

def export_race_data(year: int, race_name: str):
    """
    Export all available data for a race weekend
    
    Args:
        year: Race year
        race_name: Name of the race (e.g., 'Brazilian Grand Prix')
    """
    # Set up directories
    base_dir = Path("data/raw/race_data")
    race_dir = base_dir / f"{year}_{race_name.replace(' ', '_')}"
    
    # Create necessary subdirectories
    dirs = {
        'sessions': race_dir / 'sessions',
        'telemetry': race_dir / 'telemetry',
        'weather': race_dir / 'weather',
        'tires': race_dir / 'tires',
        'positions': race_dir / 'positions'
    }
    
    for d in dirs.values():
        ensure_dir(d)
    
    logger.info(f"Starting data export for {year} {race_name}")
    
    # Initialize client
    client = FastF1Client()
    
    # Get all sessions for the race weekend
    sessions = client.get_race_weekend(year, race_name)
    
    if not sessions:
        logger.error(f"No sessions found for {year} {race_name}")
        return
    
    # Process each session
    for session_type, session in sessions.items():
        try:
            session_dir = dirs['sessions'] / session_type
            ensure_dir(session_dir)
            
            logger.info(f"Processing {session_type} session")
            
            # 1. Save session results
            results = client.get_session_results(session)
            if not results.empty:
                results.to_csv(session_dir / 'results.csv', index=False)
            
            # 2. Save lap data with enhanced metrics
            laps = client.get_lap_data(session)
            if not laps.empty:
                laps.to_csv(session_dir / 'laps.csv', index=False)
            
            # 3. Save telemetry data for each driver
            telemetry_dir = dirs['telemetry'] / session_type
            ensure_dir(telemetry_dir)
            
            for driver in session.drivers:
                try:
                    telemetry = client.get_telemetry_data(session, driver)
                    if not telemetry.empty:
                        telemetry.to_csv(telemetry_dir / f"{driver}_telemetry.csv", index=False)
                except Exception as e:
                    logger.error(f"Error getting telemetry for {driver}: {e}")
            
            # 4. Save weather data
            if hasattr(session, 'weather_data'):
                weather = session.weather_data
                weather.to_csv(dirs['weather'] / f"{session_type}_weather.csv", index=False)
            
            # 5. Save tire data
            tires = client.get_tire_strategy_data(session)
            if not tires.empty:
                tires.to_csv(dirs['tires'] / f"{session_type}_tires.csv", index=False)
            
            # 6. Save position data
            positions = client.get_position_data(session)
            if not positions.empty:
                positions.to_csv(dirs['positions'] / f"{session_type}_positions.csv", index=False)
            
            logger.info(f"Completed processing {session_type} session")
            
        except Exception as e:
            logger.error(f"Error processing {session_type} session: {e}")
            logger.error(traceback.format_exc())
    
    logger.success(f"Completed export for {year} {race_name}")

if __name__ == "__main__":
    # Example: export_race_data(2024, "Monaco")
    export_race_data(2024, "Brazil")
