"""
Race Intelligence Module

This module focuses on collecting and processing data specifically for LLM + RAG applications,
including telemetry, race events, and contextual information.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import timedelta
from loguru import logger
import fastf1
from fastf1.core import Session

class RaceIntelligence:
    """Collects and processes race data for LLM + RAG applications"""
    
    def __init__(self, base_dir: str = "data/processed/race_intelligence"):
        """Initialize the RaceIntelligence collector"""
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure FastF1
        fastf1.Cache.enable_cache("data/raw/race_data/cache")
        
    def process_race_weekend(self, year: int, race_name: str):
        """Process all available data for a race weekend"""
        try:
            # Create output directory
            race_dir = self.base_dir / f"{year}_{race_name.replace(' ', '_')}"
            race_dir.mkdir(exist_ok=True)
            
            # Get the race session
            session = fastf1.get_session(year, race_name, 'R')
            session.load()
            
            # Process different data types
            self._process_race_events(session, race_dir)
            self._process_telemetry(session, race_dir)
            self._process_strategy(session, race_dir)
            self._generate_context(session, race_dir)
            
            logger.success(f"Successfully processed {year} {race_name} for race intelligence")
            
        except Exception as e:
            logger.error(f"Error processing race weekend: {e}")
            raise
    
    def _process_race_events(self, session: Session, output_dir: Path):
        """Process race events (safety cars, pit stops, penalties, etc.)"""
        events = []
        
        # Process pit stops
        if hasattr(session, 'pitstops'):
            for _, stop in session.pitstops.iterrows():
                events.append({
                    'lap': stop['LapNumber'],
                    'time': str(stop['Time']),
                    'driver': stop['Driver'],
                    'event_type': 'PIT_STOP',
                    'details': {
                        'stop_number': stop['Stop'],
                        'duration': stop['Time'].total_seconds(),
                        'tire_change': stop['Compound'] if 'Compound' in stop else None
                    },
                    'context': f"{stop['Driver']} pits on lap {stop['LapNumber']} for {stop.get('Compound', 'tires')}"
                })
        
        # Process safety cars and other track events
        if hasattr(session, 'laps'):
            safety_car = session.laps[session.laps['IsPersonalBest'] == False]
            for _, lap in safety_car.iterrows():
                events.append({
                    'lap': lap['LapNumber'],
                    'time': str(lap['Time']),
                    'driver': lap['DriverNumber'],
                    'event_type': 'SAFETY_CAR',
                    'details': {
                        'lap_time': lap['LapTime'].total_seconds() if pd.notna(lap['LapTime']) else None,
                        'sector': lap['Sector'] if 'Sector' in lap else None
                    },
                    'context': f"Safety car deployed on lap {lap['LapNumber']}"
                })
        
        # Save events
        with open(output_dir / 'race_events.json', 'w') as f:
            json.dump(events, f, indent=2)
    
    def _process_telemetry(self, session: Session, output_dir: Path):
        """Process telemetry data for all drivers"""
        telemetry_dir = output_dir / 'telemetry'
        telemetry_dir.mkdir(exist_ok=True)
        
        for driver in session.drivers:
            try:
                lap = session.laps.pick_driver(driver).pick_fastest()
                if lap is not None and hasattr(lap, 'get_telemetry'):
                    tel = lap.get_telemetry()
                    tel['Driver'] = driver
                    tel.to_csv(telemetry_dir / f"{driver}_telemetry.csv", index=False)
            except Exception as e:
                logger.warning(f"Could not get telemetry for {driver}: {e}")
    
    def _process_strategy(self, session: Session, output_dir: Path):
        """Process race strategy data"""
        strategy_data = []
        
        for driver in session.drivers:
            driver_laps = session.laps.pick_driver(driver)
            if driver_laps.empty:
                continue
                
            stints = []
            current_stint = None
            
            for _, lap in driver_laps.iterrows():
                if current_stint is None or lap['Stint'] != current_stint['stint_number']:
                    if current_stint is not None:
                        stints.append(current_stint)
                    current_stint = {
                        'stint_number': lap['Stint'],
                        'compound': lap['Compound'],
                        'start_lap': lap['LapNumber'],
                        'end_lap': lap['LapNumber'],
                        'laps': 1,
                        'best_lap': lap['LapTime'].total_seconds() if pd.notna(lap['LapTime']) else None,
                        'average_lap': lap['LapTime'].total_seconds() if pd.notna(lap['LapTime']) else None
                    }
                else:
                    current_stint['end_lap'] = lap['LapNumber']
                    current_stint['laps'] += 1
                    if pd.notna(lap['LapTime']):
                        lap_time = lap['LapTime'].total_seconds()
                        if current_stint['best_lap'] is None or lap_time < current_stint['best_lap']:
                            current_stint['best_lap'] = lap_time
                        # Update average lap time
                        if current_stint['average_lap'] is None:
                            current_stint['average_lap'] = lap_time
                        else:
                            current_stint['average_lap'] = (
                                (current_stint['average_lap'] * (current_stint['laps'] - 1) + lap_time) 
                                / current_stint['laps']
                            )
            
            if current_stint is not None:
                stints.append(current_stint)
            
            for stint in stints:
                strategy_data.append({
                    'driver': driver,
                    **stint
                })
        
        # Save strategy data
        pd.DataFrame(strategy_data).to_csv(output_dir / 'strategy_analysis.csv', index=False)
    
    def _generate_context(self, session: Session, output_dir: Path):
        """Generate contextual information for the race"""
        context = {
            'event': {
                'name': session.event.EventName,
                'year': session.event.year,
                'circuit': session.event.Location,
                'date': str(session.date),
                'total_laps': session.total_laps
            },
            'weather': {},
            'key_moments': []
        }
        
        # Add weather information if available
        if hasattr(session, 'weather_data'):
            weather = session.weather_data
            context['weather'] = {
                'air_temperature': {
                    'min': weather['AirTemp'].min(),
                    'max': weather['AirTemp'].max(),
                    'average': weather['AirTemp'].mean()
                },
                'track_temperature': {
                    'min': weather['TrackTemp'].min(),
                    'max': weather['TrackTemp'].max(),
                    'average': weather['TrackTemp'].mean()
                },
                'humidity': {
                    'min': weather['Humidity'].min(),
                    'max': weather['Humidity'].max(),
                    'average': weather['Humidity'].mean()
                },
                'rainfall': 'Yes' if weather['Rainfall'].any() else 'No'
            }
        
        # Save context
        with open(output_dir / 'race_context.json', 'w') as f:
            json.dump(context, f, indent=2)
        
        # Generate a simple race summary
        summary = f"""
        Race: {context['event']['name']} ({context['event']['year']})
        Circuit: {context['event']['circuit']}
        Date: {context['event']['date']}
        Total Laps: {context['event']['total_laps']}
        
        Weather Conditions:
        - Air Temperature: {context['weather'].get('air_temperature', {}).get('average', 'N/A')}°C
        - Track Temperature: {context['weather'].get('track_temperature', {}).get('average', 'N/A')}°C
        - Humidity: {context['weather'].get('humidity', {}).get('average', 'N/A')}%
        - Rain: {context['weather'].get('rainfall', 'No')}
        """
        
        with open(output_dir / 'race_summary.txt', 'w') as f:
            f.write(summary)

def create_hybrid_documents(race_dir: Path) -> List[Dict[str, Any]]:
    """
    Create hybrid documents combining telemetry, events, and context
    for RAG applications
    """
    documents = []
    
    # Load race context
    context_path = race_dir / 'race_context.json'
    if context_path.exists():
        with open(context_path) as f:
            context = json.load(f)
    else:
        context = {}
    
    # Load race events
    events_path = race_dir / 'race_events.json'
    if events_path.exists():
        with open(events_path) as f:
            events = json.load(f)
    else:
        events = []
    
    # Create documents for each lap
    for lap_num in range(1, context.get('event', {}).get('total_laps', 0) + 1):
        # Get events for this lap
        lap_events = [e for e in events if e.get('lap') == lap_num]
        
        # Create document
        doc = {
            'lap': lap_num,
            'context': context,
            'events': lap_events,
            'content': f"Lap {lap_num}: "
        }
        
        # Add event descriptions
        for event in lap_events:
            doc['content'] += f"{event.get('context', '')} "
        
        documents.append(doc)
    
    return documents

if __name__ == "__main__":
    # Example usage
    ri = RaceIntelligence()
    ri.process_race_weekend(2024, "Brazil")
