"""
ApexAI Configuration Settings
Central configuration for the entire project
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# API Keys and External Services
class APIConfig:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
    
    # FastF1 cache directory
    FASTF1_CACHE_DIR = str(DATA_DIR / "fastf1_cache")

# Data Collection Settings
class DataConfig:
    # Years and seasons to collect data for
    SEASONS = [2022, 2023, 2024]
    
    # How often to refresh data (in hours)
    DATA_REFRESH_INTERVAL = 24
    
    # Race weekend sessions to collect
    SESSIONS = ["FP1", "FP2", "FP3", "Q", "S", "R"]  # Sprint sessions included
    
    # Telemetry data sampling rate
    TELEMETRY_SAMPLE_RATE = 50  # Hz

# Model Configuration
class ModelConfig:
    # Tire degradation model
    TIRE_MODEL_FEATURES = [
        "compound", "age", "stint_length", "track_temp", 
        "air_temp", "driver_style", "fuel_load"
    ]
    
    # Prediction horizons
    TIRE_PREDICTION_HORIZON = 30  # laps
    STRATEGY_PREDICTION_HORIZON = 50  # laps
    
    # Model hyperparameters
    LGBM_PARAMS = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": 0
    }

# LLM Configuration
class LLMConfig:
    # Model selection
    PRIMARY_MODEL = "gpt-4"  # or "claude-3-sonnet"
    FALLBACK_MODEL = "gpt-3.5-turbo"
    
    # Generation parameters
    MAX_TOKENS = 1000
    TEMPERATURE = 0.3
    TOP_P = 0.9
    
    # RAG settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    SIMILARITY_THRESHOLD = 0.7
    MAX_RETRIEVED_DOCS = 5

# Dashboard Configuration
class DashboardConfig:
    # Streamlit settings
    PAGE_TITLE = "ApexAI - F1 Race Intelligence"
    PAGE_ICON = "🏁"
    LAYOUT = "wide"
    
    # Update intervals (in seconds)
    LIVE_UPDATE_INTERVAL = 10
    CHART_UPDATE_INTERVAL = 5
    
    # Visualization settings
    PLOTLY_THEME = "plotly_dark"
    COLOR_SCHEME = {
        "primary": "#FF1E1E",  # F1 Red
        "secondary": "#FFFFFF",
        "accent": "#FFD700",
        "background": "#1E1E1E"
    }

# Race Strategy Configuration
class StrategyConfig:
    # Pit stop parameters
    PIT_STOP_TIME = 25.0  # seconds (including pit lane)
    PIT_STOP_VARIANCE = 2.0  # seconds
    
    # Tire compounds and performance
    TIRE_COMPOUNDS = {
        "SOFT": {"grip": 1.0, "degradation": 0.05, "color": "#FF3333"},
        "MEDIUM": {"grip": 0.97, "degradation": 0.03, "color": "#FFD700"},
        "HARD": {"grip": 0.94, "degradation": 0.02, "color": "#FFFFFF"}
    }
    
    # Strategy constraints
    MIN_TIRE_STINTS = 1  # Minimum number of different compounds
    MAX_PIT_STOPS = 4
    
    # Weather impact factors
    RAIN_GRIP_FACTOR = 0.85
    TEMPERATURE_DEGRADATION_FACTOR = 0.02  # per degree C

# Logging Configuration
class LoggingConfig:
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE = str(LOGS_DIR / "apexai.log")
    MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB
    BACKUP_COUNT = 5

# Performance Monitoring
class PerformanceConfig:
    # Prediction accuracy thresholds
    TIRE_PREDICTION_ACCURACY_THRESHOLD = 0.85
    STRATEGY_PREDICTION_ACCURACY_THRESHOLD = 0.75
    
    # System performance
    MAX_RESPONSE_TIME = 2.0  # seconds
    MAX_MEMORY_USAGE = 2048  # MB

# Development Settings
class DevConfig:
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    TESTING = os.getenv("TESTING", "False").lower() == "true"
    
    # For development, use smaller datasets
    DEV_RACE_LIMIT = 5 if DEBUG else None
    
    # Mock data for testing
    USE_MOCK_DATA = TESTING

# Export all configs
__all__ = [
    "APIConfig",
    "DataConfig", 
    "ModelConfig",
    "LLMConfig",
    "DashboardConfig",
    "StrategyConfig",
    "LoggingConfig",
    "PerformanceConfig",
    "DevConfig",
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "MODELS_DIR",
    "LOGS_DIR"
]