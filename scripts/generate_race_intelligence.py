"""
Generate Race Intelligence Data

This script processes race data to create rich, contextual information
suitable for LLM + RAG applications.
"""

import sys
from pathlib import Path
from loguru import logger

# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from src.data_pipeline.race_intelligence import RaceIntelligence, create_hybrid_documents

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
)

def main():
    """Main function to generate race intelligence data"""
    try:
        # Initialize the race intelligence processor
        ri = RaceIntelligence()
        
        # Process the race weekend
        year = 2024
        race_name = "Brazil"  # Can be changed or made configurable
        
        logger.info(f"Starting race intelligence generation for {year} {race_name}")
        
        # Process the race weekend
        ri.process_race_weekend(year, race_name)
        
        # Generate hybrid documents for RAG
        race_dir = Path(f"data/processed/race_intelligence/{year}_{race_name.replace(' ', '_')}")
        documents = create_hybrid_documents(race_dir)
        
        # Save the documents
        import json
        with open(race_dir / 'hybrid_documents.json', 'w') as f:
            json.dump(documents, f, indent=2)
        
        logger.success(f"Successfully generated race intelligence data in {race_dir}")
        
    except Exception as e:
        logger.error(f"Error generating race intelligence: {e}")
        raise

if __name__ == "__main__":
    main()
