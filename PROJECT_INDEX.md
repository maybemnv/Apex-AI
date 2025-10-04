# 🏁 ApexAI Project Index

> **Comprehensive index of the F1 Race Intelligence System components**

## 📂 Project Structure Overview

```
ApexAI/
├── data/                          # Data storage and processing
├── deployment/                    # Docker and production configs
├── docs/                         # Documentation and guides
├── notebooks/                    # Jupyter notebooks for analysis
├── scripts/                      # Automation and utility scripts
└── src/                          # Main source code
```

## 📁 Detailed Directory Structure

### `/data/` - Data Layer
```
data/
├── processed/
│   └── race_data.db              # SQLite database with processed F1 data
└── raw/                          # Raw data cache (created by FastF1)
```
**Purpose**: Persistent storage for F1 race data, both raw and processed formats.

### `/deployment/` - Production Infrastructure
```
deployment/
├── docker-compose.yml            # Multi-container orchestration
├── Dockerfile                    # Container image definition
├── nginx.conf                    # Reverse proxy configuration
└── requirements-prod.txt         # Production dependencies
```
**Purpose**: Production deployment configurations for containerized environments.

### `/docs/` - Documentation Hub
```
docs/
├── ApexAi's first draft.txt      # Initial project concept
├── api_documentation.md          # REST API reference (WIP)
├── model_documentation.md        # ML model specifications (WIP)
├── README.md                     # Main documentation (empty)
├── setup_guide.md               # Installation instructions (WIP)
├── Tasks.txt                    # 28-day development sprint plan
└── user_manual.md               # User interface guide (WIP)
```
**Purpose**: Comprehensive project documentation and development planning.

### `/notebooks/` - Data Science Workspace
```
notebooks/
├── 01_data_exploration.ipynb     # Initial F1 data analysis
├── 02_tire_analysis.ipynb        # Tire degradation modeling
├── 03_strategy_modeling.ipynb    # Race strategy optimization
├── 04_llm_testing.ipynb          # LLM integration experiments
└── 05_model_evaluation.ipynb     # Model performance metrics
```
**Purpose**: Interactive data science environment for model development and analysis.

### `/scripts/` - Automation Tools
```
scripts/
├── __pycache__/                  # Python bytecode cache
│   ├── export_fastf1_data.cpython-311.pyc
│   └── __init__.cpython-311.pyc
├── download_data.py              # F1 data acquisition utility
├── export_fastf1_data.py         # FastF1 data export functions
├── run_tests.py                  # Test runner script
├── setup_database.py             # Database initialization
├── train_models.py               # ML model training pipeline
├── update_knowledge_base.py      # RAG knowledge base management
└── __init__.py                   # Package initialization
```
**Purpose**: Automated workflows for data management, model training, and system maintenance.

### `/src/` - Core Application Code

#### `/src/api/` - Backend Services
```
api/
├── main.py                       # FastAPI application entry point (empty)
└── __init__.py                   # Package initialization
```
**Purpose**: RESTful API endpoints for AutoPodium and PitSynth services.

#### `/src/config/` - Configuration Management
```
config/
├── config.py                     # Application configuration classes
└── __init__.py                   # Package initialization
```
**Purpose**: Centralized configuration for API endpoints, data sources, and ML models.

#### `/src/dashboard/` - Frontend Interface
```
dashboard/
├── app.py                        # Streamlit dashboard application
└── __init__.py                   # Package initialization
```
**Purpose**: Interactive web dashboard for F1 race visualization and AI insights.

#### `/src/data_pipeline/` - Data Processing Engine
```
data_pipeline/
├── database.py                   # Database connection and ORM
├── data_processor.py             # Data cleaning and transformation
├── fastf1_client.py              # FastF1 API integration
├── feature_engineering.py       # ML feature creation
├── Teampace.py                   # Team performance analytics
└── __init__.py                   # Package initialization
```
**Purpose**: ETL pipeline for F1 data ingestion, processing, and feature engineering.

#### `/src/llm/` - AI Intelligence Layer
```
llm/
├── knowledge_base.py             # RAG system knowledge management
├── pit_synth.py                  # AI race commentary generator
├── prompts.py                    # LLM prompt templates
├── rag_system.py                 # Retrieval-Augmented Generation
├── validators.py                 # AI response validation
└── __init__.py                   # Package initialization
```
**Purpose**: LLM-powered race analysis, commentary generation, and intelligent Q&A system.

#### `/src/modeling/` - Machine Learning Core
```
modeling/
├── base_model.py                 # ML model base classes
├── position_predictor.py         # Race position forecasting
├── race_simulator.py             # Race scenario simulation
├── strategy_optimizer.py        # Pit stop strategy optimization
├── tire_model.py                 # Tire degradation prediction
└── __init__.py                   # Package initialization
```
**Purpose**: ML models for race strategy prediction, position forecasting, and tire analysis.

## 🔧 Key Components

### 1. AutoPodium Engine (ML Core)
- **Location**: `src/modeling/`
- **Components**:
  - `tire_model.py` - Tire degradation forecasting
  - `strategy_optimizer.py` - Optimal pit window calculations
  - `position_predictor.py` - Race position predictions
  - `race_simulator.py` - What-if scenario analysis

### 2. PitSynth (AI Commentary)
- **Location**: `src/llm/`
- **Components**:
  - `pit_synth.py` - GPT-4 powered race analysis
  - `rag_system.py` - F1 knowledge retrieval system
  - `knowledge_base.py` - F1 domain expertise storage
  - `prompts.py` - Specialized AI prompts

### 3. Race Dashboard (Frontend)
- **Location**: `src/dashboard/`
- **Components**:
  - `app.py` - Main Streamlit application
  - Interactive F1 data visualization
  - Real-time race monitoring interface

### 4. Data Pipeline (ETL)
- **Location**: `src/data_pipeline/`
- **Components**:
  - `fastf1_client.py` - F1 telemetry data access
  - `data_processor.py` - Data cleaning and validation
  - `feature_engineering.py` - ML feature preparation

## 🚀 Development Workflow

### Data Flow Architecture
```
FastF1 API → data_pipeline → SQLite DB → modeling → API → dashboard
     ↓                                           ↑
Ergast API → knowledge_base → RAG → LLM → PitSynth
```

### Key Integration Points
1. **Data Layer**: FastF1 + Ergast APIs → SQLite Database
2. **ML Layer**: Processed data → Trained models → Predictions
3. **AI Layer**: Knowledge base → RAG system → LLM commentary
4. **API Layer**: FastAPI endpoints expose ML and AI services
5. **Frontend Layer**: Streamlit dashboard consumes API services

### Development Environment
- **Local**: Frontend development, code editing, version control
- **Google Colab**: ML training, LLM operations, backend API hosting
- **Cloud Integration**: ngrok tunneling for API access

## 📊 Data Sources & APIs

### Primary Data Sources
- **FastF1**: Real-time F1 telemetry and timing data
- **Ergast**: Historical F1 race database
- **OpenAI API**: GPT-4 for race commentary
- **Custom Knowledge Base**: F1 regulations, strategies, driver profiles

### Data Storage
- **SQLite Database**: `data/processed/race_data.db`
- **File Cache**: `data/raw/` for FastF1 cached data
- **Vector Database**: ChromaDB for RAG embeddings

## 🔗 External Dependencies

### Core Libraries
- **Data Processing**: pandas, numpy, fastf1, ergast-py
- **Machine Learning**: scikit-learn, lightgbm, xgboost
- **AI/LLM**: openai, langchain, chromadb
- **Web Framework**: streamlit, fastapi, uvicorn
- **Visualization**: plotly, matplotlib, seaborn

### Development Tools
- **Environment**: Google Colab, ngrok
- **Database**: SQLite, SQLAlchemy
- **Deployment**: Docker, nginx

## 📈 Project Status

### Completed Components
- ✅ Project structure and documentation
- ✅ Streamlit dashboard framework
- ✅ Data pipeline architecture
- ✅ Configuration management

### In Development
- 🔄 ML model implementations
- 🔄 FastAPI backend services
- 🔄 LLM integration and RAG system
- 🔄 Advanced visualizations

### Planned Features
- 📋 Real-time data streaming
- 📋 Mobile-responsive dashboard
- 📋 Multi-language support
- 📋 Social features and predictions

## 🎯 Use Cases

### For Casual F1 Fans
- Simple race summaries and explanations
- "Who's winning and why" insights
- Interactive F1 glossary
- Visual race progress tracking

### For Hardcore F1 Enthusiasts
- Detailed tire degradation analysis
- Pit window optimization strategies
- Historical performance comparisons
- Advanced telemetry visualizations

### For Data Scientists
- Comprehensive F1 dataset access
- ML model experimentation notebooks
- Feature engineering pipelines
- Model performance evaluation tools