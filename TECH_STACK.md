# 🔧 ApexAI Technical Architecture & Stack
## Comprehensive Technology Documentation

> **Detailed breakdown of the technology stack, architecture patterns, and engineering decisions in ApexAI**

---

## 🏗️ **System Architecture Overview**

### **High-Level Architecture**
```
┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│   Presentation  │   Application   │   Intelligence  │   Data Layer    │
├─────────────────┼─────────────────┼─────────────────┼─────────────────┤
│ Streamlit UI    │ FastAPI Server  │ ML Models       │ FastF1 API      │
│ React Frontend  │ Business Logic  │ LLM Integration │ Ergast API      │
│ REST Endpoints  │ Authentication  │ RAG System      │ SQLite DB       │
│ WebSocket       │ Rate Limiting   │ Vector Store    │ File Cache      │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┘
```

### **Deployment Architecture**
```
┌─────────────────────────────────────────────────────────────────┐
│                           Cloud Layer                           │
├─────────────────────┬─────────────────────┬─────────────────────┤
│   Google Colab      │   ngrok Tunnel      │   OpenAI API        │
│   - ML Training     │   - Public Access   │   - LLM Services    │
│   - API Hosting     │   - HTTPS Proxy     │   - Embeddings      │
│   - GPU Compute     │   - Load Balancing  │   - Fine-tuning     │
└─────────────────────┴─────────────────────┴─────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│                          Local Layer                            │
├─────────────────────┬─────────────────────┬─────────────────────┤
│   Development       │   Data Storage      │   User Interface    │
│   - Code Editor     │   - SQLite DB       │   - Streamlit App   │
│   - Git Repository  │   - FastF1 Cache    │   - Web Browser     │
│   - Virtual Env     │   - Model Files     │   - API Client      │
└─────────────────────┴─────────────────────┴─────────────────────┘
```

---

## 🔍 **Technology Stack Breakdown**

### **Frontend Technologies**

#### **Streamlit Framework** `1.28+`
```python
# Core dashboard framework
import streamlit as st

# Key features utilized:
- Interactive widgets (sliders, selectboxes, multiselect)
- Real-time data refresh with session state
- Custom CSS styling for F1 branding
- Multi-tab layouts for different analysis views
- Caching decorators for performance optimization
```

**Why Streamlit?**
- ✅ Rapid prototyping for data science applications
- ✅ Built-in widgets perfect for F1 data interaction
- ✅ Automatic reactive updates when data changes
- ✅ Easy deployment to Streamlit Cloud
- ❌ Limited customization compared to React
- ❌ Not mobile-optimized out of the box

#### **Plotly.js** `5.17+` (via Python bindings)
```python
import plotly.graph_objects as go
import plotly.express as px

# Visualization types used:
- Line charts: Lap time progression
- Scatter plots: Tire degradation analysis
- Bar charts: Position changes and pit stop timing
- Heatmaps: Driver performance matrices
- 3D surfaces: Strategy optimization landscapes
```

### **Backend Technologies**

#### **FastAPI Framework** `0.104+`
```python
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware

# API structure:
├── /autopodium/          # ML-powered predictions
│   ├── /tire-degradation
│   ├── /optimize-strategy
│   └── /predict-positions
├── /pitsynth/            # AI commentary
│   ├── /ask
│   ├── /commentary
│   └── /knowledge-base
└── /data/                # Data management
    ├── /races
    ├── /sessions
    └── /drivers
```

**API Design Patterns:**
- RESTful endpoint structure
- Pydantic models for request/response validation
- Async/await for concurrent request handling
- Dependency injection for database connections
- JWT authentication for secure access

#### **Uvicorn ASGI Server** `0.24+`
```bash
# Production deployment
uvicorn src.api.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --access-log \
  --reload-dir src/
```

### **Data Layer Technologies**

#### **FastF1** `3.1+` - F1 Telemetry Data
```python
import fastf1
from fastf1 import plotting

# Core capabilities:
- Real-time and historical F1 session data
- Lap times, sector times, and mini-sectors
- Telemetry: speed, throttle, brake, DRS, gear
- Track position and overtaking analysis
- Weather and track condition data
```

**Data Pipeline:**
```python
# Enable caching for performance
fastf1.Cache.enable_cache('data/raw/fastf1_cache')

# Load session data
session = fastf1.get_session(2024, 'Monaco', 'R')
session.load()

# Extract telemetry
laps = session.laps
telemetry = laps.get_car_data(car='44')  # Hamilton's car
```

#### **Ergast API** - Historical F1 Database
```python
import requests

# REST API endpoints:
- Race results: /api/f1/2024/races.json
- Driver standings: /api/f1/2024/driverStandings.json
- Constructor data: /api/f1/2024/constructorStandings.json
- Race schedules: /api/f1/2024/races/1/results.json
```

#### **SQLite Database** `3.40+`
```sql
-- Core database schema
CREATE TABLE races (
    id INTEGER PRIMARY KEY,
    year INTEGER,
    round INTEGER,
    name TEXT,
    date TEXT,
    circuit_id TEXT
);

CREATE TABLE lap_times (
    id INTEGER PRIMARY KEY,
    race_id INTEGER,
    driver_id TEXT,
    lap_number INTEGER,
    lap_time REAL,
    sector1_time REAL,
    sector2_time REAL,
    sector3_time REAL,
    tire_compound TEXT,
    tire_age INTEGER,
    FOREIGN KEY (race_id) REFERENCES races (id)
);

CREATE TABLE predictions (
    id INTEGER PRIMARY KEY,
    race_id INTEGER,
    model_type TEXT,
    prediction_data JSON,
    confidence_score REAL,
    created_at TIMESTAMP
);
```

#### **SQLAlchemy ORM** `2.0+`
```python
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class LapTime(Base):
    __tablename__ = 'lap_times'
    
    id = Column(Integer, primary_key=True)
    driver_id = Column(String(10), nullable=False)
    lap_number = Column(Integer, nullable=False)
    lap_time = Column(Float, nullable=False)
    tire_compound = Column(String(20))
    tire_age = Column(Integer)
```

---

## 🧠 **Machine Learning Stack**

### **Core ML Libraries**

#### **Scikit-learn** `1.3+` - Traditional ML
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Model pipelines used:
tire_degradation_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', RandomForestRegressor(n_estimators=100))
])

# Key algorithms:
- Random Forest: Tire degradation prediction
- Linear Regression: Lap time modeling
- KMeans Clustering: Driver behavior analysis
- PCA: Dimensionality reduction for telemetry
```

#### **LightGBM** `4.1+` - Gradient Boosting
```python
import lightgbm as lgb

# Model training configuration
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

# Use cases:
- Tire performance prediction
- Lap time forecasting
- Position change probability
- Pit stop timing optimization
```

#### **XGBoost** `2.0+` - Alternative Boosting
```python
import xgboost as xgb

# Configuration for race strategy
xgb_params = {
    'max_depth': 6,
    'eta': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'reg:squarederror'
}

# Applications:
- Strategy outcome prediction
- Weather impact modeling  
- Safety car probability
- Championship points prediction
```

### **Feature Engineering Pipeline**
```python
# Feature extraction from telemetry
def extract_features(lap_data, telemetry_data):
    features = {
        # Tire-related features
        'tire_age_laps': lap_data['TyreLife'],
        'compound_categorical': lap_data['Compound'],
        'stint_number': lap_data['Stint'],
        
        # Performance features
        'relative_pace': calculate_relative_pace(lap_data),
        'sector_consistency': calculate_sector_variance(telemetry_data),
        'throttle_aggression': calculate_throttle_metrics(telemetry_data),
        
        # Track condition features
        'track_temperature': lap_data['TrackTemp'],
        'air_temperature': lap_data['AirTemp'],
        'track_evolution': calculate_track_evolution(lap_data),
        
        # Strategic features
        'fuel_corrected_time': estimate_fuel_effect(lap_data),
        'traffic_impact': calculate_traffic_effect(lap_data),
        'drs_availability': telemetry_data['DRS'].mean()
    }
    return features
```

### **Model Training Pipeline**
```python
# Automated model training workflow
def train_tire_degradation_model():
    # Data loading and preprocessing
    data = load_race_data(years=[2022, 2023, 2024])
    features, targets = prepare_training_data(data)
    
    # Cross-validation setup
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Model selection
    models = {
        'random_forest': RandomForestRegressor(),
        'lightgbm': lgb.LGBMRegressor(),
        'xgboost': xgb.XGBRegressor()
    }
    
    # Hyperparameter tuning
    for name, model in models.items():
        param_grid = get_param_grid(name)
        grid_search = GridSearchCV(
            model, param_grid, cv=kfold, 
            scoring='neg_mean_absolute_error'
        )
        grid_search.fit(features, targets)
        
        # Save best model
        save_model(grid_search.best_estimator_, f"{name}_tire_model.pkl")
    
    # Model evaluation
    evaluate_models(models, features, targets)
```

---

## 🤖 **AI & LLM Integration Stack**

### **Large Language Model APIs**

#### **OpenAI GPT Integration** `1.3+`
```python
import openai
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Model configurations used:
models_config = {
    'gpt-4o-mini': {
        'use_case': 'Race commentary generation',
        'max_tokens': 1500,
        'temperature': 0.3,
        'cost_per_1k_tokens': 0.00015
    },
    'gpt-3.5-turbo': {
        'use_case': 'Q&A and quick responses',
        'max_tokens': 1000,
        'temperature': 0.2,
        'cost_per_1k_tokens': 0.0005
    },
    'text-embedding-3-small': {
        'use_case': 'RAG embeddings',
        'dimensions': 1536,
        'cost_per_1k_tokens': 0.00002
    }
}
```

#### **LangChain Framework** `0.0.350+`
```python
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# RAG pipeline setup
def setup_rag_chain():
    # Document loading and chunking
    documents = load_f1_knowledge_base()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    
    # Embedding and vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="data/vectorstore"
    )
    
    # QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0.3),
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    
    return qa_chain
```

### **Vector Database & Embeddings**

#### **ChromaDB** `0.4.15+` - Vector Storage
```python
import chromadb
from chromadb.config import Settings

# Initialize persistent client
client = chromadb.PersistentClient(
    path="data/vectorstore",
    settings=Settings(anonymized_telemetry=False)
)

# Create F1 knowledge collection
collection = client.create_collection(
    name="f1_knowledge",
    metadata={"hnsw:space": "cosine"}
)

# Document ingestion pipeline
def ingest_f1_documents():
    documents = [
        {
            'content': regulation_text,
            'metadata': {
                'source': 'FIA_regulations_2024.pdf',
                'category': 'technical_regulations'
            }
        },
        {
            'content': strategy_analysis,
            'metadata': {
                'source': 'monaco_gp_analysis.md',
                'category': 'race_analysis'
            }
        }
    ]
    
    # Generate embeddings and store
    collection.add(
        documents=[doc['content'] for doc in documents],
        metadatas=[doc['metadata'] for doc in documents],
        ids=[f"doc_{i}" for i in range(len(documents))]
    )
```

### **RAG System Architecture**
```python
class F1RagSystem:
    def __init__(self):
        self.vectorstore = self._setup_vectorstore()
        self.llm = self._setup_llm()
        self.retriever = self._setup_retriever()
    
    def query(self, question: str, context_type: str = "general") -> str:
        # Retrieve relevant documents
        relevant_docs = self.retriever.get_relevant_documents(
            question, 
            filter={'category': context_type}
        )
        
        # Construct context-aware prompt
        context = "\n".join([doc.page_content for doc in relevant_docs])
        prompt = self._build_prompt(question, context)
        
        # Generate response
        response = self.llm(prompt)
        
        # Post-process and validate
        validated_response = self._validate_response(response, relevant_docs)
        
        return validated_response
    
    def _build_prompt(self, question: str, context: str) -> str:
        return f"""
        Context: You are PitSynth, an expert F1 race analyst with deep knowledge 
        of racing strategy, technical regulations, and driver performance.
        
        Relevant F1 Knowledge:
        {context}
        
        Question: {question}
        
        Provide a detailed, accurate answer based on the context above. 
        Include specific data points when available and explain technical 
        concepts in an accessible way.
        """
```

---

## ⚡ **Performance & Optimization**

### **Caching Strategies**

#### **Streamlit Caching** 
```python
import streamlit as st

@st.cache_data(ttl=300)  # 5-minute cache
def load_race_data(year: int, race_name: str):
    """Cache expensive data loading operations"""
    return fetch_and_process_race_data(year, race_name)

@st.cache_resource
def initialize_models():
    """Cache model initialization (persistent across sessions)"""
    return {
        'tire_model': load_model('tire_degradation.pkl'),
        'strategy_model': load_model('strategy_optimizer.pkl')
    }
```

#### **FastAPI Response Caching**
```python
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache

@app.on_event("startup")
async def startup():
    redis = aioredis.from_url("redis://localhost")
    FastAPICache.init(RedisBackend(redis), prefix="apexai-cache")

@router.get("/autopodium/tire-degradation")
@cache(expire=300)  # 5-minute cache
async def predict_tire_degradation(request: TirePredictionRequest):
    prediction = tire_model.predict(request.features)
    return {"degradation": prediction, "confidence": 0.85}
```

#### **Database Query Optimization**
```sql
-- Index optimization for common queries
CREATE INDEX idx_lap_times_driver_race ON lap_times(driver_id, race_id);
CREATE INDEX idx_lap_times_race_lap ON lap_times(race_id, lap_number);
CREATE INDEX idx_predictions_race_model ON predictions(race_id, model_type);

-- Query optimization examples
EXPLAIN QUERY PLAN 
SELECT * FROM lap_times 
WHERE race_id = 1001 AND driver_id = 'HAM'
ORDER BY lap_number;
```

### **Memory & Processing Optimization**

#### **Pandas Optimization**
```python
import pandas as pd
import numpy as np

# Memory-efficient data types
def optimize_dataframe(df):
    """Optimize DataFrame memory usage"""
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category')
        elif df[col].dtype == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif df[col].dtype == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
    
    return df

# Chunked processing for large datasets
def process_large_dataset(file_path, chunk_size=10000):
    """Process large CSV files in chunks"""
    results = []
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        processed_chunk = process_chunk(chunk)
        results.append(processed_chunk)
    
    return pd.concat(results, ignore_index=True)
```

#### **Async Processing**
```python
import asyncio
import aiohttp

async def fetch_multiple_sessions(year: int, races: List[str]):
    """Fetch multiple race sessions concurrently"""
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_session_data(session, year, race) 
            for race in races
        ]
        results = await asyncio.gather(*tasks)
    
    return results

# Background task processing
from fastapi import BackgroundTasks

@router.post("/autopodium/train-model")
async def train_model_endpoint(background_tasks: BackgroundTasks):
    background_tasks.add_task(train_tire_model_async)
    return {"message": "Model training started in background"}
```

---

## 🔒 **Security & Authentication**

### **API Security**
```python
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
import bcrypt

# JWT authentication
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(
            credentials.credentials, 
            SECRET_KEY, 
            algorithms=[ALGORITHM]
        )
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@router.post("/pitsynth/ask")
@limiter.limit("10/minute")
async def ask_question(request: Request, question: QuestionRequest):
    """Rate-limited AI question endpoint"""
    return await process_ai_question(question)
```

### **Environment Configuration**
```python
from pydantic import BaseSettings

class Settings(BaseSettings):
    # API Keys (from environment)
    openai_api_key: str
    fastf1_api_key: Optional[str] = None
    
    # Database
    database_url: str = "sqlite:///data/apexai.db"
    
    # Security
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Rate limiting
    api_rate_limit: str = "100/minute"
    ai_rate_limit: str = "20/minute"
    
    class Config:
        env_file = ".env"

settings = Settings()
```

---

## 📊 **Monitoring & Observability**

### **Application Monitoring**
```python
import logging
from prometheus_client import Counter, Histogram, generate_latest

# Metrics collection
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('api_request_duration_seconds', 'API request latency')

@REQUEST_LATENCY.time()
def process_prediction_request(request):
    REQUEST_COUNT.labels(method='POST', endpoint='/autopodium/predict').inc()
    # Process request
    return prediction_result

# Structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/apexai.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("apexai")
```

### **Performance Profiling**
```python
import cProfile
import pstats
from functools import wraps

def profile_function(func):
    """Decorator for function profiling"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        result = func(*args, **kwargs)
        pr.disable()
        
        stats = pstats.Stats(pr)
        stats.sort_stats('cumulative')
        stats.print_stats(10)
        
        return result
    return wrapper

@profile_function
def expensive_ml_operation():
    # Profile this function's performance
    return model.predict(features)
```

---

## 🚀 **Deployment & DevOps Stack**

### **Containerization**
```dockerfile
# Multi-stage Dockerfile
FROM python:3.10-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.10-slim as runtime

WORKDIR /app
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY src/ ./src/
COPY data/ ./data/

EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### **Docker Compose Configuration**
```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=sqlite:///data/apexai.db
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      - redis
  
  dashboard:
    build: .
    command: streamlit run src/dashboard/app.py --server.port 8501 --server.address 0.0.0.0
    ports:
      - "8501:8501"
    depends_on:
      - api
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
  
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./deployment/nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - api
      - dashboard
```

### **Cloud Deployment Options**

#### **Google Colab Production Setup**
```python
# Persistent Colab deployment
!git clone https://github.com/user/apexai.git
!cd apexai && pip install -r requirements.txt

# Install production dependencies
!pip install pyngrok gunicorn

# Mount Google Drive for persistence
from google.colab import drive
drive.mount('/content/drive')

# Link data directory to Drive
!ln -sf /content/drive/MyDrive/ApexAI/data /content/apexai/data

# Start services with process management
import subprocess
import time

# Start API server
api_process = subprocess.Popen([
    'gunicorn', 
    'src.api.main:app',
    '--bind', '0.0.0.0:8000',
    '--workers', '2',
    '--worker-class', 'uvicorn.workers.UvicornWorker'
], cwd='/content/apexai')

# Start Streamlit dashboard
dashboard_process = subprocess.Popen([
    'streamlit', 'run', 'src/dashboard/app.py',
    '--server.port', '8501',
    '--server.address', '0.0.0.0'
], cwd='/content/apexai')

# Create ngrok tunnels
from pyngrok import ngrok
api_tunnel = ngrok.connect(8000)
dashboard_tunnel = ngrok.connect(8501)

print(f"API: {api_tunnel.public_url}")
print(f"Dashboard: {dashboard_tunnel.public_url}")
```

#### **Streamlit Cloud Deployment**
```toml
# .streamlit/config.toml
[server]
headless = true
port = 8501
enableCORS = false

[theme]
primaryColor = "#FF1E1E"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#262730"
textColor = "#FAFAFA"

# requirements.txt optimized for Streamlit Cloud
streamlit>=1.28.0
pandas>=2.0.0
plotly>=5.17.0
fastf1>=3.1.0
requests>=2.31.0
```

---

## 🧪 **Testing & Quality Assurance**

### **Testing Stack**
```python
import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

# Unit testing
class TestTireModel:
    def test_degradation_prediction(self):
        model = TireDegradationModel()
        prediction = model.predict({
            'tire_age': 15,
            'compound': 'medium',
            'track_temp': 45
        })
        assert 0 < prediction < 60  # Reasonable lap count

    @patch('src.llm.openai_client.chat_completion')
    def test_ai_commentary_generation(self, mock_openai):
        mock_openai.return_value = "Hamilton needs to pit soon"
        
        analyst = PitSynth()
        commentary = analyst.generate_commentary({
            'driver': 'HAM',
            'tire_age': 25
        })
        
        assert 'Hamilton' in commentary
        assert len(commentary) > 50

# Integration testing
def test_api_integration():
    with TestClient(app) as client:
        response = client.post("/autopodium/tire-degradation", json={
            "driver": "VER",
            "tire_compound": "medium",
            "tire_age": 15
        })
        assert response.status_code == 200
        assert "degradation" in response.json()

# Load testing with locust
from locust import HttpUser, task, between

class F1ApiUser(HttpUser):
    wait_time = between(1, 3)
    
    @task
    def predict_tire_degradation(self):
        self.client.post("/autopodium/tire-degradation", json={
            "driver": "HAM",
            "tire_compound": "soft",
            "tire_age": 10
        })
    
    @task
    def ask_ai_question(self):
        self.client.post("/pitsynth/ask", json={
            "question": "What is the optimal pit strategy?"
        })
```

### **Code Quality Tools**
```bash
# Pre-commit hooks configuration
pip install pre-commit black flake8 mypy isort

# .pre-commit-config.yaml
repos:
-   repo: https://github.com/psf/black
    rev: 23.9.1
    hooks:
    -   id: black
        language_version: python3.10

-   repo: https://github.com/pycqa/flake8
    rev: 6.1.0
    hooks:
    -   id: flake8
        args: [--max-line-length=88]

-   repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.6.1
    hooks:
    -   id: mypy
        additional_dependencies: [types-requests]

# Run quality checks
black src/ scripts/ tests/
flake8 src/ scripts/ tests/ --max-line-length=88
mypy src/ --ignore-missing-imports
isort src/ scripts/ tests/
```

---

## 📈 **Performance Benchmarks & Metrics**

### **System Performance Targets**
```python
# Performance SLAs
PERFORMANCE_TARGETS = {
    'api_response_time': {
        'tire_degradation': 100,  # ms
        'strategy_optimization': 500,  # ms
        'ai_commentary': 2000,  # ms
    },
    'throughput': {
        'concurrent_users': 100,
        'requests_per_minute': 1000,
    },
    'accuracy': {
        'tire_model_mae': 2.1,  # laps
        'position_prediction_accuracy': 0.78,  # 78%
        'strategy_expert_agreement': 0.75,  # 75%
    }
}

# Automated benchmarking
import time
import statistics

def benchmark_model_performance():
    """Benchmark ML model inference time"""
    model = load_tire_model()
    test_data = generate_test_features(1000)
    
    times = []
    for features in test_data:
        start = time.time()
        prediction = model.predict(features)
        end = time.time()
        times.append((end - start) * 1000)  # Convert to ms
    
    return {
        'mean_latency': statistics.mean(times),
        'p95_latency': statistics.quantiles(times, n=20)[18],  # 95th percentile
        'p99_latency': statistics.quantiles(times, n=100)[98]  # 99th percentile
    }
```

---

This comprehensive technical documentation provides a complete overview of the ApexAI technology stack, architecture decisions, and engineering practices. It demonstrates the depth of technical knowledge required to build a production-ready F1 race intelligence system with advanced AI capabilities.