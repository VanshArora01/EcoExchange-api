
import os
import json
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier
from api.predict_emissions import router as predict_router
from api.suggest import router as suggest_router
from api.match import router as match_router
from api.benchmark import router as benchmark_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lifespan for model loading
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load all models once
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    save_dir = os.path.join(root_dir, 'models', 'saved')
    data_dir = os.path.join(root_dir, 'data')
    
    logger.info(f"Loading models from {save_dir}")
    
    # 1. Load Industry Stats (Statistical Forecaster)
    app.state.industry_stats = {}
    try:
        stats_path = os.path.join(save_dir, 'industry_stats.json')
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                app.state.industry_stats = json.load(f)
            logger.info(f"Loaded industry stats for {len(app.state.industry_stats)} industries")
        else:
            logger.error(f"Industry stats missing: {stats_path}")
    except Exception as e:
        logger.error(f"Error loading industry stats: {e}")

    # 2. Load Suggester XGBoost
    try:
        suggester_model_path = os.path.join(save_dir, 'suggester_model.pkl')
        if os.path.exists(suggester_model_path):
            app.state.suggester = joblib.load(suggester_model_path)
            app.state.industry_encoder = joblib.load(os.path.join(save_dir, 'industry_encoder.pkl'))
            app.state.size_encoder = joblib.load(os.path.join(save_dir, 'size_encoder.pkl'))
            app.state.feature_columns = joblib.load(os.path.join(save_dir, 'feature_columns.pkl'))
        else:
            logger.error(f"Suggester model missing: {suggester_model_path}")
    except Exception as e:
        logger.error(f"Error loading suggester: {e}")

    # 3. Load Benchmarks & Patterns
    try:
        benchmarks_path = os.path.join(save_dir, 'benchmarks.json')
        patterns_path = os.path.join(save_dir, 'real_patterns.json')
        
        if os.path.exists(benchmarks_path):
            with open(benchmarks_path, 'r') as f:
                app.state.benchmarks = json.load(f)
        
        if os.path.exists(patterns_path):
            with open(patterns_path, 'r') as f:
                app.state.real_patterns = json.load(f)
    except Exception as e:
        logger.error(f"Error loading configs: {e}")

    # 4. Load Waste Listings
    try:
        waste_path = os.path.join(data_dir, 'waste_listings.csv')
        if os.path.exists(waste_path):
            app.state.waste_listings = pd.read_csv(waste_path)
        else:
            logger.error(f"Waste listings file missing: {waste_path}")
            app.state.waste_listings = pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading waste listings: {e}")

    yield
    # Cleanup
    app.state.industry_stats.clear()

app = FastAPI(title="EcoExchange AI API", version="1.0.0", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(predict_router)
app.include_router(suggest_router)
app.include_router(match_router)
app.include_router(benchmark_router)

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models_loaded": {
            "industry_stats": bool(getattr(app.state, 'industry_stats', {})),
            "suggester": hasattr(app.state, 'suggester'),
            "benchmarks": hasattr(app.state, 'benchmarks'),
            "waste_listings": len(app.state.waste_listings) if hasattr(app.state, 'waste_listings') else 0
        },
        "endpoints": [
            "POST /predict/emissions",
            "POST /suggestions",
            "POST /match"
        ],
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
