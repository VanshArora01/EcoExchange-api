
import os
import json
import logging
from datetime import datetime
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, field_validator
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

class EmissionsRequest(BaseModel):
    company_id: str
    historical_emissions: list[dict]
    production_forecast: list[dict]
    weather_forecast: list[dict]
    industry: str
    region: str
    confidence_threshold: float = 0.85

    @field_validator('historical_emissions')
    @classmethod
    def validate_historical(cls, v):
        if not v:
            raise ValueError("historical_emissions cannot be empty")
        for h in v:
            if h.get('scope1', 0) < 0 or h.get('scope2', 0) < 0 or h.get('scope3', 0) < 0:
                raise ValueError("Scope values must be positive")
            try:
                datetime.strptime(h['month'], '%Y-%m')
            except:
                raise ValueError(f"Invalid month format: {h['month']}. Use YYYY-MM")
        return v
    
    @field_validator('industry')
    @classmethod
    def validate_industry(cls, v):
        valid_industries = [
            'Steel', 'Energy', 'Chemical', 'Automotive', 'Manufacturing',
            'Textile', 'Food_Processing', 'Pharmaceutical', 'Electronics', 'Construction'
        ]
        if v not in valid_industries:
            raise ValueError(f"Invalid industry. Must be one of: {valid_industries}")
        return v

@router.post("/predict/emissions")
async def predict_emissions(
    request: Request, 
    body: EmissionsRequest
):
    try:
        # 1. Load industry stats
        industry_stats = getattr(request.app.state, "industry_stats", {})
        if not industry_stats:
            # Fallback load if not in state
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            stats_path = os.path.join(root_dir, 'models', 'saved', 'industry_stats.json')
            if os.path.exists(stats_path):
                with open(stats_path, 'r') as f:
                    industry_stats = json.load(f)
            else:
                raise HTTPException(status_code=500, detail="Industry stats not found")

        # 2. Get stats for requested industry
        industry = body.industry
        stats = industry_stats.get(industry, multi_industry_fallback(industry_stats))
        
        # Build historical df
        hist_data = []
        hist_totals = []
        hist_volumes = []
        for h in body.historical_emissions:
            total_val = float(h.get('scope1', 0) + h.get('scope2', 0) + h.get('scope3', 0))
            hist_totals.append(total_val)
            hist_volumes.append(float(h.get('production_volume', 1000.0)))
            hist_data.append({
                'month_dt': datetime.strptime(h['month'], '%Y-%m'),
                'total': total_val,
                'scope1': float(h.get('scope1', 0)),
                'scope2': float(h.get('scope2', 0)),
                'scope3': float(h.get('scope3', 0))
            })
        
        # Sort history by date
        hist_data.sort(key=lambda x: x['month_dt'])
        hist_totals_sorted = [x['total'] for x in hist_data]
        n_hist = len(hist_data)
        
        # 3. Baseline Logic
        if n_hist >= 3:
            recent_avg = float(np.mean(hist_totals_sorted[-3:]))
            # recent_trend = (last_month - first_of_last_3) / first_of_last_3 / 2
            v_last = hist_totals_sorted[-1]
            v_first_of_3 = hist_totals_sorted[-3]
            recent_trend = (v_last - v_first_of_3) / v_first_of_3 / 2 if v_first_of_3 > 0 else 0.0
        else:
            recent_avg = float(stats['avg_monthly_co2'])
            recent_trend = float(stats['growth_rate'])

        # 4. Forecasting Logic for 3 months
        avg_hist_volume = float(np.mean(hist_volumes)) if hist_volumes else 1000.0
        last_month_num = hist_data[-1]['month_dt'].month
        
        predictions = []
        prod_forecast = {p['month']: p['volume'] for p in body.production_forecast}
        temp_forecast = {w['month']: w['avg_temp'] for w in body.weather_forecast}
        
        # Determine the next 3 months
        last_dt = hist_data[-1]['month_dt']
        forecast_months = []
        for i in range(1, 4):
            m = last_dt.month + i
            y = last_dt.year + (m - 1) // 12
            m = (m - 1) % 12 + 1
            forecast_months.append(datetime(y, m, 1))

        for dt in forecast_months:
            m_str = dt.strftime('%Y-%m')
            month_num = dt.month
            
            # Seasonal factor
            seasonal_factor = float(stats['seasonal_multipliers'].get(str(month_num), 1.0))
            
            # Production factor
            fv = prod_forecast.get(m_str)
            if fv is not None:
                production_factor = float(fv) / avg_hist_volume if avg_hist_volume > 0 else 1.0
                production_factor = np.clip(production_factor, 0.8, 1.3)
            else:
                production_factor = 1.0
                
            # Weather factor
            temp = temp_forecast.get(m_str)
            weather_factor = 1.0
            if temp is not None:
                temp = float(temp)
                if temp > 35: weather_factor = 1.08
                elif temp > 30: weather_factor = 1.03
                elif temp < 20: weather_factor = 1.05
            
            # Prediction
            predicted = (recent_avg * (1 + recent_trend) * seasonal_factor * production_factor * weather_factor)
            
            # ANTI-HALLUCINATION HARD CAP (±30%)
            max_allowed = recent_avg * 1.30
            min_allowed = recent_avg * 0.70
            predicted = float(np.clip(predicted, min_allowed, max_allowed))
            
            # Bounds
            lower_bound = predicted * 0.88
            upper_bound = predicted * 1.12
            
            # Key drivers logic
            drivers = []
            if weather_factor > 1.05: drivers.append("temperature_increase")
            if recent_trend > 0.03: drivers.append("production_ramp_up")
            if seasonal_factor > 1.10: drivers.append("peak_season")
            if recent_trend < -0.03: drivers.append("efficiency_improving")

            predictions.append({
                "month": m_str,
                "total_predicted": round(predicted, 2),
                "confidence_interval": [round(lower_bound, 2), round(upper_bound, 2)],
                "confidence_level": 0.88,
                "key_drivers": drivers
            })

        # 5. Split predicted total into scopes
        has_scope_history = all('scope1' in h and 'scope2' in h for h in body.historical_emissions)
        if has_scope_history:
            s1_ratios = [h['scope1'] / (h['scope1'] + h['scope2'] + h['scope3']) for h in body.historical_emissions if (h['scope1'] + h['scope2'] + h['scope3']) > 0]
            s2_ratios = [h['scope2'] / (h['scope1'] + h['scope2'] + h['scope3']) for h in body.historical_emissions if (h['scope1'] + h['scope2'] + h['scope3']) > 0]
            s1_ratio = float(np.mean(s1_ratios)) if s1_ratios else stats['scope1_ratio']
            s2_ratio = float(np.mean(s2_ratios)) if s2_ratios else stats['scope2_ratio']
        else:
            s1_ratio = stats['scope1_ratio']
            s2_ratio = stats['scope2_ratio']
            
        s3_ratio = 1.0 - s1_ratio - s2_ratio
        
        for p in predictions:
            total = p['total_predicted']
            p['scope1_predicted'] = round(total * s1_ratio, 2)
            p['scope2_predicted'] = round(total * s2_ratio, 2)
            p['scope3_predicted'] = round(total * s3_ratio, 2)

        # 6. Anomaly detection using z-score
        anomalies = []
        hist_avg_all = np.mean(hist_totals_sorted)
        hist_std_all = np.std(hist_totals_sorted)
        if n_hist > 1 and hist_std_all > 0:
            for x in hist_data:
                z = (x['total'] - hist_avg_all) / hist_std_all
                if abs(z) > 1.0:
                    severity = "info"
                    if abs(z) > 1.5: severity = "warning"
                    if abs(z) > 2.0: severity = "critical"
                    anomalies.append({
                        "month": x['month_dt'].strftime('%Y-%m'),
                        "z_score": round(z, 2),
                        "type": "spike" if z > 0 else "dip",
                        "severity": severity
                    })

        # 7. Trend calculation
        first_total = hist_data[0]['total']
        last_total = hist_data[-1]['total']
        months_diff = (hist_data[-1]['month_dt'].year - hist_data[0]['month_dt'].year) * 12 + (hist_data[-1]['month_dt'].month - hist_data[0]['month_dt'].month)
        if months_diff == 0: months_diff = 1
        monthly_change = (last_total - first_total) / first_total / months_diff if first_total > 0 else 0.0
        
        trend_status = "stable"
        if monthly_change > 0.03: trend_status = "increasing"
        elif monthly_change < -0.03: trend_status = "decreasing"

        # 8. Benchmark comparison
        your_avg = float(np.mean(hist_totals_sorted))
        industry_avg = float(stats['avg_monthly_co2'])
        percentile = int((1 - your_avg/industry_avg) * 100) if industry_avg > 0 else 50
        percentile = np.clip(percentile, 1, 99)
        
        benchmark_comparison = {
            "your_avg_monthly_co2": round(your_avg, 2),
            "industry_avg_monthly_co2": round(industry_avg, 2),
            "percentile": int(percentile),
            "status": "below_average" if your_avg < industry_avg else "above_average"
        }

        # Data quality score
        if n_hist >= 12: data_quality_score = 0.95
        elif n_hist >= 6: data_quality_score = 0.80
        elif n_hist >= 3: data_quality_score = 0.60
        else: data_quality_score = 0.40

        return {
            "predictions": predictions,
            "anomalies": anomalies,
            "trend": trend_status,
            "trend_percent": round(monthly_change * 100, 1),
            "benchmark_comparison": benchmark_comparison,
            "model_version": "statistical-v3.0",
            "data_quality_score": data_quality_score
        }

    except Exception as e:
        logger.error(f"Prediction Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

def multi_industry_fallback(stats):
    # Just take Manufacturing or first one
    return stats.get("Manufacturing", list(stats.values())[0])

def multi_industry_fallback(stats):
    if not stats: return {}
    if "Manufacturing" in stats: return stats["Manufacturing"]
    return next(iter(stats.values()))
