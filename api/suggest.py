
import os
import json
import logging
from datetime import datetime
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

class SuggestRequest(BaseModel):
    company_id: str
    current_emissions: dict
    emission_breakdown: dict
    industry: str
    company_size: str
    employee_total: int
    facility_area_sqm: float
    monthly_electricity_kwh: float
    peak_demand_kw: float
    power_factor: float
    renewable_energy_percent: float
    operating_hours_per_day: float
    diesel_liters_monthly: float
    natural_gas_kg_monthly: float
    coal_tons_monthly: float
    organic_waste_kg_monthly: float
    water_consumption_kl_monthly: float
    water_recycling_percent: float
    raw_material_consumption_tons: float
    production_volume_monthly: float
    current_initiatives: list[str] = []
    budget_flexibility: str = "medium"

SUGGESTION_METADATA = {
    "night_shift": {
        "title": "Shift Production to Night Hours",
        "category": "energy",
        "threshold_field": "operating_hours_per_day",
        "threshold_value": 16,
        "reason_template": "Your operating hours ({value}h) are high, suggesting potential for load shifting to cleaner night grid."
    },
    "solar_ppa": {
        "title": "Solar Power Purchase Agreement",
        "category": "energy",
        "threshold_field": "renewable_energy_percent",
        "threshold_value": 15,
        "reason_template": "Your renewable energy share ({value}%) is below the industry standard of 15%."
    },
    "power_factor_correction": {
        "title": "Install Power Factor Correction",
        "category": "energy",
        "threshold_field": "power_factor",
        "threshold_value": 0.95,
        "reason_template": "Your power factor ({value}) is below the optimal 0.95, causing reactive power losses."
    },
    "led_lighting": {
        "title": "Complete LED Lighting Retrofit",
        "category": "energy",
        "threshold_field": "facility_area_sqm",
        "threshold_value": 1000,
        "reason_template": "Your facility size of {value} sqm offers a large area for LED savings."
    },
    "waste_heat_recovery": {
        "title": "Install Waste Heat Recovery System",
        "category": "fuel",
        "threshold_field": "coal_tons_monthly",
        "threshold_value": 5,
        "reason_template": "Your monthly coal consumption of {value} tons indicates high potential for heat recovery."
    },
    "biogas_plant": {
        "title": "Set Up Biogas Plant",
        "category": "waste",
        "threshold_field": "organic_waste_kg_monthly",
        "threshold_value": 500,
        "reason_template": "Processing {value} kg of organic waste on-site can replace fossil fuels."
    },
    "water_recycling": {
        "title": "Install Water Recycling System",
        "category": "water",
        "threshold_field": "water_consumption_kl_monthly",
        "threshold_value": 100,
        "reason_template": "High water consumption ({value} KL) makes a recycling system financially viable."
    },
    "switch_to_png": {
        "title": "Switch Diesel to Piped Natural Gas",
        "category": "fuel",
        "threshold_field": "diesel_liters_monthly",
        "threshold_value": 500,
        "reason_template": "Switching {value}L of diesel to PNG reduces carbon intensity by 37%."
    },
    "renewable_procurement": {
        "title": "Switch to Renewable Energy Tariff",
        "category": "energy",
        "threshold_field": "renewable_energy_percent",
        "threshold_value": 10,
        "reason_template": "Your renewable mix ({value}%) is low; green tariffs offer immediate CO2 reduction."
    },
    "process_optimization": {
        "title": "AI-Driven Process Optimization",
        "category": "process",
        "threshold_field": "emission_intensity",
        "threshold_value": 150,
        "reason_template": "Your emission intensity ({value}) is significantly higher than benchmarks."
    }
}

@router.post("/suggestions")
async def get_suggestions(
    request: Request,
    body: SuggestRequest
):
    try:
        if not hasattr(request.app.state, 'suggester'):
             raise HTTPException(status_code=500, detail="Suggester model not loaded")
             
        model = request.app.state.suggester
        le_industry = request.app.state.industry_encoder
        le_size = request.app.state.size_encoder
        feature_cols = request.app.state.feature_columns
        
        # Build feature vector
        scope1 = float(body.current_emissions.get('scope1', 0))
        scope2 = float(body.current_emissions.get('scope2', 0))
        scope3 = float(body.current_emissions.get('scope3', 0))
        total_co2 = scope1 + scope2 + scope3
        intensity = float(total_co2 / body.employee_total) if body.employee_total > 0 else 0.0
        
        input_dict = body.model_dump()
        input_dict['scope1_kg'] = scope1
        input_dict['scope2_kg'] = scope2
        input_dict['emission_intensity'] = intensity
        
        X_df = pd.DataFrame([input_dict])
        
        # Encode
        try: X_df['industry'] = le_industry.transform(X_df['industry'])
        except: X_df['industry'] = 0 
        try: X_df['company_size'] = le_size.transform(X_df['company_size'])
        except: X_df['company_size'] = 0
            
        X_df = X_df[feature_cols]
        probs = model.predict_proba(X_df)
        
        suggestions = []
        target_keys = [
            'night_shift', 'solar_ppa', 'power_factor_correction',
            'led_lighting', 'waste_heat_recovery', 'biogas_plant',
            'water_recycling', 'switch_to_png', 'renewable_procurement',
            'process_optimization'
        ]
        
        # 1. Deduction - Tracking categories
        seen_categories = set()
        
        for i, key in enumerate(target_keys):
            prob_1 = float(probs[i][0][1]) if isinstance(probs, list) else 0.5
            
            # 5. Cap confidence at 0.98
            confidence = min(0.98, max(0.60, prob_1))
            
            if prob_1 > 0.5 and key not in body.current_initiatives:
                meta = SUGGESTION_METADATA[key]
                category = meta["category"]
                
                # Deduplication logic: prefer diversity unless we have few suggestions
                if category in seen_categories and len(suggestions) >= 3:
                    continue
                
                impact_co2 = 0.0
                savings_inr = 0.0
                investment = 0.0
                
                kwh = float(body.monthly_electricity_kwh)
                
                # Impact Logic (simplified representative India market values)
                if key == 'night_shift':
                    impact_co2, savings_inr, investment = scope2 * 0.12, kwh * 0.3 * 4.5, 25000.0
                elif key == 'solar_ppa':
                    impact_co2, savings_inr, investment = scope2 * 0.25, kwh * 0.25 * 2.0, 500000.0
                elif key == "power_factor_correction":
                    impact_co2, savings_inr, investment = scope2 * 0.08, kwh * 0.08 * 8.5, 150000.0
                elif key == "led_lighting":
                    impact_co2, savings_inr, investment = scope2 * 0.15, kwh * 0.15 * 8.5, 200000.0
                elif key == "waste_heat_recovery":
                    impact_co2, savings_inr, investment = scope1 * 0.08, float(body.coal_tons_monthly) * 6500 * 0.15, 800000.0
                elif key == "biogas_plant":
                    impact_co2, savings_inr, investment = float(body.organic_waste_kg_monthly) * 6.25, float(body.organic_waste_kg_monthly) * 2.5, 300000.0
                elif key == "water_recycling":
                    impact_co2, savings_inr, investment = float(body.water_consumption_kl_monthly) * 0.7, float(body.water_consumption_kl_monthly) * 45, 100000.0
                elif key == "switch_to_png":
                    impact_co2, savings_inr, investment = float(body.diesel_liters_monthly) * 1.07, float(body.diesel_liters_monthly) * 12, 50000.0
                elif key == "renewable_procurement":
                    impact_co2, savings_inr, investment = scope2 * 0.30, kwh * 0.1 * 8.5, 1000000.0
                elif key == "process_optimization":
                    impact_co2, savings_inr, investment = total_co2 * 0.07, total_co2 * 0.07 * 0.12, 200000.0

                # 2. Budget Filtering
                flex = body.budget_flexibility.lower()
                if flex == "low" and investment >= 100000: continue
                if flex == "medium" and investment >= 500000: continue

                # 3. Fix Payback Calculation
                payback = investment / savings_inr if savings_inr > 0 else 999.0
                if payback > 120: payback = 120
                
                # 4. ROI Percent
                roi_percent = (savings_inr * 12 / investment) * 100 if investment > 0 else 0.0

                # 6. Dynamic Explanation
                val = getattr(body, meta["threshold_field"], 0)
                if meta["threshold_field"] == "emission_intensity": val = round(intensity, 2)
                why = meta["reason_template"].format(value=val)

                suggestions.append({
                    "suggestion_key": key,
                    "category": category,
                    "title": meta["title"],
                    "why_recommended": why,
                    "impact_co2_kg_monthly": round(impact_co2, 2),
                    "impact_inr_savings_monthly": round(savings_inr, 2),
                    "investment_required_inr": round(investment, 2),
                    "payback_months": round(payback, 1),
                    "roi_percent": round(roi_percent, 1),
                    "payback_note": "Long term investment" if payback >= 120 else "Standard investment",
                    "confidence": round(confidence, 2),
                    "priority": "HIGH" if impact_co2 > 5000 else "MEDIUM"
                })
                seen_categories.add(category)

        suggestions.sort(key=lambda x: x["impact_co2_kg_monthly"], reverse=True)
        return {
            "suggestions": suggestions[:5],
            "total_potential_co2_reduction_kg": round(sum(s["impact_co2_kg_monthly"] for s in suggestions[:5]), 2),
            "generated_at": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Suggestion Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Suggestion error: {str(e)}")
