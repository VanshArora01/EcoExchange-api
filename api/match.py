
import os
import json
import logging
from datetime import datetime
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from haversine import haversine, Unit

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

class MatchRequest(BaseModel):
    listing: dict
    max_distance_km: float = 200
    top_n: int = 10

@router.post("/match")
async def match_waste(
    request: Request,
    body: MatchRequest
):
    try:
        if not hasattr(request.app.state, 'waste_listings'):
            raise HTTPException(status_code=500, detail="Waste listings data not loaded")
            
        df_waste = request.app.state.waste_listings
        mat_type = body.listing.get('material_type')
        current_radius = body.max_distance_km
        radius_expanded = False
        
        def find_matches(radius):
            matches = []
            filtered_df = df_waste[df_waste['material_type'] == mat_type].copy()
            for _, row in filtered_df.iterrows():
                dist = haversine(
                    (body.listing.get('latitude', 0), body.listing.get('longitude', 0)),
                    (row['latitude'], row['longitude']),
                    unit=Unit.KILOMETERS
                )
                if dist <= radius:
                    # Scoring
                    dist_score = max(0, 1 - dist / radius)
                    req_qty = float(body.listing.get('quantity_kg', 0))
                    row_qty = float(row['quantity_kg'])
                    qty_ratio = min(req_qty, row_qty) / max(req_qty, row_qty) if max(req_qty, row_qty) > 0 else 0
                    
                    price_diff = abs(float(body.listing.get('price_per_kg', 0)) - float(row['price_per_kg']))
                    price_score = 1.0 if price_diff < 5 else (0.7 if price_diff < 15 else 0.4)
                    
                    final_score = (dist_score * 0.4 + qty_ratio * 0.3 + price_score * 0.3) * 100
                    
                    # 2. Match quality labels
                    label = "Possible Match"
                    if final_score >= 80: label = "Excellent Match"
                    elif final_score >= 65: label = "Good Match"
                    elif final_score >= 50: label = "Fair Match"
                    
                    # 3. Estimated transport cost
                    # cost = distance_km * 45 * (quantity_kg/1000)
                    transport_cost = dist * 45 * (row_qty / 1000)
                    
                    # 4. CO2 saved if matched
                    # Metal_Scrap: * 1.6, Plastic: * 1.2, Paper: * 0.8, Organic: * 0.5, E_Waste: * 2.1, Others: * 0.9
                    co2_factors = {
                        'Metal_Scrap': 1.6, 'Plastic': 1.2, 'Paper': 0.8,
                        'Organic': 0.5, 'E_Waste': 2.1
                    }
                    co2_saved = row_qty * co2_factors.get(mat_type, 0.9)
                    
                    matches.append({
                        "listing_id": row['listing_id'],
                        "distance_km": round(dist, 1),
                        "match_score": round(final_score, 1),
                        "quality_label": label,
                        "estimated_transport_cost_inr": round(transport_cost, 2),
                        "co2_saved_kg": round(co2_saved, 2),
                        "quantity_kg": row_qty,
                        "price_per_kg": row['price_per_kg'],
                        "latitude": row['latitude'],
                        "longitude": row['longitude'],
                        "available_monthly": row['available_monthly']
                    })
            return matches

        results = find_matches(current_radius)
        
        # 1. Radius Expansion logic
        if len(results) < 3 and current_radius < 500:
            radius_expanded = True
            original_radius = current_radius
            current_radius *= 2
            results = find_matches(current_radius)
        
        # 5. Sort by match_score then distance
        results.sort(key=lambda x: (-x['match_score'], x['distance_km']))
        results = results[:body.top_n]
        
        response = {
            "matches": results,
            "total_matches_found": len(results),
            "search_radius_km": round(current_radius, 1)
        }
        
        if radius_expanded:
            response["radius_expanded"] = True
            response["original_radius_km"] = original_radius
            response["expanded_radius_km"] = current_radius
            
        return response
    except Exception as e:
        logger.error(f"Match Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Match error: {str(e)}")
