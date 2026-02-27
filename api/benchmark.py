
import os
import json
from fastapi import APIRouter, Request, HTTPException

router = APIRouter()

@router.get("/benchmark/{industry}")
async def get_industry_benchmark(request: Request, industry: str):
    """
    Returns industry benchmark data from benchmarks.json
    """
    try:
        benchmarks = getattr(request.app.state, 'benchmarks', {})
        
        industry_key = industry.strip()
        if industry_key not in benchmarks:
            # Try to find case-insensitive or partial match
            found = False
            for k in benchmarks.keys():
                if k.lower() == industry_key.lower():
                    industry_key = k
                    found = True
                    break
            
            if not found:
                raise HTTPException(status_code=404, detail=f"Industry '{industry}' not found in benchmarks")
        
        return benchmarks[industry_key]
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving benchmark: {str(e)}")
