from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime

from recommendtion.config.recommendation_config import RecommendationConfig
from recommendtion.recommendations.core.recommendation_engine import RecommendationEngine

app = FastAPI(title="Room Recommendation API")

# Initialize recommendation engine
config = RecommendationConfig()
engine = RecommendationEngine(config=config)


class RecommendationRequest(BaseModel):
    user_id: str
    room_id: Optional[str] = None  # Optional preferred room
    start_time: datetime
    end_time: datetime
    purpose: str
    capacity: int
    requirements: Optional[Dict[str, Any]] = None


@app.post("/recommend")
def get_recommendation(request: RecommendationRequest):
    try:
        # Format request for engine
        request_data = request.dict()
        recommendations = engine.get_recommendations(request_data)
        
        return {
            "status": "success",
            "count": len(recommendations),
            "recommendations": recommendations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
