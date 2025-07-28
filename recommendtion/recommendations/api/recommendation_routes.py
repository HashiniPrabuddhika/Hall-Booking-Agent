# recommendtion/recommendations/api/recommendation_routes.py
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from ..core.recommendation_engine import RecommendationEngine
from src.database import get_db
import json

# Create the router instance - this is what the test is looking for
router = APIRouter(prefix="/recommendations", tags=["recommendations"])

class RecommendationRequest(BaseModel):
    user_id: str
    request_type: str  # 'alternative_time', 'alternative_room', 'proactive', 'smart_scheduling', 'comprehensive'
    context: Dict[str, Any]

class FeedbackRequest(BaseModel):
    user_id: str
    recommendation_id: str
    action: str  # 'accepted', 'rejected', 'ignored'
    booking_data: Optional[Dict[str, Any]] = None

@router.post("/get_recommendations/")
async def get_recommendations(
    request: RecommendationRequest,
    db: Session = Depends(get_db)
):
    """
    Get recommendations based on request type and context
    """
    try:
        engine = RecommendationEngine(db)
        recommendations = await engine.get_recommendations(
            user_id=request.user_id,
            request_type=request.request_type,
            context=request.context
        )
        
        return {
            "status": "success",
            "recommendations": recommendations,
            "user_id": request.user_id,
            "request_type": request.request_type
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate recommendations: {str(e)}")

@router.post("/feedback/")
async def provide_feedback(
    request: FeedbackRequest,
    db: Session = Depends(get_db)
):
    """
    Provide feedback on recommendations to improve future suggestions
    """
    try:
        engine = RecommendationEngine(db)
        
        # Learn from user feedback
        outcome = "success" if request.action == "accepted" else "rejected"
        await engine.learn_from_booking(
            user_id=request.user_id,
            booking_data=request.booking_data or {},
            outcome=outcome
        )
        
        return {
            "status": "success",
            "message": "Feedback recorded successfully"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record feedback: {str(e)}")

@router.get("/user_patterns/{user_id}")
async def get_user_patterns(
    user_id: str,
    db: Session = Depends(get_db)
):
    """
    Get user booking patterns and preferences
    """
    try:
        engine = RecommendationEngine(db)
        patterns = await engine.analytics.get_user_booking_patterns(user_id)
        
        return {
            "status": "success",
            "user_id": user_id,
            "patterns": patterns
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get user patterns: {str(e)}")

@router.get("/room_analytics/{room_name}")
async def get_room_analytics(
    room_name: str,
    db: Session = Depends(get_db)
):
    """
    Get analytics for a specific room
    """
    try:
        engine = RecommendationEngine(db)
        analytics = await engine.analytics.get_room_analytics(room_name)
        
        return {
            "status": "success",
            "room_name": room_name,
            "analytics": analytics
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get room analytics: {str(e)}")

@router.get("/optimal_times/")
async def get_optimal_booking_times(
    db: Session = Depends(get_db),
    limit: int = Query(10, description="Number of optimal times to return")
):
    """
    Get optimal booking times across all rooms
    """
    try:
        engine = RecommendationEngine(db)
        optimal_times = await engine.analytics.get_optimal_booking_times(limit=limit)
        
        return {
            "status": "success",
            "optimal_times": optimal_times
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get optimal times: {str(e)}")

# Keep the old variable name for backward compatibility
recommendation_router = router