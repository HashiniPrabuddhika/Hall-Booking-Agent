# recommendtion/recommendations/api/analytics_routes.py
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime, date
from ..core.recommendation_engine import RecommendationEngine
from src.database import get_db
import json

router = APIRouter(prefix="/analytics", tags=["analytics"])

class DateRange(BaseModel):
    start_date: date
    end_date: date

@router.get("/system_stats/")
async def get_system_analytics(
    db: Session = Depends(get_db),
    days: int = Query(30, description="Number of days to analyze")
):
    """
    Get overall system analytics and performance metrics
    """
    try:
        engine = RecommendationEngine(db)
        analytics = await engine.analytics.get_system_analytics(days=days)
        
        return {
            "status": "success",
            "analytics": analytics,
            "period_days": days
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system analytics: {str(e)}")

@router.get("/room_utilization/")
async def get_room_utilization(
    db: Session = Depends(get_db),
    days: int = Query(30, description="Number of days to analyze")
):
    """
    Get room utilization rates and patterns
    """
    try:
        engine = RecommendationEngine(db)
        utilization = await engine.analytics.get_room_utilization(days=days)
        
        return {
            "status": "success",
            "utilization": utilization,
            "period_days": days
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get room utilization: {str(e)}")

@router.get("/peak_hours/")
async def get_peak_hours(
    db: Session = Depends(get_db),
    room_name: Optional[str] = Query(None, description="Specific room to analyze")
):
    """
    Get peak booking hours analysis
    """
    try:
        engine = RecommendationEngine(db)
        peak_hours = await engine.analytics.get_peak_hours(room_name=room_name)
        
        return {
            "status": "success",
            "peak_hours": peak_hours,
            "room_name": room_name or "all_rooms"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get peak hours: {str(e)}")

@router.get("/booking_trends/")
async def get_booking_trends(
    db: Session = Depends(get_db),
    period: str = Query("weekly", description="Trend period: daily, weekly, monthly")
):
    """
    Get booking trends over time
    """
    try:
        engine = RecommendationEngine(db)
        trends = await engine.analytics.get_booking_trends(period=period)
        
        return {
            "status": "success",
            "trends": trends,
            "period": period
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get booking trends: {str(e)}")

@router.get("/user_activity/{user_id}")
async def get_user_activity(
    user_id: str,
    db: Session = Depends(get_db),
    days: int = Query(90, description="Number of days to analyze")
):
    """
    Get detailed user activity and booking patterns
    """
    try:
        engine = RecommendationEngine(db)
        activity = await engine.analytics.get_user_activity(user_id=user_id, days=days)
        
        return {
            "status": "success",
            "user_id": user_id,
            "activity": activity,
            "period_days": days
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get user activity: {str(e)}")

@router.get("/recommendation_effectiveness/")
async def get_recommendation_effectiveness(
    db: Session = Depends(get_db),
    days: int = Query(30, description="Number of days to analyze")
):
    """
    Get recommendation system effectiveness metrics
    """
    try:
        engine = RecommendationEngine(db)
        effectiveness = await engine.analytics.get_recommendation_effectiveness(days=days)
        
        return {
            "status": "success",
            "effectiveness": effectiveness,
            "period_days": days
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get recommendation effectiveness: {str(e)}")

@router.get("/popular_times/")
async def get_popular_booking_times(
    db: Session = Depends(get_db),
    room_name: Optional[str] = Query(None, description="Specific room to analyze"),
    limit: int = Query(20, description="Number of popular times to return")
):
    """
    Get most popular booking times and durations
    """
    try:
        engine = RecommendationEngine(db)
        popular_times = await engine.analytics.get_popular_booking_times(
            room_name=room_name, 
            limit=limit
        )
        
        return {
            "status": "success",
            "popular_times": popular_times,
            "room_name": room_name or "all_rooms",
            "limit": limit
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get popular times: {str(e)}")

@router.get("/cancellation_patterns/")
async def get_cancellation_patterns(
    db: Session = Depends(get_db),
    days: int = Query(60, description="Number of days to analyze")
):
    """
    Get booking cancellation patterns and reasons
    """
    try:
        engine = RecommendationEngine(db)
        patterns = await engine.analytics.get_cancellation_patterns(days=days)
        
        return {
            "status": "success",
            "cancellation_patterns": patterns,
            "period_days": days
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cancellation patterns: {str(e)}")

@router.get("/room_conflicts/")
async def get_room_conflicts(
    db: Session = Depends(get_db),
    days: int = Query(7, description="Number of days to analyze")
):
    """
    Get room booking conflicts and overlaps
    """
    try:
        engine = RecommendationEngine(db)
        conflicts = await engine.analytics.get_room_conflicts(days=days)
        
        return {
            "status": "success",
            "conflicts": conflicts,
            "period_days": days
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get room conflicts: {str(e)}")

@router.get("/efficiency_metrics/")
async def get_efficiency_metrics(
    db: Session = Depends(get_db)
):
    """
    Get overall system efficiency and performance metrics
    """
    try:
        engine = RecommendationEngine(db)
        metrics = await engine.analytics.get_efficiency_metrics()
        
        return {
            "status": "success",
            "efficiency_metrics": metrics
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get efficiency metrics: {str(e)}")