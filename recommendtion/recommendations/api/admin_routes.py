# recommendtion/recommendations/api/admin_routes.py
from fastapi import APIRouter, Depends, HTTPException, Query, Body
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime, date
from ..core.recommendation_engine import RecommendationEngine
from src.database import get_db
import json

router = APIRouter(prefix="/admin", tags=["admin"])

class SystemConfig(BaseModel):
    recommendation_weight: float = 1.0
    popularity_weight: float = 0.5
    recency_weight: float = 0.3
    user_preference_weight: float = 0.8
    time_penalty_factor: float = 0.1
    max_recommendations: int = 10
    learning_rate: float = 0.01

class BulkUserData(BaseModel):
    user_ids: List[str]
    action: str  # 'export', 'delete', 'reset_preferences'

@router.get("/system_health/")
async def get_system_health(
    db: Session = Depends(get_db)
):
    """
    Get comprehensive system health and status
    """
    try:
        engine = RecommendationEngine(db)
        health_data = await engine.get_system_health()
        
        return {
            "status": "success",
            "system_health": health_data,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system health: {str(e)}")

@router.post("/config/update/")
async def update_system_config(
    config: SystemConfig,
    db: Session = Depends(get_db)
):
    """
    Update system configuration parameters
    """
    try:
        engine = RecommendationEngine(db)
        await engine.update_system_config(config.dict())
        
        return {
            "status": "success",
            "message": "System configuration updated successfully",
            "new_config": config.dict()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update config: {str(e)}")

@router.get("/config/current/")
async def get_current_config(
    db: Session = Depends(get_db)
):
    """
    Get current system configuration
    """
    try:
        engine = RecommendationEngine(db)
        current_config = await engine.get_system_config()
        
        return {
            "status": "success",
            "config": current_config
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get config: {str(e)}")

@router.post("/data/cleanup/")
async def cleanup_old_data(
    days_old: int = Query(90, description="Delete data older than this many days"),
    data_types: List[str] = Query(["bookings", "recommendations", "analytics"], description="Types of data to clean"),
    db: Session = Depends(get_db)
):
    """
    Clean up old system data
    """
    try:
        engine = RecommendationEngine(db)
        cleanup_result = await engine.cleanup_old_data(days_old=days_old, data_types=data_types)
        
        return {
            "status": "success",
            "cleanup_result": cleanup_result,
            "days_old": days_old,
            "data_types": data_types
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cleanup data: {str(e)}")

@router.post("/users/bulk_action/")
async def bulk_user_action(
    request: BulkUserData,
    db: Session = Depends(get_db)
):
    """
    Perform bulk actions on user data
    """
    try:
        engine = RecommendationEngine(db)
        
        if request.action == "export":
            result = await engine.export_user_data(request.user_ids)
        elif request.action == "delete":
            result = await engine.delete_user_data(request.user_ids)
        elif request.action == "reset_preferences":
            result = await engine.reset_user_preferences(request.user_ids)
        else:
            raise HTTPException(status_code=400, detail=f"Invalid action: {request.action}")
        
        return {
            "status": "success",
            "action": request.action,
            "affected_users": len(request.user_ids),
            "result": result
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to perform bulk action: {str(e)}")

@router.get("/performance/metrics/")
async def get_performance_metrics(
    db: Session = Depends(get_db),
    hours: int = Query(24, description="Number of hours to analyze")
):
    """
    Get detailed performance metrics
    """
    try:
        engine = RecommendationEngine(db)
        metrics = await engine.get_performance_metrics(hours=hours)
        
        return {
            "status": "success",
            "performance_metrics": metrics,
            "analysis_hours": hours
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")

@router.post("/recommendations/retrain/")
async def retrain_recommendation_model(
    db: Session = Depends(get_db),
    force: bool = Query(False, description="Force retrain even if recent training exists")
):
    """
    Trigger recommendation model retraining
    """
    try:
        engine = RecommendationEngine(db)
        retrain_result = await engine.retrain_model(force=force)
        
        return {
            "status": "success",
            "retrain_result": retrain_result,
            "forced": force
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrain model: {str(e)}")

@router.get("/audit/logs/")
async def get_audit_logs(
    db: Session = Depends(get_db),
    limit: int = Query(100, description="Number of logs to return"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    action_type: Optional[str] = Query(None, description="Filter by action type")
):
    """
    Get system audit logs
    """
    try:
        engine = RecommendationEngine(db)
        logs = await engine.get_audit_logs(
            limit=limit,
            user_id=user_id,
            action_type=action_type
        )
        
        return {
            "status": "success",
            "audit_logs": logs,
            "filters": {
                "limit": limit,
                "user_id": user_id,
                "action_type": action_type
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get audit logs: {str(e)}")

@router.get("/rooms/management/")
async def get_room_management_data(
    db: Session = Depends(get_db)
):
    """
    Get comprehensive room management data
    """
    try:
        engine = RecommendationEngine(db)
        room_data = await engine.get_room_management_data()
        
        return {
            "status": "success",
            "room_management_data": room_data
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get room management data: {str(e)}")

@router.post("/alerts/configure/")
async def configure_system_alerts(
    alert_config: Dict[str, Any] = Body(...),
    db: Session = Depends(get_db)
):
    """
    Configure system alerts and notifications
    """
    try:
        engine = RecommendationEngine(db)
        await engine.configure_alerts(alert_config)
        
        return {
            "status": "success",
            "message": "Alert configuration updated successfully",
            "alert_config": alert_config
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to configure alerts: {str(e)}")

@router.get("/backup/create/")
async def create_system_backup(
    db: Session = Depends(get_db),
    include_user_data: bool = Query(True, description="Include user data in backup")
):
    """
    Create system backup
    """
    try:
        engine = RecommendationEngine(db)
        backup_result = await engine.create_system_backup(include_user_data=include_user_data)
        
        return {
            "status": "success",
            "backup_result": backup_result,
            "included_user_data": include_user_data
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create backup: {str(e)}")

@router.post("/maintenance/mode/")
async def toggle_maintenance_mode(
    enable: bool = Query(..., description="Enable or disable maintenance mode"),
    message: str = Query("System maintenance in progress", description="Maintenance message"),
    db: Session = Depends(get_db)
):
    """
    Toggle system maintenance mode
    """
    try:
        engine = RecommendationEngine(db)
        await engine.set_maintenance_mode(enable=enable, message=message)
        
        return {
            "status": "success",
            "maintenance_mode": enable,
            "message": message
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to toggle maintenance mode: {str(e)}")