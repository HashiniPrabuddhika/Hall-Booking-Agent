from datetime import datetime, date, time
from fastapi import FastAPI, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from src.models import MRBSEntry, MRBSRoom
from src.database import get_db
import logging
import os
from langchain_core.language_models import BaseLLM
from langchain_core.outputs import LLMResult, Generation
import requests
from pydantic import BaseModel,Field
from typing import Optional, List, Any, Dict
from src.api import router 
from src.deepseek_llm import DeepSeekLLM
from fastapi.middleware.cors import CORSMiddleware
from recommendtion.recommendations.api.recommendation_routes import recommendation_router
from recommendtion.config.recommendation_config import RecommendationConfig
from recommendtion.recommendations.core.recommendation_engine import RecommendationEngine

app = FastAPI()

# config = RecommendationConfig()
# engine = RecommendationEngine(config=config)

# 👇 Allow frontend on localhost:3000
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,            # 🌐 allow React frontend
    allow_credentials=True,
    allow_methods=["*"],              # 🟢 Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],              # 🟢 Allow all headers
)

app.include_router(router)
app.include_router(recommendation_router)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from bs4 import BeautifulSoup
import requests

# class RecommendationRequest(BaseModel):
#     user_id: str
#     room_id: Optional[str] = None  
#     start_time: datetime
#     end_time: datetime
#     purpose: str
#     capacity: int
#     requirements: Optional[Dict[str, Any]] = None
# @app.post("/recommend")
# def get_recommendation(request: RecommendationRequest):
#     """
#     Get room recommendations based on user requirements
#     """
#     try:
#         logger.info(f"🔵 Recommendation request for user: {request.user_id}")
        
#         request_data = request.dict()
#         recommendations = engine.get_recommendations(request_data)
        
#         logger.info(f"✅ Generated {len(recommendations)} recommendations")
        
#         return {
#             "status": "success",
#             "count": len(recommendations),
#             "recommendations": recommendations
#         }
#     except Exception as e:
#         logger.error(f"🔴 Recommendation error: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# # Enhanced recommendation endpoint with availability check
# @app.post("/recommend_with_availability")
# def get_recommendation_with_availability(
#     request: RecommendationRequest, 
#     db: Session = Depends(get_db)
# ):
#     """
#     Get room recommendations and check their availability
#     """
#     try:
#         logger.info(f"🔵 Recommendation with availability check for user: {request.user_id}")
        
#         # Get recommendations
#         request_data = request.dict()
#         recommendations = engine.get_recommendations(request_data)
        
#         # Check availability for each recommended room
#         available_recommendations = []
        
#         for rec in recommendations:
#             room_name = rec.get('room_name')  # Adjust based on your recommendation structure
            
#             if room_name:
#                 # Check if room exists
#                 room = db.query(MRBSRoom).filter(MRBSRoom.room_name == room_name).first()
#                 if room:
#                     # Convert datetime to timestamps
#                     start_timestamp = int(request.start_time.timestamp())
#                     end_timestamp = int(request.end_time.timestamp())
                    
#                     # Check for overlapping bookings
#                     existing_booking = (
#                         db.query(MRBSEntry)
#                         .filter(
#                             MRBSEntry.room_id == room.id,
#                             MRBSEntry.start_time < end_timestamp,
#                             MRBSEntry.end_time > start_timestamp
#                         )
#                         .first()
#                     )
                    
#                     # Add availability status to recommendation
#                     rec['available'] = existing_booking is None
#                     rec['room_id'] = room.id
#                 else:
#                     rec['available'] = False
#                     rec['room_id'] = None
#             else:
#                 rec['available'] = False
#                 rec['room_id'] = None
            
#             available_recommendations.append(rec)
        
#         # Sort by availability (available rooms first)
#         available_recommendations.sort(key=lambda x: x.get('available', False), reverse=True)
        
#         logger.info(f"✅ Generated {len(available_recommendations)} recommendations with availability")
        
#         return {
#             "status": "success",
#             "count": len(available_recommendations),
#             "recommendations": available_recommendations,
#             "request_details": {
#                 "start_time": request.start_time,
#                 "end_time": request.end_time,
#                 "purpose": request.purpose,
#                 "capacity": request.capacity
#             }
#         }
#     except Exception as e:
#         logger.error(f"🔴 Recommendation with availability error: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# dieerect end apis for testing
# remove when project comes to the end
@app.get("/fetch_bookings")
def fetch_bookings(room_name: str, db: Session = Depends(get_db)):
    print(f"🔵 Received Input -> room_name: {room_name}")

    room = db.query(MRBSRoom).filter(MRBSRoom.room_name == room_name).first()
    if not room:
        print(f"🔴 Room '{room_name}' not found")
        raise HTTPException(status_code=404, detail="Room not found")

    print(f"✅ Room Found -> room_id: {room.id}")

    existing_bookings = (
        db.query(MRBSEntry)
        .filter(MRBSEntry.room_id == room.id)
        .all()
    )

    if existing_bookings:
        print(f"✅ Room '{room_name}' has bookings")
        return existing_bookings
    
    print(f"ℹ️ Room '{room_name}' isn't booked at this time")
    return {"message": f"{room_name} isn't booked at this time"}

@app.get("/check_availability/")
def check_availability(room_name: str, date: date, start_time: str, end_time: str, db: Session = Depends(get_db)):
    print(f"🔵 Received Input -> room_name: {room_name}, date: {date}, start_time: {start_time}, end_time: {end_time}")

    # Strip newline characters and spaces (if any)
    start_time = start_time.strip()
    end_time = end_time.strip()
    print(f"🟢 Cleaned Input -> start_time: {start_time}, end_time: {end_time}")

    # Convert start_time and end_time to `time` objects
    try:
        start_time_obj = datetime.strptime(start_time, "%H:%M").time()
        end_time_obj = datetime.strptime(end_time, "%H:%M").time()
        print(f"🟡 Parsed Time -> start_time_obj: {start_time_obj}, end_time_obj: {end_time_obj}")
    except ValueError as e:
        print(f"🔴 Time Parsing Error: {e}")
        raise HTTPException(status_code=400, detail="Invalid time format. Use HH:MM.")

    # Convert date and time to timestamps
    start_timestamp = int(datetime.combine(date, start_time_obj).timestamp())
    end_timestamp = int(datetime.combine(date, end_time_obj).timestamp())
    print(f"🟣 Converted Timestamps -> start_timestamp: {start_timestamp}, end_timestamp: {end_timestamp}")

    # Check if the room exists
    room = db.query(MRBSRoom).filter(MRBSRoom.room_name == room_name).first()
    if not room:
        print(f"🔴 Room '{room_name}' not found")
        raise HTTPException(status_code=404, detail="Room not found")

    print(f"✅ Room Found -> room_id: {room.id}")

    # Check for overlapping bookings
    existing_booking = (
        db.query(MRBSEntry)
        .filter(
            MRBSEntry.room_id == room.id,
            MRBSEntry.start_time < end_timestamp,
            MRBSEntry.end_time > start_timestamp
        )
        .first()
    )

    if existing_booking:
        print(f"❌ Room '{room_name}' is NOT available at this time")
        return {"message": f"{room_name} is NOT available at this time"}
    
    print(f"✅ Room '{room_name}' is available for booking")
    return {"message": f"{room_name} is available. You can book it."}

