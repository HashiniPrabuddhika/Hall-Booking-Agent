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
from pydantic import Field
from typing import Optional, List, Any
from src.api import router 
from src.deepseek_llm import DeepSeekLLM
from fastapi.middleware.cors import CORSMiddleware
from recommendtion.recommendations.api.recommendation_routes import recommendation_router

app = FastAPI()

# 👇 Allow frontend on localhost:3000
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
app.include_router(recommendation_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,            # 🌐 allow React frontend
    allow_credentials=True,
    allow_methods=["*"],              # 🟢 Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],              # 🟢 Allow all headers
)

app.include_router(router)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from bs4 import BeautifulSoup
import requests

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

