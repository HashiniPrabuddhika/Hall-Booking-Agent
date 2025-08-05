from sqlalchemy.orm import Session
from datetime import datetime
import time
from fastapi import HTTPException
from . import models
from datetime import datetime, timedelta
from recommendtion.config.recommendation_config import RecommendationConfig
from recommendtion.recommendations.core.recommendation_engine import RecommendationEngine
from typing import Dict, Any


config = RecommendationConfig()
recommendation_engine = RecommendationEngine(config=config)

def get_room_recommendations(room_name: str, date: str, start_time: str, end_time: str, db: Session):
    try:
        start_dt = datetime.strptime(f"{date} {start_time}", "%Y-%m-%d %H:%M")
        end_dt = datetime.strptime(f"{date} {end_time}", "%Y-%m-%d %H:%M")
        
        request_data = {
            "user_id": "system",  
            "room_id": room_name,  
            "start_time": start_dt,
            "end_time": end_dt,
            "purpose": "meeting",  
            "capacity": 1, 
            "requirements": {"original_room": room_name}
        }
        
        recommendations = recommendation_engine.get_recommendations(request_data)
        return recommendations
    except Exception as e:
        print(f"Recommendation system error: {e}")
        return []

def check_availability(room_name: str, date: str, start_time: str, end_time: str, db: Session):
    print(f"Checking availability for room: {room_name}")
    print(f"Date: {date}, Start time: {start_time}, End time: {end_time}")

    room = db.query(models.MRBSRoom).filter(models.MRBSRoom.room_name == room_name).first()
    print(f"Queried room from DB: {room}")

    # Convert to datetime objects first
    start_dt = datetime.strptime(f"{date} {start_time}", "%Y-%m-%d %H:%M")
    end_dt = datetime.strptime(f"{date} {end_time}", "%Y-%m-%d %H:%M")

    # Convert datetime to Unix timestamps (int)
    start_ts = int(time.mktime(start_dt.timetuple()))
    end_ts = int(time.mktime(end_dt.timetuple()))
    print(f"Converted start datetime to Unix timestamp: {start_ts}")
    print(f"Converted end datetime to Unix timestamp: {end_ts}")

    # Query for conflicting bookings using Unix timestamps
    conflicting = db.query(models.MRBSEntry).filter(
        models.MRBSEntry.room_id == room.id,
        models.MRBSEntry.start_time < end_ts,
        models.MRBSEntry.end_time > start_ts,
    ).first()
    print(f"Conflicting booking found: {conflicting}")

    if conflicting:
        message = f"{room_name} is already booked for that time. Here are some available alternatives you might like:"
        print(message)
        recommendations = get_room_recommendations(room_name, date, start_time, end_time, db)
        return {"status": "unavailable",
            "message": message,
            "recommendations": recommendations
            }
    
    if not room:
        print("Room not found!")
        recommendations = get_room_recommendations(room_name, date, start_time, end_time, db)
    
        return {
                "status": "room_not_found",
                "message": f"Room '{room_name}' not found.",
                "recommendations": recommendations
        }

    message = f"{room_name} is available from {start_time} to {end_time} on {date}."
    print(message)
    return {"status": "available", "message": message}

def add_booking(room_name: str, date: str, start_time: str, end_time: str, created_by: str, db: Session):
    
    try:
        room = db.query(models.MRBSRoom).filter(models.MRBSRoom.room_name == room_name).first()
        
        if not room:
            recommendations = get_room_recommendations(room_name, date, start_time, end_time, db)
            raise HTTPException(
                status_code=404, 
                detail={
                    "error": "Room not found",
                    "message": f"Room '{room_name}' not found.",
                    "recommendations": recommendations
                }
            )
        
        try:
            start_dt = datetime.strptime(f"{date} {start_time}", "%Y-%m-%d %H:%M")
            end_dt = datetime.strptime(f"{date} {end_time}", "%Y-%m-%d %H:%M")

            
            start_ts = int(time.mktime(start_dt.timetuple()))
            end_ts = int(time.mktime(end_dt.timetuple()))

            
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid date/time format: {e}")
        
        if end_ts <= start_ts:
            raise HTTPException(status_code=400, detail="End time must be after start time")
        
        
        conflict = db.query(models.MRBSEntry).filter(
            models.MRBSEntry.room_id == room.id,
            models.MRBSEntry.start_time < end_ts,
            models.MRBSEntry.end_time > start_ts,
        ).first()
        
        if conflict:
            recommendations = get_room_recommendations(room_name, date, start_time, end_time, db)
            return {
                "status": "unavailable",
                "message": f"Room '{room_name}' is already booked for that time. Here are some available alternatives you might like:",
                "recommendations": recommendations
            }
        
        current_datetime = datetime.now()
        
        try:
            new_booking = models.MRBSEntry(
                start_time=start_ts,
                end_time=end_ts,
                entry_type=0,
                repeat_id=None,
                room_id=room.id,
                timestamp=current_datetime, 
                create_by=created_by,
                modified_by=created_by,
                name=f"Booking for {room_name}",
                type='E',
                description=f"Booked by {created_by}",
                status=0,
                reminded=None,
                info_time=None,
                info_user=None,
                info_text=None,
                ical_uid=f"{room_name}_{start_ts}_{end_ts}",
                ical_sequence=0,
                ical_recur_id=None
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error creating booking object: {e}")
        
        try:
            db.add(new_booking)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error adding booking to session: {e}")
        
        try:
            db.commit()
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Database commit failed: {e}")
        
        try:
            db.refresh(new_booking)
        except Exception as e:
            pass
        
        return {
            "message": "Booking created successfully",
            "booking_id": new_booking.id,
            "room": room_name,
            "date": date,
            "start_time": start_time,
            "end_time": end_time
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")
    
    
def book_recommendation_directly(recommendation: Dict[str, Any], created_by: str, db: Session):
   
    try:
        # Extract suggestion data from recommendation
        suggestion = recommendation.get('suggestion', {})
        if not suggestion:
            raise HTTPException(status_code=400, detail="Invalid recommendation: missing suggestion data")
        
        room_name = suggestion.get('room_name')
        start_time_str = suggestion.get('start_time')
        end_time_str = suggestion.get('end_time')
        
        if not all([room_name, start_time_str, end_time_str]):
            raise HTTPException(
                status_code=400, 
                detail="Invalid recommendation: missing room_name, start_time, or end_time"
            )
        
        # Parse the datetime strings
        try:
            start_dt = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(end_time_str.replace('Z', '+00:00'))
        except ValueError:
            # Try alternative parsing if ISO format fails
            try:
                start_dt = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
                end_dt = datetime.strptime(end_time_str, "%Y-%m-%d %H:%M:%S")
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid datetime format in recommendation: {e}")
        
        date = start_dt.strftime("%Y-%m-%d")
        start_time = start_dt.strftime("%H:%M") 
        end_time = end_dt.strftime("%H:%M")
        
        print(f"Booking recommendation: {room_name} on {date} from {start_time} to {end_time}")
        
        result = add_booking(
            room_name=room_name,
            date=date,
            start_time=start_time,
            end_time=end_time,
            created_by=created_by,
            db=db,
            validate_availability=False  
        )
        
        result['recommendation_type'] = recommendation.get('type', 'unknown')
        result['recommendation_score'] = recommendation.get('score', 0)
        result['recommendation_reason'] = recommendation.get('reason', '')
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error booking recommendation: {e}")
 
 
def check_available_slotes(room_name: str, date: str, start_time: str, end_time: str, db: Session):
    # alternative slotes
    print(f"Checking availability for room: {room_name}")
    print(f"Date: {date}, Start time: {start_time}, End time: {end_time}")

    room = db.query(models.MRBSRoom).filter(models.MRBSRoom.room_name == room_name).first()
    print(f"Queried room from DB: {room}")

    # Convert to datetime objects first
    date_obj = datetime.strptime(date, "%Y-%m-%d")
    start_time = datetime.combine(date_obj, datetime.min.time()) + timedelta(hours=7)  # 7 AM
    end_time = datetime.combine(date_obj, datetime.min.time()) + timedelta(hours=21)  # 9 PM

    all_slots = []
    current = start_time
    while current < end_time:
        slot_start = current
        slot_end = current + timedelta(minutes=30)
        all_slots.append((int(time.mktime(slot_start.timetuple())), int(time.mktime(slot_end.timetuple()))))
        current = slot_end

    # Step 3: Get all bookings for that day and room
    day_start_ts = int(time.mktime(start_time.timetuple()))
    day_end_ts = int(time.mktime(end_time.timetuple()))

    bookings = db.query(models.MRBSEntry).filter(
        models.MRBSEntry.room_id == room.id,
        models.MRBSEntry.start_time < day_end_ts,
        models.MRBSEntry.end_time > day_start_ts
    ).all()

    # Step 4: Filter available slots
    available_slots = []
    for slot_start, slot_end in all_slots:
        conflict = any(
            booking.start_time < slot_end and booking.end_time > slot_start
            for booking in bookings
        )
        if not conflict:
            available_slots.append({
                "start_time": datetime.fromtimestamp(slot_start).strftime("%H:%M"),
                "end_time": datetime.fromtimestamp(slot_end).strftime("%H:%M")
            })
            
    if not available_slots:
        recommendations = get_room_recommendations(room_name, date, start_time, end_time, db)
        return {
            "status": "no_slots_available",
            "message": f"No available time slots found for {room_name} on {date}. Here are some available alternatives you might like:",
            "room": room_name,
            "date": date,
            "available_slots": [],
            "recommendations": recommendations
        } 
            
    if not room:
        print("Room not found!")
        recommendations = get_room_recommendations(room_name, date, start_time, end_time, db)
        raise HTTPException(
            status_code=404, 
            detail={
                "error": "Room not found",
                "message": f"Room '{room_name}' not found.",
                "recommendations": recommendations
            }
        )
    
    return {"room": room_name, "date": date, "available_slots": available_slots}