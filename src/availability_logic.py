from sqlalchemy.orm import Session
from datetime import datetime
import time
from fastapi import HTTPException
from . import models
from datetime import datetime, timedelta
from recommendtion.recommendations.core.recommendation_engine import RecommendationEngine
from recommendtion.config.recommendation_config import RecommendationConfig
import logging

logger = logging.getLogger(__name__)


async def check_availability(room_name: str, date: str, start_time: str, end_time: str, db: Session, user_id: str = None):
    print(f"🔵 DEBUG: Checking availability for room: {room_name}")
    print(f"🔵 DEBUG: User ID provided: {user_id}")
    print(f"🔵 DEBUG: Date: {date}, Start time: {start_time}, End time: {end_time}")

    room = db.query(models.MRBSRoom).filter(models.MRBSRoom.room_name == room_name).first()
    print(f"🔵 DEBUG: Queried room from DB: {room}")

    if not room:
        print("🔴 DEBUG: Room not found!")
        print(f"🔴 DEBUG: User ID for recommendations: '{user_id}' (type: {type(user_id)})")
        
        if user_id and user_id.strip(): 
            print("🟢 DEBUG: User ID provided, attempting to get recommendations...")
            try:
                config = RecommendationConfig()
                rec_engine = RecommendationEngine(db=db, config=config)
                print("🟢 DEBUG: Recommendation engine initialized")
                
                start_dt = datetime.strptime(f"{date} {start_time}", "%Y-%m-%d %H:%M")
                end_dt = datetime.strptime(f"{date} {end_time}", "%Y-%m-%d %H:%M")
                print("🟢 DEBUG: Datetime objects created")
                
                recommendations = await rec_engine.get_recommendations(
                    user_id=user_id.strip(),
                    request_type="room_not_found",
                    context={
                        "requested_room": room_name,
                        "date": date,
                        "start_time": start_time,
                        "end_time": end_time,
                        "start_datetime": start_dt,
                        "end_datetime": end_dt,
                        "failure_reason": "room_not_found"
                    }
                )
                print(f"🟢 DEBUG: Recommendations received: {recommendations}")
                
                return {
                    "status": "room_not_found",
                    "message": f"Room '{room_name}' not found. Here are some alternative rooms you might like:",
                    "recommendations": recommendations or [],
                    "suggestion": "Try one of the recommended rooms or check the room name spelling.",
                    "debug_info": {
                        "user_id": user_id,
                        "recommendation_engine_used": True,
                        "recommendations_count": len(recommendations) if recommendations else 0
                    }
                }
                
            except Exception as e:
                print(f"🔴 DEBUG: Failed to get recommendations: {str(e)}")
                import traceback
                print(f"🔴 DEBUG: Full traceback: {traceback.format_exc()}")
                
                logger.error(f"Failed to get recommendations for room not found: {str(e)}")
                # Fallback to basic room suggestions
                available_rooms = db.query(models.MRBSRoom).limit(5).all()
                room_suggestions = [{"room_name": room.room_name, "capacity": getattr(room, 'capacity', 'N/A')} for room in available_rooms]
                
                return {
                    "status": "room_not_found", 
                    "message": f"Room '{room_name}' not found. Here are some available rooms:",
                    "suggestions": room_suggestions,
                    "note": "Please check the room name spelling or try one of the suggested rooms.",
                    "debug_info": {
                        "user_id": user_id,
                        "recommendation_engine_used": False,
                        "error": str(e),
                        "fallback_used": True
                    }
                }
        else:
            print("🔴 DEBUG: No user_id provided, returning basic error")
            # No user_id provided, just return basic error
            return {
                "status": "room_not_found",
                "message": f"Room '{room_name}' not found. Please check the room name and try again.",
                "debug_info": {
                    "user_id": None,
                    "recommendation_engine_used": False,
                    "reason": "No user_id provided"
                }
            }

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
        message = f"{room_name} is NOT available at that time."
        print(message)
        return {"status": "unavailable", "message": message}

    message = f"{room_name} is available from {start_time} to {end_time} on {date}."
    print(message)
    return {"status": "available", "message": message}

async def add_booking(room_name: str, date: str, start_time: str, end_time: str, created_by: str, db: Session, user_id: str = None):

    room = db.query(models.MRBSRoom).filter(models.MRBSRoom.room_name == room_name).first()
    
    if not room:
        
        if user_id:
            try:
                config = RecommendationConfig()
                rec_engine = RecommendationEngine(db=db, config=config)
                
                # Convert time strings to datetime for recommendation context
                start_dt = datetime.strptime(f"{date} {start_time}", "%Y-%m-%d %H:%M")
                end_dt = datetime.strptime(f"{date} {end_time}", "%Y-%m-%d %H:%M")
                
                # Get recommendations for alternative rooms with booking intent
                recommendations = await rec_engine.get_recommendations(
                    user_id=user_id,
                    request_type="booking_room_not_found",
                    context={
                        "requested_room": room_name,
                        "date": date,
                        "start_time": start_time,
                        "end_time": end_time,
                        "start_datetime": start_dt,
                        "end_datetime": end_dt,
                        "created_by": created_by,
                        "intent": "booking",
                        "failure_reason": "room_not_found"
                    }
                )
                
                return {
                    "status": "booking_failed_room_not_found",
                    "message": f"Cannot book '{room_name}' - room not found. Here are some alternative rooms available at your requested time:",
                    "recommendations": recommendations or [],
                    "suggested_action": "Would you like to book one of these recommended rooms instead?",
                    "original_request": {
                        "room_name": room_name,
                        "date": date,
                        "start_time": start_time,
                        "end_time": end_time
                    }
                }
                
            except Exception as e:
                logger.error(f"Failed to get booking recommendations for room not found: {str(e)}")
                # Fallback to checking available rooms at the requested time
                available_rooms_at_time = await get_available_rooms_at_time(date, start_time, end_time, db)
                
                return {
                    "status": "booking_failed_room_not_found",
                    "message": f"Cannot book '{room_name}' - room not found. Here are rooms available at your requested time:",
                    "available_alternatives": available_rooms_at_time,
                    "suggestion": "Try booking one of these available rooms instead."
                }
        else:
            return {
                "status": "booking_failed_room_not_found",
                "message": f"Cannot book '{room_name}' - room not found. Please check the room name and try again."
            }

    # Step 2: Convert times
    start_dt = datetime.strptime(f"{date} {start_time}", "%Y-%m-%d %H:%M")
    end_dt = datetime.strptime(f"{date} {end_time}", "%Y-%m-%d %H:%M")
    start_ts = int(time.mktime(start_dt.timetuple()))
    end_ts = int(time.mktime(end_dt.timetuple()))

    if end_ts <= start_ts:
        return {
            "status": "invalid_time",
            "message": "End time must be after start time"
        }

    # Step 3: Check for conflicts
    conflict = db.query(models.MRBSEntry).filter(
        models.MRBSEntry.room_id == room.id,
        models.MRBSEntry.start_time < end_ts,
        models.MRBSEntry.end_time > start_ts,
    ).first()
    
    if conflict:
        return {
            "status": "time_conflict",
            "message": "Time slot is already booked",
            "conflict_details": {
                "conflicting_booking_id": conflict.id,
                "conflict_start": datetime.fromtimestamp(conflict.start_time).strftime("%Y-%m-%d %H:%M"),
                "conflict_end": datetime.fromtimestamp(conflict.end_time).strftime("%Y-%m-%d %H:%M")
            }
        }

    # Step 4: Insert booking
    new_booking = models.MRBSEntry(
        start_time=start_ts,
        end_time=end_ts,
        entry_type=0,
        repeat_id=None,
        room_id=room.id,
        create_by=created_by,
        modified_by=created_by,
        name=f"Booking for {room_name}",
        type='E',
        description=f"Booked by {created_by}",
        status=0,
        ical_uid=f"{room_name}_{start_ts}_{end_ts}",
        ical_sequence=0,
        ical_recur_id=None
    )
    db.add(new_booking)
    db.commit()
    db.refresh(new_booking)

    return {
        "status": "success",
        "message": "Booking created successfully",
        "booking_id": new_booking.id,
        "room": room_name,
        "date": date,
        "start_time": start_time,
        "end_time": end_time
    }

async def check_available_slotes(date: str, start_time: str, end_time: str, db: Session, room_name: str = None, user_id: str = None):
    """
    Modified to work with or without room_name, and integrate recommendations
    """
    print(f"Checking available slots for date: {date}, Start time: {start_time}, End time: {end_time}")
    
    target_rooms = []
    
    if room_name:
        # Check specific room
        room = db.query(models.MRBSRoom).filter(models.MRBSRoom.room_name == room_name).first()
        print(f"Queried room from DB: {room}")

        if not room:
            print("Room not found!")
            
            # Get recommendations when specific room not found
            if user_id:
                try:
                    config = RecommendationConfig()
                    rec_engine = RecommendationEngine(db=db, config=config)
                    
                    recommendations = await rec_engine.get_recommendations(
                        user_id=user_id,
                        request_type="slot_check_room_not_found",
                        context={
                            "requested_room": room_name,
                            "date": date,
                            "start_time": start_time,
                            "end_time": end_time,
                            "failure_reason": "room_not_found"
                        }
                    )
                    
                    return {
                        "status": "room_not_found",
                        "message": f"Room '{room_name}' not found. Here are recommended alternatives with available slots:",
                        "recommendations": recommendations or [],
                        "date": date
                    }
                    
                except Exception as e:
                    logger.error(f"Failed to get slot recommendations: {str(e)}")
            
            return {
                "status": "room_not_found",
                "message": f"Room '{room_name}' not found. Please check the room name.",
                "date": date
            }
        
        target_rooms = [room]
    else:
        # Get all rooms for general availability check
        target_rooms = db.query(models.MRBSRoom).all()
    
    all_available_slots = {}
    
    for room in target_rooms:
        # Convert to datetime objects first
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        slot_start_time = datetime.combine(date_obj, datetime.min.time()) + timedelta(hours=7)  # 7 AM
        slot_end_time = datetime.combine(date_obj, datetime.min.time()) + timedelta(hours=21)  # 9 PM

        all_slots = []
        current = slot_start_time
        while current < slot_end_time:
            slot_start = current
            slot_end = current + timedelta(minutes=30)
            all_slots.append((int(time.mktime(slot_start.timetuple())), int(time.mktime(slot_end.timetuple()))))
            current = slot_end

        # Step 3: Get all bookings for that day and room
        day_start_ts = int(time.mktime(slot_start_time.timetuple()))
        day_end_ts = int(time.mktime(slot_end_time.timetuple()))

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

        all_available_slots[room.room_name] = available_slots

    # If checking specific room, return in original format
    if room_name and len(target_rooms) == 1:
        return {
            "status": "success",
            "room": room_name, 
            "date": date, 
            "available_slots": all_available_slots.get(room_name, [])
        }
    
    # Return all rooms with their available slots
    return {
        "status": "success",
        "date": date,
        "rooms_availability": all_available_slots,
        "total_rooms": len(target_rooms)
    }

async def get_available_rooms_at_time(date: str, start_time: str, end_time: str, db: Session):
    """
    Helper function to get all available rooms at a specific time
    """
    try:
        # Convert to datetime objects and then to timestamps
        start_dt = datetime.strptime(f"{date} {start_time}", "%Y-%m-%d %H:%M")
        end_dt = datetime.strptime(f"{date} {end_time}", "%Y-%m-%d %H:%M")
        start_ts = int(time.mktime(start_dt.timetuple()))
        end_ts = int(time.mktime(end_dt.timetuple()))
        
        # Get all rooms
        all_rooms = db.query(models.MRBSRoom).all()
        available_rooms = []
        
        for room in all_rooms:
            # Check if room has any conflicts at the requested time
            conflict = db.query(models.MRBSEntry).filter(
                models.MRBSEntry.room_id == room.id,
                models.MRBSEntry.start_time < end_ts,
                models.MRBSEntry.end_time > start_ts,
            ).first()
            
            if not conflict:
                available_rooms.append({
                    "room_name": room.room_name,
                    "room_id": room.id,
                    "capacity": getattr(room, 'capacity', 'N/A'),
                    "available_time": f"{start_time} - {end_time}"
                })
        
        return available_rooms
        
    except Exception as e:
        logger.error(f"Error getting available rooms at time: {str(e)}")
        return []
    
async def get_alternative_time_slots(room_name: str, date: str, preferred_duration: int, db: Session):
    """Get alternative time slots for the same room"""
    try:
        room = db.query(models.MRBSRoom).filter(models.MRBSRoom.room_name == room_name).first()
        if not room:
            return []
        
        # Get available slots for the day
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        day_start = datetime.combine(date_obj, datetime.min.time()) + timedelta(hours=8)  # 8 AM
        day_end = datetime.combine(date_obj, datetime.min.time()) + timedelta(hours=20)  # 8 PM
        
        # Get all bookings for that day
        day_start_ts = int(time.mktime(day_start.timetuple()))
        day_end_ts = int(time.mktime(day_end.timetuple()))
        
        bookings = db.query(models.MRBSEntry).filter(
            models.MRBSEntry.room_id == room.id,
            models.MRBSEntry.start_time < day_end_ts,
            models.MRBSEntry.end_time > day_start_ts
        ).all()
        
        # Find available slots of the preferred duration
        available_slots = []
        current = day_start
        
        while current + timedelta(minutes=preferred_duration) <= day_end:
            slot_start_ts = int(time.mktime(current.timetuple()))
            slot_end_ts = int(time.mktime((current + timedelta(minutes=preferred_duration)).timetuple()))
            
            # Check if this slot conflicts with any booking
            conflict = any(
                booking.start_time < slot_end_ts and booking.end_time > slot_start_ts
                for booking in bookings
            )
            
            if not conflict:
                available_slots.append({
                    "room_name": room_name,
                    "date": date,
                    "start_time": current.strftime("%H:%M"),
                    "end_time": (current + timedelta(minutes=preferred_duration)).strftime("%H:%M"),
                    "duration_minutes": preferred_duration,
                    "quick_book_query": f"Book {room_name} on {date} from {current.strftime('%H:%M')} to {(current + timedelta(minutes=preferred_duration)).strftime('%H:%M')}"
                })
            
            current += timedelta(minutes=30)  # Check every 30 minutes
        
        return available_slots[:5]  # Return top 5 alternatives
        
    except Exception as e:
        logger.error(f"Error getting alternative time slots: {str(e)}")
        return []

def calculate_duration(start_time: str, end_time: str) -> int:
    """Calculate duration in minutes between start and end time"""
    try:
        start = datetime.strptime(start_time, "%H:%M")
        end = datetime.strptime(end_time, "%H:%M")
        duration = (end - start).total_seconds() / 60
        return int(duration)
    except:
        return 60  # Default to 1 hour

async def get_basic_alternatives(params: dict, db: Session):
    """Get basic alternatives when AI recommendations fail"""
    try:
        # Get alternative rooms available at the same time
        available_rooms = await get_available_rooms_at_time(
            date=params["date"],
            start_time=params["start_time"],
            end_time=params["end_time"],
            db=db
        )
        
        # Get alternative times for the same room
        alternative_times = await get_alternative_time_slots(
            room_name=params["room_name"],
            date=params["date"],
            preferred_duration=calculate_duration(params["start_time"], params["end_time"]),
            db=db
        )
        
        return {
            "alternative_rooms": available_rooms[:3],
            "alternative_times": alternative_times[:3],
            "note": "Basic alternatives - AI recommendations unavailable"
        }
        
    except Exception as e:
        logger.error(f"Error getting basic alternatives: {str(e)}")
        return {
            "alternative_rooms": [],
            "alternative_times": [],
            "note": "Unable to find alternatives at this time"
        }

def format_quick_book_options(recommendations, alternative_slots):
    """Format quick booking options for easy user selection"""
    quick_options = []
    
    if recommendations:
        for i, rec in enumerate(recommendations[:3]):
            quick_options.append({
                "option_id": f"room_{i+1}",
                "type": "alternative_room",
                "description": f"Book {rec.get('room_name', 'Unknown')} instead",
                "quick_query": f"Book {rec.get('room_name', 'Unknown')} on {rec.get('date', '')} from {rec.get('start_time', '')} to {rec.get('end_time', '')}"
            })
    
    # Add time slot alternatives
    if alternative_slots:
        for i, slot in enumerate(alternative_slots[:2]):
            quick_options.append({
                "option_id": f"time_{i+1}",
                "type": "alternative_time",
                "description": f"Book same room at {slot['start_time']}-{slot['end_time']}",
                "quick_query": slot.get("quick_book_query", "")
            })
    
    return quick_options

async def get_smart_recommendations(
    params: dict, 
    db, 
    user_id: str, 
    request_type: str, 
    failure_reason: str = None
) -> dict:
    """
    Shared function to get intelligent recommendations for any scenario
    """
    try:
        config = RecommendationConfig()
        rec_engine = RecommendationEngine(db=db, config=config)
        
        # Prepare datetime objects if needed
        context = params.copy()
        if "date" in params and "start_time" in params and "end_time" in params:
            try:
                start_dt = datetime.strptime(f"{params['date']} {params['start_time']}", "%Y-%m-%d %H:%M")
                end_dt = datetime.strptime(f"{params['date']} {params['end_time']}", "%Y-%m-%d %H:%M")
                context.update({
                    "start_datetime": start_dt,
                    "end_datetime": end_dt
                })
            except ValueError:
                logger.warning("Could not parse datetime from params")
        
        # Add request context
        context.update({
            "original_request": request_type,
            "failure_reason": failure_reason,
            "purpose": "meeting",  # default purpose
            "capacity": 10  # default capacity
        })
        
        # Get AI recommendations
        recommendations = await rec_engine.get_recommendations(
            user_id=user_id,
            request_type="comprehensive",
            context=context
        )
        
        # Get alternative time slots if room_name is provided
        alternative_slots = []
        if "room_name" in params and "date" in params and "start_time" in params and "end_time" in params:
            try:
                duration = calculate_duration(params["start_time"], params["end_time"])
                alternative_slots = await get_alternative_time_slots(
                    room_name=params["room_name"],
                    date=params["date"],
                    preferred_duration=duration,
                    db=db
                )
            except Exception as e:
                logger.warning(f"Could not get alternative time slots: {e}")
        
        # Learn from this interaction
        if failure_reason:
            try:
                await rec_engine.learn_from_booking(
                    user_id=user_id,
                    booking_data=params,
                    outcome=failure_reason
                )
            except Exception as e:
                logger.warning(f"Learning from interaction failed: {e}")
        
        return {
            "recommendations": recommendations or [],
            "alternative_slots": alternative_slots or [],
            "total_alternatives": len(recommendations or []) + len(alternative_slots or []),
            "quick_book_options": format_quick_book_options(recommendations, alternative_slots)
        }
        
    except Exception as e:
        logger.error(f"Smart recommendations failed: {e}")
        # Fallback to basic alternatives
        try:
            basic_alternatives = await get_basic_alternatives(params, db)
            return {
                "recommendations": [],
                "alternative_slots": [],
                "basic_alternatives": basic_alternatives,
                "total_alternatives": len(basic_alternatives.get("alternative_rooms", [])) + len(basic_alternatives.get("alternative_times", [])),
                "fallback_used": True
            }
        except Exception as fallback_error:
            logger.error(f"Even basic alternatives failed: {fallback_error}")
            return {
                "recommendations": [],
                "alternative_slots": [],
                "basic_alternatives": {"alternative_rooms": [], "alternative_times": []},
                "total_alternatives": 0,
                "error": "Could not generate any alternatives"
            }

async def handle_unavailable_room(
    params: dict, 
    db, 
    user_id: str, 
    base_message: str,
    request_type: str = "general"
) -> dict:
    """
    Shared function to handle room unavailability with smart recommendations
    """
    smart_recs = await get_smart_recommendations(
        params=params,
        db=db,
        user_id=user_id,
        request_type=request_type,
        failure_reason="room_unavailable"
    )
    
    if smart_recs.get("fallback_used"):
        return {
            "status": "alternatives_found",
            "message": f"{base_message} Don't worry - we found some alternatives for you!",
            "basic_alternatives": smart_recs["basic_alternatives"],
            "total_alternatives": smart_recs["total_alternatives"],
            "note": "AI recommendations temporarily unavailable. Here are basic alternatives.",
            "suggested_actions": [
                "Try one of the alternative rooms at your preferred time",
                "Book the same room at a different time",
                "Modify your requirements and search again"
            ]
        }
    
    # AI recommendations available
    total_alternatives = smart_recs["total_alternatives"]
    
    if total_alternatives > 0:
        return {
            "status": "smart_alternatives_found",
            "message": f"{base_message} Great news! We found {total_alternatives} intelligent alternatives based on your preferences:",
            "recommendations": {
                "alternative_rooms": smart_recs["recommendations"],
                "alternative_times": smart_recs["alternative_slots"],
                "total_alternatives": total_alternatives
            },
            "suggested_actions": [
                "📍 Book one of the recommended alternative rooms",
                "⏰ Choose a different time slot for the same room", 
                "🔧 Modify your time requirements",
                "💡 Say 'Book option 1' to instantly book the first alternative"
            ],
            "quick_book_options": smart_recs["quick_book_options"],
            "user_friendly_message": f"We analyzed your booking patterns and found perfect matches for you!",
            "original_request": {
                "room_name": params.get("room_name"),
                "date": params.get("date"),
                "start_time": params.get("start_time"),
                "end_time": params.get("end_time")
            }
        }
    else:
        return {
            "status": "no_alternatives",
            "message": f"{base_message} Unfortunately, we couldn't find any suitable alternatives at this time.",
            "suggested_actions": [
                "Try a different date",
                "Modify your time requirements",
                "Check back later for cancellations",
                "Contact support for assistance"
            ],
            "help_text": "You can also try asking for recommendations with more flexible requirements."
        }