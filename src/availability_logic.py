from sqlalchemy.orm import Session
from datetime import datetime
import time
from fastapi import HTTPException
from . import models
from datetime import datetime, timedelta

def check_availability(room_name: str, date: str, start_time: str, end_time: str, db: Session):
    print(f"Checking availability for room: {room_name}")
    print(f"Date: {date}, Start time: {start_time}, End time: {end_time}")

    room = db.query(models.MRBSRoom).filter(models.MRBSRoom.room_name == room_name).first()
    print(f"Queried room from DB: {room}")

    if not room:
        print("Room not found!")
        raise HTTPException(status_code=404, detail="Room not found")

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
        message = f"{room_name} is NOT available at that time."
        print(message)
        return {"status": "unavailable", "message": message}

    message = f"{room_name} is available from {start_time} to {end_time} on {date}."
    print(message)
    return {"status": "available", "message": message}

def add_booking(room_name: str, date: str, start_time: str, end_time: str, created_by: str, db: Session):
    # Step 1: Check if room exists
    room = db.query(models.MRBSRoom).filter(models.MRBSRoom.room_name == room_name).first()
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")

    # Step 2: Convert times
    start_dt = datetime.strptime(f"{date} {start_time}", "%Y-%m-%d %H:%M")
    end_dt = datetime.strptime(f"{date} {end_time}", "%Y-%m-%d %H:%M")
    start_ts = int(time.mktime(start_dt.timetuple()))
    end_ts = int(time.mktime(end_dt.timetuple()))

    if end_ts <= start_ts:
        raise HTTPException(status_code=400, detail="End time must be after start time")

    # Step 3: Check for conflicts
    conflict = db.query(models.MRBSEntry).filter(
        models.MRBSEntry.room_id == room.id,
        models.MRBSEntry.start_time < end_ts,
        models.MRBSEntry.end_time > start_ts,
    ).first()
    if conflict:
        raise HTTPException(status_code=409, detail="Time slot is already booked")

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
        "message": "Booking created successfully",
        "booking_id": new_booking.id,
        "room": room_name,
        "date": date,
        "start_time": start_time,
        "end_time": end_time
    }
def check_available_slotes(room_name: str, date: str, start_time: str, end_time: str, db: Session):
    # alternative slotes
    print(f"Checking availability for room: {room_name}")
    print(f"Date: {date}, Start time: {start_time}, End time: {end_time}")

    room = db.query(models.MRBSRoom).filter(models.MRBSRoom.room_name == room_name).first()
    print(f"Queried room from DB: {room}")

    if not room:
        print("Room not found!")
        raise HTTPException(status_code=404, detail="Room not found")

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

    return {"room": room_name, "date": date, "available_slots": available_slots}