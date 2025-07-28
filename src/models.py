#src.models.py
from sqlalchemy import Column, Integer, String, Text, ForeignKey, TIMESTAMP, SmallInteger, Boolean
from sqlalchemy.orm import relationship
from src.database import Base


class MRBSRoom(Base):
    __tablename__ = "mrbs_room"

    id = Column(Integer, primary_key=True, autoincrement=True)
    disabled = Column(Boolean, nullable=False, default=False)
    area_id = Column(Integer, ForeignKey("mrbs_area.id", onupdate="CASCADE"), nullable=False, default=0)
    room_name = Column(String(25), nullable=False, unique=True)
    sort_key = Column(String(25), nullable=False, default="")
    description = Column(String(60), nullable=True)
    capacity = Column(Integer, nullable=False, default=0)
    room_admin_email = Column(Text, nullable=True)
    custom_html = Column(Text, nullable=True)

    # Relationship to bookings
    bookings = relationship("MRBSEntry", back_populates="room")


class MRBSRepeat(Base):
    __tablename__ = "mrbs_repeat"

    id = Column(Integer, primary_key=True, autoincrement=True)
    start_time = Column(Integer, nullable=False, default=0)  # Unix timestamp
    end_time = Column(Integer, nullable=False, default=0)  # Unix timestamp
    entry_type = Column(Integer, nullable=False, default=0)
    timestamp = Column(TIMESTAMP, nullable=False)
    create_by = Column(String(80), nullable=False, default="")
    modified_by = Column(String(80), nullable=False, default="")
    name = Column(String(80), nullable=False, default="")
    type = Column(String(1), nullable=False, default="E")
    description = Column(Text, nullable=True)
    status = Column(SmallInteger, nullable=False, default=0)
    reminded = Column(Integer, nullable=True)
    info_time = Column(Integer, nullable=True)
    info_user = Column(String(80), nullable=True)
    info_text = Column(Text, nullable=True)
    ical_uid = Column(String(255), nullable=False, default="")
    ical_sequence = Column(SmallInteger, nullable=False, default=0)
    ical_recur_id = Column(String(16), nullable=True)

    # Relationship to entries
    entries = relationship("MRBSEntry", back_populates="repeat")


class MRBSEntry(Base):
    __tablename__ = "mrbs_entry"

    id = Column(Integer, primary_key=True, autoincrement=True)
    start_time = Column(Integer, nullable=False, default=0)  # Unix timestamp
    end_time = Column(Integer, nullable=False, default=0)  # Unix timestamp
    entry_type = Column(Integer, nullable=False, default=0)
    repeat_id = Column(Integer, ForeignKey("mrbs_repeat.id", ondelete="CASCADE", onupdate="CASCADE"), nullable=True)
    room_id = Column(Integer, ForeignKey("mrbs_room.id", onupdate="CASCADE"), nullable=False, default=1)
    timestamp = Column(TIMESTAMP, nullable=False)
    create_by = Column(String(80), nullable=False, default="")
    modified_by = Column(String(80), nullable=False, default="")
    name = Column(String(80), nullable=False, default="")
    type = Column(String(1), nullable=False, default="E")
    description = Column(Text, nullable=True)
    status = Column(SmallInteger, nullable=False, default=0)
    reminded = Column(Integer, nullable=True)
    info_time = Column(Integer, nullable=True)
    info_user = Column(String(80), nullable=True)
    info_text = Column(Text, nullable=True)
    ical_uid = Column(String(255), nullable=False, default="")
    ical_sequence = Column(SmallInteger, nullable=False, default=0)
    ical_recur_id = Column(String(16), nullable=True)

    # Relationships
    room = relationship("MRBSRoom", back_populates="bookings")
    repeat = relationship("MRBSRepeat", back_populates="entries")
