# recommendtion/recommendations/utils/time_utils.py
from datetime import datetime, timedelta, time
from typing import List, Dict, Any, Optional, Tuple
import calendar
import logging

logger = logging.getLogger(__name__)

class TimeUtils:
    """
    Utility class for time-related operations in the recommendation system
    """
    
    @staticmethod
    def time_to_minutes(time_str: str) -> int:
        """
        Convert time string to minutes since midnight
        
        Args:
            time_str: Time in HH:MM format
            
        Returns:
            Minutes since midnight
        """
        try:
            if isinstance(time_str, str):
                hour, minute = map(int, time_str.split(':'))
            else:
                # Handle datetime.time objects
                hour, minute = time_str.hour, time_str.minute
            return hour * 60 + minute
        except (ValueError, AttributeError) as e:
            logger.error(f"Error converting time to minutes: {time_str}, {e}")
            return 0
    
    @staticmethod
    def minutes_to_time(minutes: int) -> str:
        """
        Convert minutes since midnight to time string
        
        Args:
            minutes: Minutes since midnight
            
        Returns:
            Time string in HH:MM format
        """
        try:
            hours = minutes // 60
            mins = minutes % 60
            return f"{hours:02d}:{mins:02d}"
        except Exception as e:
            logger.error(f"Error converting minutes to time: {minutes}, {e}")
            return "00:00"
    
    @staticmethod
    def parse_time_range(time_range: str) -> Tuple[str, str]:
        """
        Parse time range string like "09:00-11:00" into start and end times
        
        Args:
            time_range: Time range string
            
        Returns:
            Tuple of (start_time, end_time)
        """
        try:
            start_str, end_str = time_range.split('-')
            return start_str.strip(), end_str.strip()
        except ValueError:
            logger.error(f"Invalid time range format: {time_range}")
            return "09:00", "10:00"
    
    @staticmethod
    def is_business_hours(time_str: str, start_hour: int = 7, end_hour: int = 21) -> bool:
        """
        Check if time falls within business hours
        
        Args:
            time_str: Time string in HH:MM format
            start_hour: Business start hour (default 7 AM)
            end_hour: Business end hour (default 9 PM)
            
        Returns:
            True if within business hours
        """
        try:
            minutes = TimeUtils.time_to_minutes(time_str)
            start_minutes = start_hour * 60
            end_minutes = end_hour * 60
            return start_minutes <= minutes < end_minutes
        except Exception:
            return False
    
    @staticmethod
    def get_time_slot_category(time_str: str) -> str:
        """
        Categorize time into morning, afternoon, evening
        
        Args:
            time_str: Time string in HH:MM format
            
        Returns:
            Time category string
        """
        try:
            minutes = TimeUtils.time_to_minutes(time_str)
            
            if minutes < 720:  # Before 12:00
                return "morning"
            elif minutes < 1080:  # Before 18:00
                return "afternoon"
            else:
                return "evening"
        except Exception:
            return "unknown"
    
    @staticmethod
    def calculate_duration_hours(start_time: str, end_time: str) -> float:
        """
        Calculate duration in hours between two time strings
        
        Args:
            start_time: Start time in HH:MM format
            end_time: End time in HH:MM format
            
        Returns:
            Duration in hours
        """
        try:
            start_minutes = TimeUtils.time_to_minutes(start_time)
            end_minutes = TimeUtils.time_to_minutes(end_time)
            
            # Handle overnight meetings
            if end_minutes < start_minutes:
                end_minutes += 24 * 60
            
            return (end_minutes - start_minutes) / 60.0
        except Exception as e:
            logger.error(f"Error calculating duration: {e}")
            return 1.0
    
    @staticmethod
    def generate_time_slots(
        start_time: str, 
        end_time: str, 
        slot_duration: int = 30
    ) -> List[str]:
        """
        Generate time slots between start and end time
        
        Args:
            start_time: Start time in HH:MM format
            end_time: End time in HH:MM format
            slot_duration: Duration of each slot in minutes
            
        Returns:
            List of time slot strings
        """
        try:
            slots = []
            start_minutes = TimeUtils.time_to_minutes(start_time)
            end_minutes = TimeUtils.time_to_minutes(end_time)
            
            current = start_minutes
            while current < end_minutes:
                slots.append(TimeUtils.minutes_to_time(current))
                current += slot_duration
            
            return slots
        except Exception as e:
            logger.error(f"Error generating time slots: {e}")
            return []
    
    @staticmethod
    def is_weekend(date_obj: datetime) -> bool:
        """Check if date is weekend"""
        return date_obj.weekday() >= 5
    
    @staticmethod
    def get_weekday_name(date_obj: datetime) -> str:
        """Get weekday name from datetime object"""
        return date_obj.strftime('%A')
    
    @staticmethod
    def get_next_business_day(date_obj: datetime) -> datetime:
        """
        Get next business day (Monday-Friday) from given date
        
        Args:
            date_obj: Starting date
            
        Returns:
            Next business day
        """
        next_day = date_obj + timedelta(days=1)
        while next_day.weekday() >= 5:  # Skip weekends
            next_day += timedelta(days=1)
        return next_day
    
    @staticmethod
    def get_business_days_between(start_date: datetime, end_date: datetime) -> List[datetime]:
        """
        Get all business days between two dates
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            List of business day datetime objects
        """
        business_days = []
        current_date = start_date
        
        while current_date <= end_date:
            if current_date.weekday() < 5:  # Monday to Friday
                business_days.append(current_date)
            current_date += timedelta(days=1)
        
        return business_days
    
    @staticmethod
    def format_duration(hours: float) -> str:
        """
        Format duration hours into human readable string
        
        Args:
            hours: Duration in hours
            
        Returns:
            Formatted duration string
        """
        if hours < 1:
            minutes = int(hours * 60)
            return f"{minutes} minutes"
        elif hours == 1:
            return "1 hour"
        elif hours == int(hours):
            return f"{int(hours)} hours"
        else:
            whole_hours = int(hours)
            minutes = int((hours - whole_hours) * 60)
            return f"{whole_hours}h {minutes}m"
    
    @staticmethod
    def get_time_preference_score(
        time_str: str, 
        user_patterns: Dict[str, Any]
    ) -> float:
        """
        Calculate preference score for a time based on user patterns
        
        Args:
            time_str: Time string in HH:MM format
            user_patterns: User booking patterns
            
        Returns:
            Preference score between 0 and 1
        """
        try:
            base_score = 0.5
            time_minutes = TimeUtils.time_to_minutes(time_str)
            
            # Check against preferred times
            preferred_times = user_patterns.get('preferred_times', [])
            for pref_time in preferred_times:
                pref_minutes = TimeUtils.time_to_minutes(pref_time)
                if abs(time_minutes - pref_minutes) <= 30:
                    base_score += 0.3
                    break
            
            # Check time category preference
            time_category = TimeUtils.get_time_slot_category(time_str)
            category_preferences = user_patterns.get('time_category_preferences', {})
            
            if time_category in category_preferences:
                base_score += category_preferences[time_category] * 0.2
            
            return min(1.0, max(0.0, base_score))
            
        except Exception as e:
            logger.error(f"Error calculating time preference score: {e}")
            return 0.5
    
    @staticmethod
    def normalize_time_format(time_input: Any) -> str:
        """
        Normalize various time input formats to HH:MM string
        
        Args:
            time_input: Time in various formats (string, datetime.time, etc.)
            
        Returns:
            Normalized time string in HH:MM format
        """
        try:
            if isinstance(time_input, str):
                # Handle different string formats
                if ':' in time_input:
                    parts = time_input.split(':')
                    hour = int(parts[0])
                    minute = int(parts[1])
                    return f"{hour:02d}:{minute:02d}"
                else:
                    # Assume it's just hour
                    hour = int(time_input)
                    return f"{hour:02d}:00"
            
            elif isinstance(time_input, time):
                return time_input.strftime("%H:%M")
            
            elif isinstance(time_input, datetime):
                return time_input.strftime("%H:%M")
            
            else:
                logger.warning(f"Unknown time format: {time_input}")
                return "09:00"
                
        except Exception as e:
            logger.error(f"Error normalizing time format: {time_input}, {e}")
            return "09:00"
    
    @staticmethod
    def is_time_conflict(
        start1: str, end1: str, 
        start2: str, end2: str
    ) -> bool:
        """
        Check if two time ranges conflict
        
        Args:
            start1, end1: First time range
            start2, end2: Second time range
            
        Returns:
            True if times conflict
        """
        try:
            start1_min = TimeUtils.time_to_minutes(start1)
            end1_min = TimeUtils.time_to_minutes(end1)
            start2_min = TimeUtils.time_to_minutes(start2)
            end2_min = TimeUtils.time_to_minutes(end2)
            
            return start1_min < end2_min and start2_min < end1_min
            
        except Exception as e:
            logger.error(f"Error checking time conflict: {e}")
            return True  # Assume conflict on error for safety
    
    @staticmethod
    def get_optimal_meeting_times() -> List[str]:
        """
        Get list of commonly preferred meeting times
        
        Returns:
            List of optimal meeting time strings
        """
        return [
            "09:00", "09:30", "10:00", "10:30", "11:00", "11:30",
            "13:00", "13:30", "14:00", "14:30", "15:00", "15:30",
            "16:00", "16:30"
        ]
    
    @staticmethod
    def calculate_time_distance(time1: str, time2: str) -> int:
        """
        Calculate distance between two times in minutes
        
        Args:
            time1: First time string
            time2: Second time string
            
        Returns:
            Distance in minutes
        """
        try:
            min1 = TimeUtils.time_to_minutes(time1)
            min2 = TimeUtils.time_to_minutes(time2)
            return abs(min1 - min2)
        except Exception as e:
            logger.error(f"Error calculating time distance: {e}")
            return 0
    
    @staticmethod
    def round_to_nearest_slot(time_str: str, slot_duration: int = 30) -> str:
        """
        Round time to nearest time slot
        
        Args:
            time_str: Time string to round
            slot_duration: Slot duration in minutes
            
        Returns:
            Rounded time string
        """
        try:
            minutes = TimeUtils.time_to_minutes(time_str)
            rounded_minutes = (minutes // slot_duration) * slot_duration
            return TimeUtils.minutes_to_time(rounded_minutes)
        except Exception as e:
            logger.error(f"Error rounding time to slot: {e}")
            return time_str