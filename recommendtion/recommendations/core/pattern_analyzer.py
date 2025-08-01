# recommendations/core/pattern_analyzer.py

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import text
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)

class PatternAnalyzer:
    """
    Analyzes booking patterns from user data to identify trends and preferences
    """
    
    def __init__(self, db_session: Session = None):
        """
        Initialize PatternAnalyzer
        
        Args:
            db_session: Database session for querying booking data
        """
        self.db_session = db_session
        self.patterns_cache = {}
        
        if not db_session:
            logger.warning("PatternAnalyzer initialized without database session - running in mock mode")
    
    def analyze_user_patterns(
        self, 
        user_id: str, 
        booking_history: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze booking patterns for a specific user
        
        Args:
            user_id: User identifier
            booking_history: List of booking records (optional, will query if not provided)
            
        Returns:
            Dictionary containing analyzed patterns
        """
        try:
            # Check cache first
            cache_key = f"user_patterns_{user_id}"
            if cache_key in self.patterns_cache:
                logger.debug(f"Returning cached patterns for user {user_id}")
                return self.patterns_cache[cache_key]
            
            # Get booking history if not provided
            if booking_history is None:
                booking_history = self._get_user_booking_history(user_id)
            
            if not booking_history:
                logger.info(f"No booking history found for user {user_id}, returning default patterns")
                return self._get_default_patterns()
            
            # Analyze patterns
            patterns = {
                'preferred_hours': self._analyze_time_preferences(booking_history),
                'preferred_days': self._analyze_day_preferences(booking_history),
                'meeting_duration_patterns': self._analyze_duration_patterns(booking_history),
                'room_preferences': self._analyze_room_preferences(booking_history),
                'booking_frequency': self._analyze_booking_frequency(booking_history),
                'advance_booking_patterns': self._analyze_advance_booking(booking_history),
                'seasonal_patterns': self._analyze_seasonal_patterns(booking_history),
                'total_bookings': len(booking_history),
                'analysis_date': datetime.now().isoformat()
            }
            
            self.patterns_cache[cache_key] = patterns
            
            logger.info(f"Analyzed patterns for user {user_id}: {len(booking_history)} bookings")
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing patterns for user {user_id}: {e}")
            return self._get_default_patterns()
    
    def _get_user_booking_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Get booking history from database"""
        if not self.db_session:
            logger.warning("No database session available, returning empty history")
            return []
        
        try:
            # Query last 90 days of bookings
            query = text("""
                SELECT e.*, r.room_name, r.capacity
                FROM mrbs_entry e
                JOIN mrbs_room r ON e.room_id = r.id
                WHERE e.create_by = :user_id 
                AND e.start_time >= :start_date
                ORDER BY e.start_time DESC
            """)
            
            start_date = int((datetime.now() - timedelta(days=90)).timestamp())
            
            result = self.db_session.execute(query, {
                'user_id': user_id,
                'start_date': start_date
            }).fetchall()
            
            # Convert to list of dictionaries
            bookings = []
            for row in result:
                booking = {
                    'id': row.id,
                    'start_time': datetime.fromtimestamp(row.start_time),
                    'end_time': datetime.fromtimestamp(row.end_time),
                    'room_id': row.room_id,
                    'room_name': row.room_name,
                    'capacity': row.capacity,
                    'name': row.name,
                    'description': row.description or '',
                    'create_by': row.create_by,
                    'created_at': row.timestamp
                }
                bookings.append(booking)
            
            return bookings
            
        except Exception as e:
            logger.error(f"Error fetching booking history for user {user_id}: {e}")
            return []
    
    def _analyze_time_preferences(self, bookings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze preferred booking times"""
        hours = [booking['start_time'].hour for booking in bookings]
        hour_counts = Counter(hours)
        
        # Get top 3 preferred hours
        top_hours = [hour for hour, count in hour_counts.most_common(3)]
        
        # Calculate morning vs afternoon preference
        morning_count = sum(1 for hour in hours if 6 <= hour < 12)
        afternoon_count = sum(1 for hour in hours if 12 <= hour < 18)
        evening_count = sum(1 for hour in hours if 18 <= hour < 22)
        
        total = len(hours)
        
        return {
            'preferred_hours': top_hours,
            'hour_distribution': dict(hour_counts),
            'morning_preference': morning_count / total if total > 0 else 0,
            'afternoon_preference': afternoon_count / total if total > 0 else 0,
            'evening_preference': evening_count / total if total > 0 else 0,
            'peak_hour': hour_counts.most_common(1)[0][0] if hour_counts else 9
        }
    
    def _analyze_day_preferences(self, bookings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze preferred booking days"""
        days = [booking['start_time'].weekday() for booking in bookings]
        day_counts = Counter(days)
        
        # Convert to day names
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_name_counts = {day_names[day]: count for day, count in day_counts.items()}
        
        # Get top 3 preferred days
        top_days = [day for day, count in day_counts.most_common(3)]
        
        return {
            'preferred_days': top_days,
            'day_distribution': day_name_counts,
            'weekday_vs_weekend': {
                'weekday': sum(1 for day in days if day < 5) / len(days) if days else 0,
                'weekend': sum(1 for day in days if day >= 5) / len(days) if days else 0
            }
        }
    
    def _analyze_duration_patterns(self, bookings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze meeting duration patterns"""
        durations = []
        for booking in bookings:
            duration_minutes = int((booking['end_time'] - booking['start_time']).total_seconds() / 60)
            durations.append(duration_minutes)
        
        if not durations:
            return {'average_duration': 60, 'common_durations': [30, 60, 90]}
        
        duration_counts = Counter(durations)
        common_durations = [duration for duration, count in duration_counts.most_common(3)]
        
        return {
            'average_duration': sum(durations) / len(durations),
            'common_durations': common_durations,
            'duration_distribution': dict(duration_counts),
            'short_meetings': sum(1 for d in durations if d <= 30) / len(durations),
            'medium_meetings': sum(1 for d in durations if 30 < d <= 90) / len(durations),
            'long_meetings': sum(1 for d in durations if d > 90) / len(durations)
        }
    
    def _analyze_room_preferences(self, bookings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze room booking preferences"""
        rooms = [booking['room_name'] for booking in bookings if booking.get('room_name')]
        room_counts = Counter(rooms)
        
        capacities = [booking['capacity'] for booking in bookings if booking.get('capacity')]
        capacity_counts = Counter(capacities)
        
        return {
            'preferred_rooms': [room for room, count in room_counts.most_common(3)],
            'room_distribution': dict(room_counts),
            'preferred_capacities': [cap for cap, count in capacity_counts.most_common(3)],
            'average_capacity': sum(capacities) / len(capacities) if capacities else 6
        }
    
    def _analyze_booking_frequency(self, bookings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze how frequently user books meetings"""
        if not bookings:
            return {'bookings_per_week': 0, 'booking_trend': 'stable'}
        
        # Group bookings by week
        weekly_counts = defaultdict(int)
        for booking in bookings:
            week_key = booking['start_time'].strftime('%Y-W%U')
            weekly_counts[week_key] += 1
        
        weekly_values = list(weekly_counts.values())
        avg_per_week = sum(weekly_values) / len(weekly_values) if weekly_values else 0
        
        # Determine trend (simple linear trend)
        if len(weekly_values) >= 2:
            recent_avg = sum(weekly_values[-4:]) / min(4, len(weekly_values))
            older_avg = sum(weekly_values[:-4]) / max(1, len(weekly_values) - 4)
            
            if recent_avg > older_avg * 1.1:
                trend = 'increasing'
            elif recent_avg < older_avg * 0.9:
                trend = 'decreasing'
            else:
                trend = 'stable'
        else:
            trend = 'stable'
        
        return {
            'bookings_per_week': avg_per_week,
            'booking_trend': trend,
            'total_weeks': len(weekly_counts),
            'weekly_distribution': dict(weekly_counts)
        }
    
    def _analyze_advance_booking(self, bookings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze how far in advance user books meetings"""
        advance_days = []
        
        for booking in bookings:
            if booking.get('created_at'):
                created_at = booking['created_at']
                if isinstance(created_at, datetime):
                    advance = (booking['start_time'] - created_at).days
                    advance_days.append(max(0, advance))  # Ensure non-negative
        
        if not advance_days:
            return {'average_advance_days': 1, 'booking_style': 'last_minute'}
        
        avg_advance = sum(advance_days) / len(advance_days)
        
        # Categorize booking style
        if avg_advance < 1:
            style = 'last_minute'
        elif avg_advance < 7:
            style = 'short_term'
        elif avg_advance < 30:
            style = 'medium_term'
        else:
            style = 'long_term'
        
        return {
            'average_advance_days': avg_advance,
            'booking_style': style,
            'advance_distribution': {
                'same_day': sum(1 for d in advance_days if d == 0) / len(advance_days),
                'within_week': sum(1 for d in advance_days if 0 < d <= 7) / len(advance_days),
                'within_month': sum(1 for d in advance_days if 7 < d <= 30) / len(advance_days),
                'beyond_month': sum(1 for d in advance_days if d > 30) / len(advance_days)
            }
        }
    
    def _analyze_seasonal_patterns(self, bookings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze seasonal booking patterns"""
        months = [booking['start_time'].month for booking in bookings]
        month_counts = Counter(months)
        
        # Map months to seasons
        season_mapping = {
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'fall', 10: 'fall', 11: 'fall'
        }
        
        seasons = [season_mapping[month] for month in months]
        season_counts = Counter(seasons)
        
        return {
            'monthly_distribution': dict(month_counts),
            'seasonal_distribution': dict(season_counts),
            'peak_month': month_counts.most_common(1)[0][0] if month_counts else 1,
            'peak_season': season_counts.most_common(1)[0][0] if season_counts else 'spring'
        }
    
    def _get_default_patterns(self) -> Dict[str, Any]:
        """Return default patterns when no data is available"""
        return {
            'preferred_hours': [9, 10, 14, 15],
            'preferred_days': [0, 1, 2, 3, 4],  # Monday to Friday
            'meeting_duration_patterns': {
                'average_duration': 60,
                'common_durations': [30, 60, 90]
            },
            'room_preferences': {
                'preferred_rooms': [],
                'average_capacity': 6
            },
            'booking_frequency': {
                'bookings_per_week': 2,
                'booking_trend': 'stable'
            },
            'advance_booking_patterns': {
                'average_advance_days': 2,
                'booking_style': 'short_term'
            },
            'seasonal_patterns': {
                'peak_month': 3,
                'peak_season': 'spring'
            },
            'total_bookings': 0,
            'analysis_date': datetime.now().isoformat()
        }
    
    def clear_cache(self, user_id: str = None):
        """Clear pattern cache for specific user or all users"""
        if user_id:
            cache_key = f"user_patterns_{user_id}"
            self.patterns_cache.pop(cache_key, None)
            logger.info(f"Cleared pattern cache for user {user_id}")
        else:
            self.patterns_cache.clear()
            logger.info("Cleared all pattern cache")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'cached_users': len(self.patterns_cache),
            'cache_keys': list(self.patterns_cache.keys()),
            'memory_usage_estimate': len(str(self.patterns_cache))
        }