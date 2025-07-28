# recommendtion/recommendations/data/analytics_processor.py
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from src.models import MRBSEntry, MRBSRoom
import json

class AnalyticsProcessor:
    """
    Processes booking data to extract patterns and insights for recommendations
    """
    
    def __init__(self, db: Session):
        self.db = db
        self._cache = {}  # Simple in-memory cache
        self._cache_ttl = {}
    
    async def get_user_booking_patterns(self, user_id: str) -> Dict[str, Any]:
        """
        Analyze user's booking patterns and preferences
        """
        cache_key = f"user_patterns_{user_id}"
        if self._is_cached_valid(cache_key):
            return self._cache[cache_key]
        
        # Get user's booking history
        user_bookings = self.db.query(MRBSEntry).filter(
            MRBSEntry.create_by == user_id
        ).order_by(MRBSEntry.start_time.desc()).limit(100).all()
        
        if not user_bookings:
            return self._cache_result(cache_key, {
                'total_bookings': 0,
                'preferred_rooms': [],
                'preferred_times': [],
                'booking_patterns': {},
                'seasonal_patterns': {},
                'recurring_patterns': []
            })
        
        patterns = {
            'total_bookings': len(user_bookings),
            'preferred_rooms': self._analyze_room_preferences(user_bookings),
            'preferred_times': self._analyze_time_preferences(user_bookings),
            'booking_patterns': self._analyze_booking_patterns(user_bookings),
            'seasonal_patterns': self._analyze_seasonal_patterns(user_bookings),
            'recurring_patterns': self._analyze_recurring_patterns(user_bookings),
            'recent_bookings': self._get_recent_bookings(user_bookings),
            'typically_available_times': self._infer_available_times(user_bookings),
            'prefers_weekdays': self._analyze_weekday_preference(user_bookings)
        }
        
        return self._cache_result(cache_key, patterns)
    
    def _analyze_room_preferences(self, bookings: List[MRBSEntry]) -> List[str]:
        """Analyze which rooms the user prefers"""
        room_counts = defaultdict(int)
        
        for booking in bookings:
            room = self.db.query(MRBSRoom).filter(MRBSRoom.id == booking.room_id).first()
            if room:
                room_counts[room.room_name] += 1
        
        # Sort by frequency and return top rooms
        sorted_rooms = sorted(room_counts.items(), key=lambda x: x[1], reverse=True)
        return [room[0] for room in sorted_rooms[:5]]
    
    def _analyze_time_preferences(self, bookings: List[MRBSEntry]) -> List[str]:
        """Analyze user's preferred booking times"""
        time_counts = defaultdict(int)
        
        for booking in bookings:
            start_time = datetime.fromtimestamp(booking.start_time)
            time_slot = start_time.strftime("%H:%M")
            time_counts[time_slot] += 1
        
        # Sort by frequency and return top times
        sorted_times = sorted(time_counts.items(), key=lambda x: x[1], reverse=True)
        return [time[0] for time in sorted_times[:5]]
    
    def _analyze_booking_patterns(self, bookings: List[MRBSEntry]) -> Dict[str, Any]:
        """Analyze general booking patterns"""
        durations = []
        weekdays = []
        advance_bookings = []
        
        for booking in bookings:
            # Duration analysis
            duration_hours = (booking.end_time - booking.start_time) / 3600
            durations.append(duration_hours)
            
            # Weekday analysis
            booking_date = datetime.fromtimestamp(booking.start_time)
            weekdays.append(booking_date.weekday())
            
            # Advance booking analysis (if timestamp available)
            if hasattr(booking, 'timestamp') and booking.timestamp:
                booking_created = booking.timestamp
                advance_days = (booking_date - booking_created).days
                advance_bookings.append(advance_days)
        
        return {
            'avg_duration': np.mean(durations) if durations else 0,
            'preferred_weekdays': Counter(weekdays).most_common(3),
            'avg_advance_booking': np.mean(advance_bookings) if advance_bookings else 0,
            'frequency': self._calculate_booking_frequency(bookings)
        }
    
    def _analyze_seasonal_patterns(self, bookings: List[MRBSEntry]) -> Dict[str, Any]:
        """Analyze seasonal booking patterns"""
        monthly_patterns = defaultdict(lambda: {'count': 0, 'preferred_rooms': [], 'preferred_times': []})
        
        for booking in bookings:
            booking_date = datetime.fromtimestamp(booking.start_time)
            month = booking_date.month
            
            monthly_patterns[month]['count'] += 1
            
            # Get room name
            room = self.db.query(MRBSRoom).filter(MRBSRoom.id == booking.room_id).first()
            if room:
                monthly_patterns[month]['preferred_rooms'].append(room.room_name)
            
            # Get time
            time_slot = booking_date.strftime("%H:%M")
            monthly_patterns[month]['preferred_times'].append({
                'start_time': time_slot,
                'end_time': datetime.fromtimestamp(booking.end_time).strftime("%H:%M")
            })
        
        # Process patterns to get most common items
        for month, data in monthly_patterns.items():
            data['preferred_rooms'] = [item[0] for item in Counter(data['preferred_rooms']).most_common(3)]
            data['preferred_times'] = data['preferred_times'][:3]  # Keep recent times
        
        return dict(monthly_patterns)
    
    def _analyze_recurring_patterns(self, bookings: List[MRBSEntry]) -> List[Dict[str, Any]]:
        """Detect recurring booking patterns"""
        recurring_patterns = []
        
        # Group bookings by room and time
        grouped_bookings = defaultdict(list)
        
        for booking in bookings:
            room = self.db.query(MRBSRoom).filter(MRBSRoom.id == booking.room_id).first()
            if not room:
                continue
            
            start_time = datetime.fromtimestamp(booking.start_time)
            end_time = datetime.fromtimestamp(booking.end_time)
            
            key = (room.room_name, start_time.strftime("%H:%M"), end_time.strftime("%H:%M"))
            grouped_bookings[key].append(start_time)
        
        # Analyze for recurring patterns
        for (room_name, start_time, end_time), dates in grouped_bookings.items():
            if len(dates) >= 3:  # At least 3 occurrences
                dates.sort()
                
                # Check for weekly pattern
                weekly_intervals = []
                for i in range(1, len(dates)):
                    interval = (dates[i] - dates[i-1]).days
                    if 6 <= interval <= 8:  # Weekly (allowing some variance)
                        weekly_intervals.append(interval)
                
                if len(weekly_intervals) >= 2:
                    recurring_patterns.append({
                        'room_name': room_name,
                        'start_time': start_time,
                        'end_time': end_time,
                        'frequency': 'weekly',
                        'occurrences': len(dates),
                        'consistency_score': len(weekly_intervals) / (len(dates) - 1),
                        'last_booking_date': dates[-1].strftime("%Y-%m-%d")
                    })
        
        return recurring_patterns
    
    def _get_recent_bookings(self, bookings: List[MRBSEntry]) -> List[Dict[str, Any]]:
        """Get recent bookings for collaborative filtering"""
        recent = []
        
        for booking in bookings[:10]:  # Last 10 bookings
            room = self.db.query(MRBSRoom).filter(MRBSRoom.id == booking.room_id).first()
            if room:
                start_time = datetime.fromtimestamp(booking.start_time)
                end_time = datetime.fromtimestamp(booking.end_time)
                
                recent.append({
                    'room_name': room.room_name,
                    'date': start_time.strftime("%Y-%m-%d"),
                    'start_time': start_time.strftime("%H:%M"),
                    'end_time': end_time.strftime("%H:%M")
                })
        
        return recent
    
    def _infer_available_times(self, bookings: List[MRBSEntry]) -> List[str]:
        """Infer when user is typically available based on booking patterns"""
        booking_times = set()
        
        for booking in bookings:
            start_time = datetime.fromtimestamp(booking.start_time)
            # Create time slots for the entire duration
            current = start_time
            end_time = datetime.fromtimestamp(booking.end_time)
            
            while current < end_time:
                booking_times.add(current.strftime("%H:%M"))
                current += timedelta(minutes=30)
        
        # Generate all possible business hour slots
        all_slots = []
        for hour in range(7, 21):  # 7 AM to 9 PM
            for minute in [0, 30]:
                all_slots.append(f"{hour:02d}:{minute:02d}")
        
        # Return slots not frequently booked (likely available times)
        available_times = [slot for slot in all_slots if slot not in booking_times]
        return available_times[:10]  # Return top 10 available slots
    
    def _analyze_weekday_preference(self, bookings: List[MRBSEntry]) -> bool:
        """Check if user prefers weekdays over weekends"""
        weekday_count = 0
        weekend_count = 0
        
        for booking in bookings:
            booking_date = datetime.fromtimestamp(booking.start_time)
            if booking_date.weekday() < 5:  # Monday-Friday
                weekday_count += 1
            else:
                weekend_count += 1
        
        return weekday_count > weekend_count * 2  # Strong preference for weekdays
    
    def _calculate_booking_frequency(self, bookings: List[MRBSEntry]) -> str:
        """Calculate how frequently user books rooms"""
        if len(bookings) < 2:
            return 'occasional'
        
        # Get time span of bookings
        first_booking = datetime.fromtimestamp(bookings[-1].start_time)
        last_booking = datetime.fromtimestamp(bookings[0].start_time)
        
        days_span = (last_booking - first_booking).days
        if days_span == 0:
            days_span = 1
        
        frequency = len(bookings) / days_span
        
        if frequency > 0.5:
            return 'frequent'
        elif frequency > 0.1:
            return 'regular'
        else:
            return 'occasional'
    
    async def get_room_features(self, room_name: str) -> Dict[str, Any]:
        """Get features and characteristics of a room"""
        cache_key = f"room_features_{room_name}"
        if self._is_cached_valid(cache_key):
            return self._cache[cache_key]
        
        room = self.db.query(MRBSRoom).filter(MRBSRoom.room_name == room_name).first()
        if not room:
            return self._cache_result(cache_key, {})
        
        # Get booking statistics for this room
        recent_bookings = self.db.query(MRBSEntry).filter(
            MRBSEntry.room_id == room.id,
            MRBSEntry.start_time > int((datetime.now() - timedelta(days=30)).timestamp())
        ).all()
        
        features = {
            'id': room.id,
            'name': room.room_name,
            'capacity': room.capacity,
            'description': room.description,
            'area_id': room.area_id,
            'popularity_score': len(recent_bookings),
            'avg_booking_duration': self._calculate_avg_duration(recent_bookings),
            'peak_hours': self._get_peak_hours(recent_bookings),
            'utilization_rate': self._calculate_utilization_rate(recent_bookings)
        }
        
        return self._cache_result(cache_key, features)
    
    def _calculate_avg_duration(self, bookings: List[MRBSEntry]) -> float:
        """Calculate average booking duration for a room"""
        if not bookings:
            return 0
        
        durations = [(booking.end_time - booking.start_time) / 3600 for booking in bookings]
        return np.mean(durations)
    
    def _get_peak_hours(self, bookings: List[MRBSEntry]) -> List[str]:
        """Get peak booking hours for a room"""
        hour_counts = defaultdict(int)
        
        for booking in bookings:
            start_hour = datetime.fromtimestamp(booking.start_time).hour
            hour_counts[start_hour] += 1
        
        # Return top 3 peak hours
        sorted_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)
        return [f"{hour:02d}:00" for hour, _ in sorted_hours[:3]]
    
    def _calculate_utilization_rate(self, bookings: List[MRBSEntry]) -> float:
        """Calculate room utilization rate"""
        if not bookings:
            return 0
        
        # Calculate total booked hours in the last 30 days
        total_booked_hours = sum((booking.end_time - booking.start_time) / 3600 for booking in bookings)
        
        # Assume 14 hours per day available (7 AM to 9 PM) for 30 days
        total_available_hours = 14 * 30
        
        return min(1.0, total_booked_hours / total_available_hours)
    
    async def get_user_room_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user's room preferences and historical choices"""
        patterns = await self.get_user_booking_patterns(user_id)
        
        return {
            'preferred_rooms': patterns.get('preferred_rooms', []),
            'avoided_rooms': [],  # Could be inferred from rejected recommendations
            'capacity_preference': self._infer_capacity_preference(user_id),
            'location_preference': self._infer_location_preference(user_id)
        }
    
    def _infer_capacity_preference(self, user_id: str) -> Dict[str, Any]:
        """Infer user's capacity preferences"""
        user_bookings = self.db.query(MRBSEntry).filter(
            MRBSEntry.create_by == user_id
        ).limit(50).all()
        
        capacities = []
        for booking in user_bookings:
            room = self.db.query(MRBSRoom).filter(MRBSRoom.id == booking.room_id).first()
            if room:
                capacities.append(room.capacity)
        
        if capacities:
            return {
                'avg_preferred_capacity': np.mean(capacities),
                'min_capacity': min(capacities),
                'max_capacity': max(capacities)
            }
        
        return {'avg_preferred_capacity': 20, 'min_capacity': 10, 'max_capacity': 50}
    
    def _infer_location_preference(self, user_id: str) -> Dict[str, Any]:
        """Infer user's location preferences"""
        user_bookings = self.db.query(MRBSEntry).filter(
            MRBSEntry.create_by == user_id
        ).limit(50).all()
        
        area_counts = defaultdict(int)
        for booking in user_bookings:
            room = self.db.query(MRBSRoom).filter(MRBSRoom.id == booking.room_id).first()
            if room:
                area_counts[room.area_id] += 1
        
        preferred_areas = sorted(area_counts.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'preferred_areas': [area[0] for area in preferred_areas[:3]],
            'area_flexibility': len(area_counts) > 1  # User books in multiple areas
        }
    
    def get_room_popular_times(self, room_name: str) -> List[str]:
        """Get popular booking times for a specific room"""
        room = self.db.query(MRBSRoom).filter(MRBSRoom.room_name == room_name).first()
        if not room:
            return []
        
        recent_bookings = self.db.query(MRBSEntry).filter(
            MRBSEntry.room_id == room.id,
            MRBSEntry.start_time > int((datetime.now() - timedelta(days=60)).timestamp())
        ).all()
        
        time_counts = defaultdict(int)
        for booking in recent_bookings:
            start_time = datetime.fromtimestamp(booking.start_time)
            time_slot = start_time.strftime("%H:%M")
            time_counts[time_slot] += 1
        
        # Return top 5 popular times
        sorted_times = sorted(time_counts.items(), key=lambda x: x[1], reverse=True)
        return [time[0] for time in sorted_times[:5]]
    
    async def get_optimal_booking_times(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get optimal booking times across all rooms"""
        cache_key = f"optimal_times_{limit}"
        if self._is_cached_valid(cache_key):
            return self._cache[cache_key]
        
        # Get all recent bookings
        recent_bookings = self.db.query(MRBSEntry).filter(
            MRBSEntry.start_time > int((datetime.now() - timedelta(days=30)).timestamp())
        ).all()
        
        # Analyze time slots and availability
        time_room_stats = defaultdict(lambda: {'bookings': 0, 'rooms': set(), 'avg_duration': []})
        
        for booking in recent_bookings:
            start_time = datetime.fromtimestamp(booking.start_time)
            time_slot = start_time.strftime("%H:%M")
            duration = (booking.end_time - booking.start_time) / 3600
            
            room = self.db.query(MRBSRoom).filter(MRBSRoom.id == booking.room_id).first()
            if room:
                time_room_stats[time_slot]['bookings'] += 1
                time_room_stats[time_slot]['rooms'].add(room.room_name)
                time_room_stats[time_slot]['avg_duration'].append(duration)
        
        # Calculate optimal times
        optimal_times = []
        for time_slot, stats in time_room_stats.items():
            if stats['bookings'] > 0:
                popularity = stats['bookings'] / 30  # Average per day
                room_variety = len(stats['rooms'])
                avg_duration = np.mean(stats['avg_duration'])
                
                # Calculate optimality score
                optimality_score = popularity * 0.4 + room_variety * 0.3 + (2 / avg_duration) * 0.3
                
                optimal_times.append({
                    'time_slot': time_slot,
                    'start_time': time_slot,
                    'end_time': self._add_hours(time_slot, avg_duration),
                    'room_name': list(stats['rooms'])[0] if stats['rooms'] else 'Various',
                    'popularity': popularity,
                    'optimality_score': optimality_score,
                    'avg_duration': avg_duration
                })
        
        # Sort by optimality score
        optimal_times.sort(key=lambda x: x['optimality_score'], reverse=True)
        
        return self._cache_result(cache_key, optimal_times[:limit])
    
    def _add_hours(self, time_str: str, hours: float) -> str:
        """Add hours to a time string"""
        try:
            time_obj = datetime.strptime(time_str, "%H:%M")
            new_time = time_obj + timedelta(hours=hours)
            return new_time.strftime("%H:%M")
        except:
            return time_str
    
    async def get_room_analytics(self, room_name: str) -> Dict[str, Any]:
        """Get comprehensive analytics for a room"""
        cache_key = f"room_analytics_{room_name}"
        if self._is_cached_valid(cache_key):
            return self._cache[cache_key]
        
        room_features = await self.get_room_features(room_name)
        popular_times = self.get_room_popular_times(room_name)
        
        # Get additional analytics
        room = self.db.query(MRBSRoom).filter(MRBSRoom.room_name == room_name).first()
        if not room:
            return self._cache_result(cache_key, {})
        
        # Recent booking patterns
        recent_bookings = self.db.query(MRBSEntry).filter(
            MRBSEntry.room_id == room.id,
            MRBSEntry.start_time > int((datetime.now() - timedelta(days=30)).timestamp())
        ).all()
        
        analytics = {
            **room_features,
            'popular_times': popular_times,
            'total_bookings_30_days': len(recent_bookings),
            'unique_users': len(set(booking.create_by for booking in recent_bookings)),
            'booking_trends': self._analyze_booking_trends(recent_bookings),
            'conflict_analysis': self._analyze_conflicts(room.id),
            'recommendations': self._generate_room_recommendations(room_features, recent_bookings)
        }
        
        return self._cache_result(cache_key, analytics)
    
    def _analyze_booking_trends(self, bookings: List[MRBSEntry]) -> Dict[str, Any]:
        """Analyze booking trends for a room"""
        if not bookings:
            return {'trend': 'stable', 'weekly_pattern': {}}
        
        # Group by week
        weekly_bookings = defaultdict(int)
        for booking in bookings:
            week = datetime.fromtimestamp(booking.start_time).isocalendar()[1]
            weekly_bookings[week] += 1
        
        weeks = sorted(weekly_bookings.keys())
        if len(weeks) >= 2:
            first_half = np.mean([weekly_bookings[w] for w in weeks[:len(weeks)//2]])
            second_half = np.mean([weekly_bookings[w] for w in weeks[len(weeks)//2:]])
            
            if second_half > first_half * 1.2:
                trend = 'increasing'
            elif second_half < first_half * 0.8:
                trend = 'decreasing'
            else:
                trend = 'stable'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'weekly_pattern': dict(weekly_bookings)
        }
    
    def _analyze_conflicts(self, room_id: int) -> Dict[str, Any]:
        """Analyze booking conflicts for a room"""
        # Get overlapping bookings (shouldn't exist but good to check)
        overlapping = self.db.query(MRBSEntry).filter(
            MRBSEntry.room_id == room_id
        ).all()
        
        conflicts = 0
        for i, booking1 in enumerate(overlapping):
            for booking2 in overlapping[i+1:]:
                if (booking1.start_time < booking2.end_time and 
                    booking1.end_time > booking2.start_time):
                    conflicts += 1
        
        return {
            'total_conflicts': conflicts,
            'conflict_rate': conflicts / len(overlapping) if overlapping else 0
        }
    
    def _generate_room_recommendations(
        self, room_features: Dict[str, Any], bookings: List[MRBSEntry]
    ) -> List[str]:
        """Generate recommendations for room usage optimization"""
        recommendations = []
        
        utilization = room_features.get('utilization_rate', 0)
        if utilization < 0.3:
            recommendations.append("Room is underutilized - consider promoting for smaller meetings")
        elif utilization > 0.8:
            recommendations.append("Room is highly utilized - consider capacity management")
        
        avg_duration = room_features.get('avg_booking_duration', 0)
        if avg_duration < 1:
            recommendations.append("Mostly short bookings - suitable for quick meetings")
        elif avg_duration > 3:
            recommendations.append("Long bookings typical - consider for workshops/training")
        
        return recommendations
    
    async def update_user_preferences(self, user_id: str, booking_data: Dict[str, Any]):
        """Update user preferences based on successful booking"""
        # This would update user preference models
        # For now, we'll invalidate the cache to force refresh
        cache_key = f"user_patterns_{user_id}"
        if cache_key in self._cache:
            del self._cache[cache_key]
            del self._cache_ttl[cache_key]
    
    def _is_cached_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self._cache:
            return False
        
        if cache_key not in self._cache_ttl:
            return False
        
        return datetime.now() < self._cache_ttl[cache_key]
    
    def _cache_result(self, cache_key: str, result: Any, ttl_minutes: int = 30) -> Any:
        """Cache result with TTL"""
        self._cache[cache_key] = result
        self._cache_ttl[cache_key] = datetime.now() + timedelta(minutes=ttl_minutes)
        return result