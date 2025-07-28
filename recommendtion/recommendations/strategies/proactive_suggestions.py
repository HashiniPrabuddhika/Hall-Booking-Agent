# recommendtion/recommendations/strategies/proactive_suggestions.py
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from ..data.analytics_processor import AnalyticsProcessor
from ..models.embedding_model import EmbeddingModel
from ..utils.time_utils import TimeUtils
from ..utils.metrics import RecommendationMetrics
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class BookingSuggestion:
    """Data class for booking suggestions"""
    type: str
    room_name: str
    date: str
    start_time: str
    end_time: str
    confidence_score: float
    reason: str
    metadata: Dict[str, Any]

class ProactiveSuggestionStrategy:
    """
    Strategy for proactive booking suggestions based on patterns and predictions
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.analytics = AnalyticsProcessor(db)
        self.embedding_model = EmbeddingModel()
        self.time_utils = TimeUtils()
        self.metrics = RecommendationMetrics()
        
        # Configuration
        self.max_suggestions = 5
        self.min_confidence_threshold = 0.3
        self.business_hours = {
            'start': '08:00',
            'end': '20:00'
        }
        self.prediction_window_days = 14
    
    async def predict_future_bookings(
        self,
        user_id: str,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Predict and suggest future bookings for the user
        """
        try:
            suggestions = []
            
            # Get user's booking history and patterns
            user_patterns = await self.analytics.get_user_booking_patterns(user_id)
            
            if not user_patterns:
                logger.warning(f"No patterns found for user {user_id}")
                return await self._suggest_default_bookings(user_id, context)
            
            # Strategy 1: Recurring pattern suggestions
            recurring_suggestions = await self._suggest_recurring_bookings(
                user_id, user_patterns
            )
            suggestions.extend(recurring_suggestions)
            
            # Strategy 2: Seasonal/trend-based suggestions
            seasonal_suggestions = await self._suggest_seasonal_bookings(
                user_id, user_patterns
            )
            suggestions.extend(seasonal_suggestions)
            
            # Strategy 3: Collaborative filtering suggestions
            collaborative_suggestions = await self._suggest_collaborative_bookings(
                user_id, user_patterns
            )
            suggestions.extend(collaborative_suggestions)
            
            # Strategy 4: Context-aware suggestions
            context_suggestions = await self._suggest_context_aware_bookings(
                user_id, context, user_patterns
            )
            suggestions.extend(context_suggestions)
            
            # Strategy 5: Gap-filling suggestions
            gap_suggestions = await self._suggest_gap_filling_bookings(
                user_id, user_patterns
            )
            suggestions.extend(gap_suggestions)
            
            # Filter by confidence threshold and remove duplicates
            filtered_suggestions = self._filter_and_deduplicate_suggestions(suggestions)
            
            # Sort by confidence and return top suggestions
            filtered_suggestions.sort(key=lambda x: x['confidence_score'], reverse=True)
            
            # Log metrics
            await self.metrics.log_proactive_suggestions(
                user_id, len(filtered_suggestions), context
            )
            
            return filtered_suggestions[:self.max_suggestions]
            
        except Exception as e:
            logger.error(f"Error predicting future bookings for user {user_id}: {str(e)}")
            return await self._suggest_default_bookings(user_id, context)
    
    async def _suggest_recurring_bookings(
        self,
        user_id: str,
        user_patterns: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Suggest bookings based on recurring patterns"""
        
        suggestions = []
        recurring_patterns = user_patterns.get('recurring_patterns', [])
        
        for pattern in recurring_patterns:
            # Calculate next occurrence
            last_booking = pattern.get('last_booking_date')
            frequency = pattern.get('frequency')  # 'weekly', 'monthly', etc.
            
            if not last_booking or not frequency:
                continue
            
            try:
                last_date = datetime.strptime(last_booking, "%Y-%m-%d")
                next_dates = self._calculate_next_occurrences(last_date, frequency, 3)
                
                for next_date in next_dates:
                    # Only suggest future dates
                    if next_date > datetime.now():
                        confidence = self._calculate_recurring_confidence(pattern)
                        
                        # Check room availability
                        is_available = await self._check_room_availability(
                            pattern.get('room_name'),
                            next_date.strftime("%Y-%m-%d"),
                            pattern.get('start_time'),
                            pattern.get('end_time')
                        )
                        
                        if is_available:
                            suggestions.append({
                                'type': 'recurring',
                                'room_name': pattern.get('room_name'),
                                'date': next_date.strftime("%Y-%m-%d"),
                                'start_time': pattern.get('start_time'),
                                'end_time': pattern.get('end_time'),
                                'confidence_score': confidence,
                                'reason': f'Based on your {frequency} booking pattern',
                                'metadata': {
                                    'pattern_strength': pattern.get('occurrences', 1),
                                    'last_booking': last_booking,
                                    'frequency': frequency,
                                    'consistency_score': pattern.get('consistency_score', 0.5)
                                }
                            })
                    
            except ValueError as e:
                logger.warning(f"Error parsing date {last_booking}: {str(e)}")
                continue
        
        return suggestions
    
    async def _suggest_seasonal_bookings(
        self,
        user_id: str,
        user_patterns: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Suggest bookings based on seasonal trends"""
        
        suggestions = []
        seasonal_patterns = user_patterns.get('seasonal_patterns', {})
        
        current_month = datetime.now().month
        current_season = self._get_season(current_month)
        
        # Check patterns for current month and season
        month_pattern = seasonal_patterns.get(str(current_month), {})
        season_pattern = seasonal_patterns.get(current_season, {})
        
        patterns_to_check = [month_pattern, season_pattern]
        
        for pattern in patterns_to_check:
            if not pattern:
                continue
                
            preferred_rooms = pattern.get('preferred_rooms', [])[:2]
            preferred_times = pattern.get('preferred_times', [])[:2]
            
            for room in preferred_rooms:
                for time_slot in preferred_times:
                    # Suggest for next suitable dates
                    suggested_dates = self._get_next_suitable_dates(
                        pattern.get('preferred_weekdays', []), 
                        days_ahead=7
                    )
                    
                    for suggested_date in suggested_dates[:2]:
                        is_available = await self._check_room_availability(
                            room,
                            suggested_date.strftime("%Y-%m-%d"),
                            time_slot.get('start_time'),
                            time_slot.get('end_time')
                        )
                        
                        if is_available:
                            confidence = self._calculate_seasonal_confidence(pattern, current_month)
                            
                            suggestions.append({
                                'type': 'seasonal',
                                'room_name': room,
                                'date': suggested_date.strftime("%Y-%m-%d"),
                                'start_time': time_slot.get('start_time'),
                                'end_time': time_slot.get('end_time'),
                                'confidence_score': confidence,
                                'reason': f'You typically book {room} in {datetime.now().strftime("%B")}',
                                'metadata': {
                                    'seasonal_factor': pattern.get('frequency', 1),
                                    'season': current_season,
                                    'month_preference': pattern.get('preference_score', 0.5)
                                }
                            })
        
        return suggestions
    
    async def _suggest_collaborative_bookings(
        self,
        user_id: str,
        user_patterns: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Suggest bookings based on similar users' behavior"""
        
        suggestions = []
        
        try:
            # Find similar users using embeddings
            similar_users = await self.embedding_model.find_similar_users(user_id, n_results=5)
            
            for similar_user in similar_users:
                similar_user_id = similar_user['user_id']
                similarity_score = similar_user['similarity_score']
                
                if similarity_score < 0.3:  # Skip users with low similarity
                    continue
                
                # Get recent bookings of similar user
                similar_user_patterns = await self.analytics.get_user_booking_patterns(similar_user_id)
                recent_bookings = similar_user_patterns.get('recent_bookings', [])
                
                for booking in recent_bookings[:3]:  # Top 3 recent bookings
                    try:
                        # Suggest similar booking for future dates
                        booking_date = datetime.strptime(booking['date'], "%Y-%m-%d")
                        
                        # Suggest for next week at same time
                        suggested_dates = [
                            booking_date + timedelta(days=7),
                            booking_date + timedelta(days=14)
                        ]
                        
                        for suggested_date in suggested_dates:
                            # Only suggest future dates
                            if suggested_date > datetime.now():
                                is_available = await self._check_room_availability(
                                    booking['room_name'],
                                    suggested_date.strftime("%Y-%m-%d"),
                                    booking['start_time'],
                                    booking['end_time']
                                )
                                
                                if is_available:
                                    confidence = similarity_score * 0.7  # Reduce confidence for collaborative suggestions
                                    
                                    suggestions.append({
                                        'type': 'collaborative',
                                        'room_name': booking['room_name'],
                                        'date': suggested_date.strftime("%Y-%m-%d"),
                                        'start_time': booking['start_time'],
                                        'end_time': booking['end_time'],
                                        'confidence_score': confidence,
                                        'reason': 'Similar users have booked this recently',
                                        'metadata': {
                                            'similarity_score': similarity_score,
                                            'similar_user_id': similar_user_id,
                                            'original_booking_date': booking['date']
                                        }
                                    })
                    except ValueError:
                        continue
        
        except Exception as e:
            logger.error(f"Error in collaborative filtering: {str(e)}")
        
        return suggestions
    
    async def _suggest_context_aware_bookings(
        self,
        user_id: str,
        context: Dict[str, Any],
        user_patterns: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Suggest bookings based on current context and smart scheduling"""
        
        suggestions = []
        
        # Context-based suggestions
        await self._handle_recent_booking_context(context, suggestions)
        await self._handle_time_preference_context(user_patterns, suggestions)
        await self._handle_room_preference_context(user_patterns, suggestions)
        await self._handle_workload_context(context, user_patterns, suggestions)
        
        return suggestions
    
    async def _suggest_gap_filling_bookings(
        self,
        user_id: str,
        user_patterns: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Suggest bookings to fill gaps in user's schedule"""
        
        suggestions = []
        
        try:
            # Get user's existing bookings for next 2 weeks
            upcoming_bookings = await self.analytics.get_upcoming_bookings(
                user_id, days_ahead=14
            )
            
            if not upcoming_bookings:
                return suggestions
            
            # Identify gaps in schedule
            schedule_gaps = self._identify_schedule_gaps(upcoming_bookings)
            
            # Get user's preferred rooms and times
            preferred_rooms = user_patterns.get('preferred_rooms', [])[:3]
            preferred_duration = user_patterns.get('average_duration_minutes', 60)
            
            for gap in schedule_gaps:
                if gap['duration_minutes'] >= preferred_duration:
                    for room in preferred_rooms:
                        # Check if room is available during the gap
                        is_available = await self._check_room_availability(
                            room,
                            gap['date'],
                            gap['start_time'],
                            self._add_minutes_to_time(gap['start_time'], preferred_duration)
                        )
                        
                        if is_available:
                            confidence = self._calculate_gap_filling_confidence(gap, user_patterns)
                            
                            suggestions.append({
                                'type': 'gap_filling',
                                'room_name': room,
                                'date': gap['date'],
                                'start_time': gap['start_time'],
                                'end_time': self._add_minutes_to_time(gap['start_time'], preferred_duration),
                                'confidence_score': confidence,
                                'reason': 'Fill gap in your schedule',
                                'metadata': {
                                    'gap_duration': gap['duration_minutes'],
                                    'gap_type': gap['type'],
                                    'before_booking': gap.get('before_booking'),
                                    'after_booking': gap.get('after_booking')
                                }
                            })
        
        except Exception as e:
            logger.error(f"Error in gap filling suggestions: {str(e)}")
        
        return suggestions
    
    async def _suggest_default_bookings(
        self,
        user_id: str,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Provide default suggestions when no patterns are available"""
        
        suggestions = []
        
        # Get popular time slots and rooms
        popular_slots = await self.analytics.get_popular_time_slots()
        popular_rooms = await self.analytics.get_popular_rooms()
        
        # Suggest popular combinations for next week
        next_week_start = datetime.now() + timedelta(days=7)
        
        for i in range(3):  # Suggest 3 popular options
            date = next_week_start + timedelta(days=i)
            
            if popular_slots and popular_rooms:
                slot = popular_slots[i % len(popular_slots)]
                room = popular_rooms[i % len(popular_rooms)]
                
                is_available = await self._check_room_availability(
                    room['name'],
                    date.strftime("%Y-%m-%d"),
                    slot['start_time'],
                    slot['end_time']
                )
                
                if is_available:
                    suggestions.append({
                        'type': 'default',
                        'room_name': room['name'],
                        'date': date.strftime("%Y-%m-%d"),
                        'start_time': slot['start_time'],
                        'end_time': slot['end_time'],
                        'confidence_score': 0.4,
                        'reason': 'Popular time slot and room combination',
                        'metadata': {
                            'popularity_score': slot.get('popularity', 0),
                            'room_rating': room.get('rating', 0)
                        }
                    })
        
        return suggestions
    
    # Helper methods
    
    def _calculate_next_occurrences(
        self, 
        last_date: datetime, 
        frequency: str, 
        count: int = 3
    ) -> List[datetime]:
        """Calculate next occurrences based on frequency"""
        
        occurrences = []
        current_date = last_date
        
        for _ in range(count):
            if frequency == 'daily':
                current_date = current_date + timedelta(days=1)
            elif frequency == 'weekly':
                current_date = current_date + timedelta(weeks=1)
            elif frequency == 'monthly':
                current_date = current_date + timedelta(days=30)
            elif frequency == 'bi-weekly':
                current_date = current_date + timedelta(weeks=2)
            else:
                current_date = current_date + timedelta(weeks=1)  # Default to weekly
            
            occurrences.append(current_date)
        
        return occurrences
    
    def _calculate_recurring_confidence(self, pattern: Dict[str, Any]) -> float:
        """Calculate confidence score for recurring pattern"""
        
        base_confidence = 0.8
        consistency_score = pattern.get('consistency_score', 0.5)
        occurrences = pattern.get('occurrences', 1)
        
        # Boost confidence based on consistency and frequency
        confidence = base_confidence * consistency_score
        
        # Boost for frequent patterns
        if occurrences >= 5:
            confidence += 0.1
        elif occurrences >= 10:
            confidence += 0.15
        
        return min(confidence, 1.0)
    
    def _calculate_seasonal_confidence(
        self, 
        pattern: Dict[str, Any], 
        current_month: int
    ) -> float:
        """Calculate confidence for seasonal suggestions"""
        
        base_confidence = 0.6
        frequency = pattern.get('frequency', 1)
        preference_score = pattern.get('preference_score', 0.5)
        
        confidence = base_confidence * preference_score
        
        # Boost for frequent seasonal patterns
        if frequency >= 3:
            confidence += 0.1
        
        return min(confidence, 0.85)
    
    def _calculate_gap_filling_confidence(
        self, 
        gap: Dict[str, Any], 
        user_patterns: Dict[str, Any]
    ) -> float:
        """Calculate confidence for gap filling suggestions"""
        
        base_confidence = 0.5
        gap_duration = gap['duration_minutes']
        preferred_duration = user_patterns.get('average_duration_minutes', 60)
        
        # Higher confidence if gap matches preferred duration
        duration_match = 1 - abs(gap_duration - preferred_duration) / preferred_duration
        confidence = base_confidence + (duration_match * 0.2)
        
        # Boost for gaps between meetings (likely intentional)
        if gap['type'] == 'between_meetings':
            confidence += 0.1
        
        return min(confidence, 0.7)
    
    def _get_season(self, month: int) -> str:
        """Get season based on month"""
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'fall'
    
    def _get_next_suitable_dates(
        self, 
        preferred_weekdays: List[int], 
        days_ahead: int = 7
    ) -> List[datetime]:
        """Get next suitable dates based on preferred weekdays"""
        
        suitable_dates = []
        current_date = datetime.now() + timedelta(days=1)
        
        for i in range(days_ahead * 2):  # Check next 2 weeks
            check_date = current_date + timedelta(days=i)
            
            if not preferred_weekdays or check_date.weekday() in preferred_weekdays:
                suitable_dates.append(check_date)
                
                if len(suitable_dates) >= 3:  # Limit to 3 dates
                    break
        
        return suitable_dates
    
    def _identify_schedule_gaps(self, bookings: List[Dict]) -> List[Dict]:
        """Identify gaps in user's schedule"""
        
        gaps = []
        
        if not bookings:
            return gaps
        
        # Sort bookings by date and time
        sorted_bookings = sorted(bookings, key=lambda x: (x['date'], x['start_time']))
        
        # Group bookings by date
        bookings_by_date = {}
        for booking in sorted_bookings:
            date = booking['date']
            if date not in bookings_by_date:
                bookings_by_date[date] = []
            bookings_by_date[date].append(booking)
        
        # Find gaps within each day
        for date, day_bookings in bookings_by_date.items():
            day_bookings = sorted(day_bookings, key=lambda x: x['start_time'])
            
            # Check gaps between bookings
            for i in range(len(day_bookings) - 1):
                current_end = day_bookings[i]['end_time']
                next_start = day_bookings[i + 1]['start_time']
                
                gap_minutes = self._calculate_time_difference(current_end, next_start)
                
                if gap_minutes >= 30:  # At least 30 minutes gap
                    gaps.append({
                        'date': date,
                        'start_time': current_end,
                        'end_time': next_start,
                        'duration_minutes': gap_minutes,
                        'type': 'between_meetings',
                        'before_booking': day_bookings[i],
                        'after_booking': day_bookings[i + 1]
                    })
        
        return gaps
    
    def _calculate_time_difference(self, start_time: str, end_time: str) -> int:
        """Calculate difference between two times in minutes"""
        try:
            start = datetime.strptime(start_time, "%H:%M")
            end = datetime.strptime(end_time, "%H:%M")
            return int((end - start).total_seconds() / 60)
        except:
            return 0
    
    def _add_minutes_to_time(self, time_str: str, minutes: int) -> str:
        """Add minutes to a time string"""
        try:
            time_obj = datetime.strptime(time_str, "%H:%M")
            new_time = time_obj + timedelta(minutes=minutes)
            return new_time.strftime("%H:%M")
        except:
            return time_str
    
    async def _check_room_availability(
        self,
        room_name: str,
        date: str,
        start_time: str,
        end_time: str
    ) -> bool:
        """Check if room is available at given time"""
        try:
            # This would integrate with your existing availability logic
            from src.availability_logic import check_availability
            
            return await check_availability(
                room_name=room_name,
                date=date,
                start_time=start_time,
                end_time=end_time
            )
        except Exception as e:
            logger.error(f"Error checking availability: {str(e)}")
            return False
    
    def _filter_and_deduplicate_suggestions(
        self, 
        suggestions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter suggestions by confidence and remove duplicates"""
        
        # Filter by confidence threshold
        filtered = [
            s for s in suggestions 
            if s['confidence_score'] >= self.min_confidence_threshold
        ]
        
        # Remove duplicates based on room, date, and time
        seen = set()
        deduplicated = []
        
        for suggestion in filtered:
            key = (
                suggestion['room_name'],
                suggestion['date'],
                suggestion['start_time'],
                suggestion['end_time']
            )
            
            if key not in seen:
                seen.add(key)
                deduplicated.append(suggestion)
        
        return deduplicated
    
    async def _handle_recent_booking_context(
        self, 
        context: Dict[str, Any], 
        suggestions: List[Dict[str, Any]]
    ):
        """Handle suggestions based on recent booking context"""
        
        if context.get('recent_booking'):
            recent_booking = context['recent_booking']
            
            # Suggest follow-up meeting room for same day
            if recent_booking.get('room_type') == 'presentation':
                end_time = recent_booking.get('end_time')
                if end_time:
                    follow_up_start = self._add_minutes_to_time(end_time, 15)  # 15 min break
                    follow_up_end = self._add_minutes_to_time(follow_up_start, 60)
                    
                    is_available = await self._check_room_availability(
                        'Discussion Room',
                        recent_booking['date'],
                        follow_up_start,
                        follow_up_end
                    )
                    
                    if is_available:
                        suggestions.append({
                            'type': 'context_aware',
                            'room_name': 'Discussion Room',
                            'date': recent_booking['date'],
                            'start_time': follow_up_start,
                            'end_time': follow_up_end,
                            'confidence_score': 0.7,
                            'reason': 'Suggested follow-up discussion after presentation',
                            'metadata': {
                                'context_type': 'follow_up',
                                'parent_booking_id': recent_booking.get('id')
                            }
                        })
    
    async def _handle_time_preference_context(
        self, 
        user_patterns: Dict[str, Any], 
        suggestions: List[Dict[str, Any]]
    ):
        """Handle suggestions based on time preferences"""
        
        optimal_times = await self.analytics.get_optimal_booking_times()
        user_available_times = user_patterns.get('typically_available_times', [])
        
        # Find intersection of popular times and user availability
        for popular_time in optimal_times[:3]:
            if popular_time['time_slot'] in user_available_times:
                next_week = datetime.now() + timedelta(days=7)
                
                is_available = await self._check_room_availability(
                    popular_time['room_name'],
                    next_week.strftime("%Y-%m-%d"),
                    popular_time['start_time'],
                    popular_time['end_time']
                )
                
                if is_available:
                    suggestions.append({
                        'type': 'context_aware',
                        'room_name': popular_time['room_name'],
                        'date': next_week.strftime("%Y-%m-%d"),
                        'start_time': popular_time['start_time'],
                        'end_time': popular_time['end_time'],
                        'confidence_score': 0.65,
                        'reason': 'Optimal time slot with high availability',
                        'metadata': {
                            'context_type': 'optimal_time',
                            'popularity_score': popular_time['popularity']
                        }
                    })
    
    async def _handle_room_preference_context(
        self, 
        user_patterns: Dict[str, Any], 
        suggestions: List[Dict[str, Any]]
    ):
        """Handle suggestions based on room preferences"""
        
        preferred_rooms = user_patterns.get('preferred_rooms', [])[:2]
        preferred_times = user_patterns.get('preferred_times', [])[:2]
        
        for room in preferred_rooms:
            for time_slot in preferred_times:
                # Suggest for tomorrow
                tomorrow = datetime.now() + timedelta(days=1)
                
                is_available = await self._check_room_availability(
                    room,
                    tomorrow.strftime("%Y-%m-%d"),
                    time_slot.get('start_time'),
                    time_slot.get('end_time')
                )
                
                if is_available:
                    suggestions.append({
                        'type': 'context_aware',
                        'room_name': room,
                        'date': tomorrow.strftime("%Y-%m-%d"),
                        'start_time': time_slot.get('start_time'),
                        'end_time': time_slot.get('end_time'),
                        'confidence_score': 0.6,
                        'reason': 'Based on your room and time preferences',
                        'metadata': {
                            'context_type': 'preference_match',
                            'room_preference_score': user_patterns.get('room_scores', {}).get(room, 0.5)
                        }
                    })
    
    async def _handle_workload_context(
        self, 
        context: Dict[str, Any], 
        user_patterns: Dict[str, Any], 
        suggestions: List[Dict[str, Any]]
    ):
        """Handle suggestions based on workload context"""
        
        workload = context.get('workload_level', 'medium')
        
        if workload == 'high':
            # Suggest shorter meetings
            preferred_duration = 30  # 30 minutes for high workload
            reason = 'Shorter meeting suggested due to busy schedule'
        elif workload == 'low':
            # Suggest longer meetings or multiple bookings
            preferred_duration = 120  # 2 hours for low workload
            reason = 'Extended session available due to light schedule'
        else:
            preferred_duration = user_patterns.get('average_duration_minutes', 60)
            reason = 'Standard duration based on your usual patterns'
        
        # Find available slots with preferred duration
        tomorrow = datetime.now() + timedelta(days=1)
        popular_rooms = await self.analytics.get_popular_rooms()
        
        for room in popular_rooms[:2]:
            # Check morning slot
            morning_start = '09:00'
            morning_end = self._add_minutes_to_time(morning_start, preferred_duration)
            
            is_available = await self._check_room_availability(
                room['name'],
                tomorrow.strftime("%Y-%m-%d"),
                morning_start,
                morning_end
            )
            
            if is_available:
                confidence = 0.55 if workload != 'medium' else 0.5
                
                suggestions.append({
                    'type': 'context_aware',
                    'room_name': room['name'],
                    'date': tomorrow.strftime("%Y-%m-%d"),
                    'start_time': morning_start,
                    'end_time': morning_end,
                    'confidence_score': confidence,
                    'reason': reason,
                    'metadata': {
                        'context_type': 'workload_optimization',
                        'workload_level': workload,
                        'duration_minutes': preferred_duration
                    }
                })

    async def generate_alternative_time_suggestions(
        self,
        user_id: str,
        requested_room: str,
        requested_date: str,
        requested_start_time: str,
        requested_end_time: str,
        context: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate alternative time suggestions when requested slot is not available
        """
        try:
            alternatives = []
            user_patterns = await self.analytics.get_user_booking_patterns(user_id)
            
            # Parse requested time
            requested_duration = self._calculate_time_difference(
                requested_start_time, requested_end_time
            )
            date_obj = datetime.strptime(requested_date, "%Y-%m-%d")
            
            # Get room's existing bookings for the day
            existing_bookings = await self.analytics.get_room_bookings_for_date(
                requested_room, requested_date
            )
            
            if not existing_bookings:
                # Room is completely free, suggest optimal times
                alternatives = await self._suggest_optimal_times_for_room(
                    requested_room, requested_date, requested_duration, user_patterns
                )
            else:
                # Find gaps between existing bookings
                alternatives = await self._find_time_gaps_in_schedule(
                    existing_bookings, requested_date, requested_duration, user_patterns
                )
            
            # If same day alternatives are limited, suggest next few days
            if len(alternatives) < 3:
                future_alternatives = await self._suggest_future_date_alternatives(
                    requested_room, date_obj, requested_start_time, 
                    requested_end_time, user_patterns
                )
                alternatives.extend(future_alternatives)
            
            # Sort by confidence and user preference alignment
            alternatives.sort(key=lambda x: x['confidence_score'], reverse=True)
            
            return alternatives[:5]  # Return top 5 alternatives
            
        except Exception as e:
            logger.error(f"Error generating alternative time suggestions: {str(e)}")
            return []

    async def _suggest_optimal_times_for_room(
        self,
        room_name: str,
        date: str,
        duration_minutes: int,
        user_patterns: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Suggest optimal times when room is completely free"""
        
        alternatives = []
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        
        # Get user's preferred times
        user_preferred_times = user_patterns.get('preferred_times', [])
        
        # Business hours
        business_start = datetime.strptime(self.business_hours['start'], "%H:%M").time()
        business_end = datetime.strptime(self.business_hours['end'], "%H:%M").time()
        
        # If user has preferred times, prioritize those
        for pref_time in user_preferred_times[:3]:
            start_time = pref_time.get('start_time')
            if start_time:
                start_dt = datetime.strptime(start_time, "%H:%M").time()
                
                # Check if preferred time fits in business hours
                end_dt = (datetime.combine(date_obj, start_dt) + 
                         timedelta(minutes=duration_minutes)).time()
                
                if business_start <= start_dt and end_dt <= business_end:
                    confidence = self._calculate_time_confidence(start_time, user_patterns)
                    
                    alternatives.append({
                        'room_name': room_name,
                        'date': date,
                        'start_time': start_time,
                        'end_time': self._add_minutes_to_time(start_time, duration_minutes),
                        'confidence_score': confidence,
                        'reason': 'Matches your preferred booking time',
                        'metadata': {
                            'preference_match': True,
                            'optimal_slot': True
                        }
                    })
        
        # Add some general optimal times if we don't have enough suggestions
        optimal_start_times = ['09:00', '10:00', '14:00', '15:00', '16:00']
        
        for start_time in optimal_start_times:
            if len(alternatives) >= 5:
                break
                
            # Skip if we already suggested this time
            if any(alt['start_time'] == start_time for alt in alternatives):
                continue
            
            start_dt = datetime.strptime(start_time, "%H:%M").time()
            end_dt = (datetime.combine(date_obj, start_dt) + 
                     timedelta(minutes=duration_minutes)).time()
            
            if business_start <= start_dt and end_dt <= business_end:
                confidence = self._calculate_time_confidence(start_time, user_patterns)
                
                alternatives.append({
                    'room_name': room_name,
                    'date': date,
                    'start_time': start_time,
                    'end_time': self._add_minutes_to_time(start_time, duration_minutes),
                    'confidence_score': confidence,
                    'reason': 'Optimal time slot for productivity',
                    'metadata': {
                        'optimal_slot': True,
                        'general_recommendation': True
                    }
                })
        
        return alternatives

    async def _find_time_gaps_in_schedule(
        self,
        existing_bookings: List[Dict],
        date: str,
        duration_minutes: int,
        user_patterns: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Find available time gaps between existing bookings"""
        
        alternatives = []
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        
        # Sort bookings by start time
        sorted_bookings = sorted(existing_bookings, 
                               key=lambda x: datetime.strptime(x['start_time'], "%H:%M"))
        
        business_start = datetime.strptime(self.business_hours['start'], "%H:%M")
        business_end = datetime.strptime(self.business_hours['end'], "%H:%M")
        duration = timedelta(minutes=duration_minutes)
        
        # If no bookings, return optimal times
        if not sorted_bookings:
            current = business_start
            while current + duration <= business_end:
                confidence = self._calculate_time_confidence(
                    current.strftime("%H:%M"), user_patterns
                )
                alternatives.append({
                    'room_name': existing_bookings[0]['room_name'] if existing_bookings else 'Room',
                    'date': date,
                    'start_time': current.strftime("%H:%M"),
                    'end_time': (current + duration).strftime("%H:%M"),
                    'confidence_score': confidence,
                    'reason': 'Available time slot with no conflicts',
                    'metadata': {
                        'gap_type': 'open_schedule'
                    }
                })
                current += timedelta(minutes=30)  # Check every 30 minutes
            return alternatives
        
        # Check gap before first booking
        first_booking_start = datetime.strptime(sorted_bookings[0]['start_time'], "%H:%M")
        if business_start + duration <= first_booking_start:
            current = business_start
            while current + duration <= first_booking_start:
                confidence = self._calculate_time_confidence(
                    current.strftime("%H:%M"), user_patterns
                )
                alternatives.append({
                    'room_name': sorted_bookings[0]['room_name'],
                    'date': date,
                    'start_time': current.strftime("%H:%M"),
                    'end_time': (current + duration).strftime("%H:%M"),
                    'confidence_score': confidence,
                    'reason': 'Available before existing bookings',
                    'metadata': {
                        'gap_type': 'before_first'
                    }
                })
                current += timedelta(minutes=30)
        
        # Check gaps between bookings
        for i in range(len(sorted_bookings) - 1):
            gap_start = datetime.strptime(sorted_bookings[i]['end_time'], "%H:%M")
            gap_end = datetime.strptime(sorted_bookings[i + 1]['start_time'], "%H:%M")
            
            if gap_end - gap_start >= duration:
                current = gap_start
                while current + duration <= gap_end:
                    confidence = self._calculate_time_confidence(
                        current.strftime("%H:%M"), user_patterns
                    )
                    alternatives.append({
                        'room_name': sorted_bookings[i]['room_name'],
                        'date': date,
                        'start_time': current.strftime("%H:%M"),
                        'end_time': (current + duration).strftime("%H:%M"),
                        'confidence_score': confidence,
                        'reason': 'Available between existing bookings',
                        'metadata': {
                            'gap_type': 'between_bookings',
                            'before_booking': sorted_bookings[i]['id'],
                            'after_booking': sorted_bookings[i + 1]['id']
                        }
                    })
                    current += timedelta(minutes=30)
        
        # Check gap after last booking
        last_booking_end = datetime.strptime(sorted_bookings[-1]['end_time'], "%H:%M")
        if last_booking_end + duration <= business_end:
            current = last_booking_end
            while current + duration <= business_end:
                confidence = self._calculate_time_confidence(
                    current.strftime("%H:%M"), user_patterns
                )
                alternatives.append({
                    'room_name': sorted_bookings[-1]['room_name'],
                    'date': date,
                    'start_time': current.strftime("%H:%M"),
                    'end_time': (current + duration).strftime("%H:%M"),
                    'confidence_score': confidence,
                    'reason': 'Available after existing bookings',
                    'metadata': {
                        'gap_type': 'after_last'
                    }
                })
                current += timedelta(minutes=30)
        
        return alternatives

    async def _suggest_future_date_alternatives(
        self,
        room_name: str,
        original_date: datetime,
        start_time: str,
        end_time: str,
        user_patterns: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Suggest alternatives on future dates"""
        
        alternatives = []
        
        # Check next 7 days
        for days_ahead in range(1, 8):
            future_date = original_date + timedelta(days=days_ahead)
            future_date_str = future_date.strftime("%Y-%m-%d")
            
            # Skip weekends if user typically doesn't book on weekends
            if future_date.weekday() >= 5:  # Saturday or Sunday
                weekend_bookings = user_patterns.get('weekend_bookings', 0)
                if weekend_bookings < 0.1:  # Less than 10% weekend bookings
                    continue
            
            # Check if room is available at the same time
            is_available = await self._check_room_availability(
                room_name, future_date_str, start_time, end_time
            )
            
            if is_available:
                # Calculate confidence based on day preference
                day_preference = self._get_day_preference(
                    future_date.weekday(), user_patterns
                )
                base_confidence = 0.7
                confidence = base_confidence * day_preference
                
                # Reduce confidence slightly for future dates
                confidence *= (1 - (days_ahead * 0.05))  # 5% reduction per day
                
                alternatives.append({
                    'room_name': room_name,
                    'date': future_date_str,
                    'start_time': start_time,
                    'end_time': end_time,
                    'confidence_score': max(confidence, 0.3),
                    'reason': f'Same time slot available {days_ahead} day{"s" if days_ahead > 1 else ""} later',
                    'metadata': {
                        'days_ahead': days_ahead,
                        'same_time_slot': True,
                        'day_preference': day_preference
                    }
                })
                
                if len(alternatives) >= 3:
                    break
        
        return alternatives

    def _calculate_time_confidence(
        self, 
        time_slot: str, 
        user_patterns: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for a time slot based on user patterns"""
        
        base_confidence = 0.5
        
        # Check if this time matches user's preferred times
        preferred_times = user_patterns.get('preferred_times', [])
        for pref_time in preferred_times:
            if pref_time.get('start_time') == time_slot:
                frequency = pref_time.get('frequency', 1)
                base_confidence += min(frequency * 0.1, 0.3)
                break
        
        # Boost for optimal productivity hours (9-11 AM, 2-4 PM)
        hour = int(time_slot.split(':')[0])
        if 9 <= hour <= 11 or 14 <= hour <= 16:
            base_confidence += 0.1
        
        # Reduce confidence for very early or late hours
        if hour < 8 or hour > 18:
            base_confidence -= 0.2
        
        return max(min(base_confidence, 1.0), 0.1)

    def _get_day_preference(
        self, 
        weekday: int, 
        user_patterns: Dict[str, Any]
    ) -> float:
        """Get user's preference score for a specific weekday"""
        
        weekday_patterns = user_patterns.get('weekday_patterns', {})
        day_name = ['monday', 'tuesday', 'wednesday', 'thursday', 
                   'friday', 'saturday', 'sunday'][weekday]
        
        day_pattern = weekday_patterns.get(day_name, {})
        frequency = day_pattern.get('frequency', 0.2)  # Default low frequency
        
        # Normalize frequency to preference score (0.1 to 1.0)
        preference = min(max(frequency * 2, 0.1), 1.0)
        
        return preference

    async def get_proactive_suggestions_summary(
        self, 
        user_id: str
    ) -> Dict[str, Any]:
        """Get a summary of proactive suggestions and their performance"""
        
        try:
            # Get user patterns for context
            user_patterns = await self.analytics.get_user_booking_patterns(user_id)
            
            # Get recent suggestions performance
            suggestion_metrics = await self.metrics.get_user_suggestion_metrics(user_id)
            
            # Calculate next likely booking time
            next_likely_booking = await self._predict_next_booking_time(user_patterns)
            
            # Get trending rooms and times
            trending_data = await self.analytics.get_trending_booking_data()
            
            summary = {
                'user_id': user_id,
                'summary_generated_at': datetime.now().isoformat(),
                'booking_patterns': {
                    'total_bookings': user_patterns.get('total_bookings', 0),
                    'recurring_patterns': len(user_patterns.get('recurring_patterns', [])),
                    'preferred_rooms': user_patterns.get('preferred_rooms', [])[:3],
                    'preferred_times': user_patterns.get('preferred_times', [])[:3],
                    'average_duration': user_patterns.get('average_duration_minutes', 60)
                },
                'suggestions_performance': {
                    'total_suggestions_made': suggestion_metrics.get('total_suggestions', 0),
                    'suggestions_accepted': suggestion_metrics.get('accepted_suggestions', 0),
                    'acceptance_rate': suggestion_metrics.get('acceptance_rate', 0.0),
                    'most_successful_strategy': suggestion_metrics.get('best_strategy', 'recurring')
                },
                'predictions': {
                    'next_likely_booking': next_likely_booking,
                    'confidence_level': next_likely_booking.get('confidence', 0.0) if next_likely_booking else 0.0
                },
                'trending_insights': {
                    'popular_rooms_this_week': trending_data.get('popular_rooms', [])[:3],
                    'peak_booking_times': trending_data.get('peak_times', [])[:3],
                    'emerging_patterns': trending_data.get('emerging_patterns', [])
                },
                'recommendations': {
                    'should_enable_proactive': suggestion_metrics.get('acceptance_rate', 0) > 0.3,
                    'optimal_notification_time': self._calculate_optimal_notification_time(user_patterns),
                    'suggested_improvements': self._generate_usage_suggestions(user_patterns, suggestion_metrics)
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating proactive suggestions summary: {str(e)}")
            return {
                'user_id': user_id,
                'error': 'Unable to generate summary',
                'summary_generated_at': datetime.now().isoformat()
            }

    async def _predict_next_booking_time(
        self, 
        user_patterns: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Predict when user is most likely to make their next booking"""
        
        recurring_patterns = user_patterns.get('recurring_patterns', [])
        
        if not recurring_patterns:
            return None
        
        # Find the most consistent recurring pattern
        best_pattern = max(recurring_patterns, 
                          key=lambda p: p.get('consistency_score', 0))
        
        if best_pattern.get('consistency_score', 0) < 0.3:
            return None
        
        try:
            last_booking = datetime.strptime(
                best_pattern['last_booking_date'], "%Y-%m-%d"
            )
            frequency = best_pattern['frequency']
            
            next_dates = self._calculate_next_occurrences(last_booking, frequency, 1)
            
            if next_dates and next_dates[0] > datetime.now():
                return {
                    'predicted_date': next_dates[0].strftime("%Y-%m-%d"),
                    'predicted_time': best_pattern.get('start_time'),
                    'room_name': best_pattern.get('room_name'),
                    'confidence': best_pattern.get('consistency_score', 0.5),
                    'based_on_pattern': frequency
                }
        
        except (ValueError, KeyError):
            pass
        
        return None

    def _calculate_optimal_notification_time(
        self, 
        user_patterns: Dict[str, Any]
    ) -> str:
        """Calculate optimal time to send proactive notifications"""
        
        # Analyze when user typically makes bookings
        booking_times = user_patterns.get('booking_creation_times', [])
        
        if not booking_times:
            return '09:00'  # Default morning time
        
        # Find the most common hour for making bookings
        hours = [int(time.split(':')[0]) for time in booking_times if ':' in time]
        
        if hours:
            most_common_hour = max(set(hours), key=hours.count)
            return f"{most_common_hour:02d}:00"
        
        return '09:00'

    def _generate_usage_suggestions(
        self, 
        user_patterns: Dict[str, Any], 
        suggestion_metrics: Dict[str, Any]
    ) -> List[str]:
        """Generate suggestions to improve booking experience"""
        
        suggestions = []
        
        # Check booking frequency
        total_bookings = user_patterns.get('total_bookings', 0)
        if total_bookings < 5:
            suggestions.append(
                "Try booking rooms more regularly to help us learn your preferences"
            )
        
        # Check pattern consistency
        recurring_patterns = user_patterns.get('recurring_patterns', [])
        if not recurring_patterns:
            suggestions.append(
                "Consider establishing regular meeting schedules for better predictions"
            )
        
        # Check acceptance rate
        acceptance_rate = suggestion_metrics.get('acceptance_rate', 0)
        if acceptance_rate < 0.2:
            suggestions.append(
                "Let us know your feedback on suggestions to improve recommendations"
            )
        
        # Check room variety
        preferred_rooms = user_patterns.get('preferred_rooms', [])
        if len(preferred_rooms) < 2:
            suggestions.append(
                "Try different room types to discover new preferences"
            )
        
        # Check time variety
        preferred_times = user_patterns.get('preferred_times', [])
        if len(preferred_times) < 2:
            suggestions.append(
                "Booking at different times can help optimize your schedule"
            )
        
        return suggestions[:3]  # Return top 3 suggestions

    async def cleanup_old_suggestions(self, days_old: int = 30) -> int:
        """Clean up old suggestion records"""
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            # This would integrate with your database cleanup logic
            deleted_count = await self.analytics.cleanup_old_suggestions(cutoff_date)
            
            logger.info(f"Cleaned up {deleted_count} old suggestion records")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up old suggestions: {str(e)}")
            return 0

    def get_strategy_weights(self) -> Dict[str, float]:
        """Get current strategy weights for recommendation mixing"""
        
        return {
            'recurring': 0.35,      # Highest weight for established patterns
            'seasonal': 0.20,       # Medium weight for seasonal trends
            'collaborative': 0.15,  # Lower weight due to privacy concerns
            'context_aware': 0.20,  # Medium weight for contextual suggestions
            'gap_filling': 0.10     # Lowest weight for schedule optimization
        }

    async def update_strategy_weights(
        self, 
        performance_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """Update strategy weights based on performance"""
        
        try:
            current_weights = self.get_strategy_weights()
            
            # Adjust weights based on acceptance rates
            for strategy, acceptance_rate in performance_metrics.items():
                if strategy in current_weights:
                    # Increase weight for well-performing strategies
                    if acceptance_rate > 0.5:
                        current_weights[strategy] *= 1.1
                    # Decrease weight for poor-performing strategies
                    elif acceptance_rate < 0.2:
                        current_weights[strategy] *= 0.9
            
            # Normalize weights to sum to 1.0
            total_weight = sum(current_weights.values())
            normalized_weights = {
                strategy: weight / total_weight 
                for strategy, weight in current_weights.items()
            }
            
            return normalized_weights
            
        except Exception as e:
            logger.error(f"Error updating strategy weights: {str(e)}")
            return self.get_strategy_weights()  # Return default weights