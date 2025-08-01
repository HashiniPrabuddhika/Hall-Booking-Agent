import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, Counter
import numpy as np
from dataclasses import dataclass
from enum import Enum

from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_

try:
    from ..models.embedding_model import EmbeddingModel
except ImportError:
    EmbeddingModel = None

try:
    from ..utils.time_utils import TimeUtils
except ImportError:
    class TimeUtils:
        pass

try:
    from ..utils.metrics import RecommendationMetrics
except ImportError:
    class RecommendationMetrics:
        pass

try:
    from ..data.cache_manager import CacheManager
except ImportError:
    class CacheManager:
        def get(self, key): return None
        def set(self, key, value, ttl=None): pass

try:
    from src.models import MRBSEntry, MRBSRoom, MRBSRepeat
except ImportError:
    
    class MRBSEntry:
        def __init__(self):
            self.create_by = None
            self.timestamp = datetime.now()
            self.start_time = 0
            self.end_time = 0
            self.room_id = None
            self.room = None
            self.repeat_id = None
    
    class MRBSRoom:
        def __init__(self):
            self.id = None
            self.capacity = 10
            self.description = ""
    
    class MRBSRepeat:
        def __init__(self):
            self.id = None

logger = logging.getLogger(__name__)


class PreferenceType(Enum):
    """Types of user preferences"""
    TIME_SLOT = "time_slot"
    ROOM_TYPE = "room_type"
    DURATION = "duration"
    CAPACITY = "capacity"
    FEATURES = "features"
    LOCATION = "location"
    RECURRENCE = "recurrence"


class LearningStrategy(Enum):
    """Learning strategies for preference adaptation"""
    IMPLICIT = "implicit"  # Learn from behavior
    EXPLICIT = "explicit"  # Learn from feedback
    HYBRID = "hybrid"     # Combine both approaches


@dataclass
class PreferenceScore:
    """Represents a preference score for a specific attribute"""
    attribute: str
    value: Any
    score: float
    confidence: float
    last_updated: datetime
    source: str  # 'booking', 'feedback', 'interaction'


@dataclass
class UserProfile:
    """Complete user profile with preferences and patterns"""
    user_id: str
    preferences: Dict[PreferenceType, List[PreferenceScore]]
    booking_patterns: Dict[str, Any]
    interaction_history: List[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime


class PreferenceLearner:
    """
    Main class for learning and managing user preferences
    """
    from ..models.embedding_model import EmbeddingModel
    def __init__(self, db: Session, embedding_model: Optional[EmbeddingModel] = None, 
                 cache_manager: Optional[CacheManager] = None):
        self.db = db
        self.embedding_model = embedding_model or (EmbeddingModel() if EmbeddingModel else None)
        self.cache_manager = cache_manager or CacheManager()
        self.time_utils = TimeUtils()
        self.metrics = RecommendationMetrics()
        
        # Learning parameters
        self.decay_factor = 0.95  # For time-based preference decay
        self.min_interactions = 5  # Minimum interactions to establish preference
        self.confidence_threshold = 0.6  # Minimum confidence for strong preferences
        
        # Check if database tables exist
        self.tables_exist = self._check_tables_exist()
        
    def _check_tables_exist(self) -> bool:
        """Check if required database tables exist"""
        try:
            self.db.query(MRBSEntry).limit(1).all()
            return True
        except Exception as e:
            logger.warning(f"Database tables not available, running in mock mode: {e}")
            return False
        
    def learn_user_preferences(self, user_id: str, 
                             strategy: LearningStrategy = LearningStrategy.HYBRID) -> UserProfile:
        """
        Learn comprehensive user preferences using specified strategy
        
        Args:
            user_id: User identifier
            strategy: Learning strategy to use
            
        Returns:
            Complete user profile with learned preferences
        """
        try:
            cached_profile = self._get_cached_profile(user_id)
            if cached_profile and self._is_profile_fresh(cached_profile):
                return cached_profile
            
            if not self.tables_exist:
                return self._get_mock_profile(user_id)
            
            # Learn preferences based on strategy
            if strategy == LearningStrategy.IMPLICIT:
                preferences = self._learn_implicit_preferences(user_id)
            elif strategy == LearningStrategy.EXPLICIT:
                preferences = self._learn_explicit_preferences(user_id)
            else:  # HYBRID
                implicit_prefs = self._learn_implicit_preferences(user_id)
                explicit_prefs = self._learn_explicit_preferences(user_id)
                preferences = self._merge_preferences(implicit_prefs, explicit_prefs)
            
            # Extract booking patterns
            patterns = self._extract_booking_patterns(user_id)
            
            # Get interaction history
            interactions = self._get_interaction_history(user_id)
            
            # Create user profile
            profile = UserProfile(
                user_id=user_id,
                preferences=preferences,
                booking_patterns=patterns,
                interaction_history=interactions,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            # Cache the profile
            self._cache_profile(profile)
            
            # Update database
            self._update_user_preferences_db(profile)
            
            logger.info(f"Learned preferences for user {user_id} using {strategy.value} strategy")
            return profile
            
        except Exception as e:
            logger.error(f"Error learning preferences for user {user_id}: {str(e)}")
            return self._get_default_profile(user_id)
    
    def _get_mock_profile(self, user_id: str) -> UserProfile:
        """Get a mock profile with sample preferences for testing"""
        mock_preferences = {
            PreferenceType.TIME_SLOT: [
                PreferenceScore(
                    attribute="preferred_hour",
                    value=10,
                    score=0.8,
                    confidence=0.7,
                    last_updated=datetime.now(),
                    source="mock"
                ),
                PreferenceScore(
                    attribute="preferred_day_of_week",
                    value=1,  # Tuesday
                    score=0.6,
                    confidence=0.6,
                    last_updated=datetime.now(),
                    source="mock"
                )
            ],
            PreferenceType.ROOM_TYPE: [
                PreferenceScore(
                    attribute="room_size_category",
                    value="medium",
                    score=0.7,
                    confidence=0.8,
                    last_updated=datetime.now(),
                    source="mock"
                )
            ],
            PreferenceType.DURATION: [
                PreferenceScore(
                    attribute="preferred_duration",
                    value=2.0,  # 2 hours
                    score=0.9,
                    confidence=0.8,
                    last_updated=datetime.now(),
                    source="mock"
                )
            ]
        }
        
        mock_patterns = {
            'total_bookings': 15,
            'booking_frequency': {'per_week': 2.5, 'per_month': 10},
            'peak_usage_times': {'peak_hours': [10, 14, 16], 'peak_days': [1, 2, 3]},
            'typical_advance_booking': 24.0,
            'cancellation_rate': 0.1,
            'preferred_booking_duration': 2.0
        }
        
        return UserProfile(
            user_id=user_id,
            preferences=mock_preferences,
            booking_patterns=mock_patterns,
            interaction_history=[],
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    
    def _learn_implicit_preferences(self, user_id: str) -> Dict[PreferenceType, List[PreferenceScore]]:
        """Learn preferences from user booking behavior"""
        preferences = defaultdict(list)
        
        if not self.tables_exist:
            return dict(preferences)
        
        try:
            # Get user's booking history
            bookings = self.db.query(MRBSEntry).filter(
                MRBSEntry.create_by == user_id
            ).order_by(MRBSEntry.timestamp.desc()).limit(500).all()
            
            if len(bookings) < self.min_interactions:
                logger.info(f"Insufficient booking history for user {user_id}")
                return dict(preferences)
            
            # Learn time slot preferences
            self._learn_time_preferences(bookings, preferences)
            
            self._learn_room_preferences(bookings, preferences)
            
            self._learn_duration_preferences(bookings, preferences)
            
            self._learn_capacity_preferences(bookings, preferences)
            
            self._learn_feature_preferences(bookings, preferences)
            
            self._learn_recurrence_preferences(bookings, preferences)
            
        except Exception as e:
            logger.error(f"Error learning implicit preferences: {e}")
        
        return dict(preferences)
    
    def _learn_explicit_preferences(self, user_id: str) -> Dict[PreferenceType, List[PreferenceScore]]:
        """Learn preferences from explicit user feedback - simplified for current models"""
        preferences = defaultdict(list)
        
        # Since we don't have RecommendationFeedback and UserInteraction tables yet,
        # we'll return empty preferences for now
        # This can be implemented when those tables are added later
        
        logger.info(f"Explicit preference learning not available without feedback tables for user {user_id}")
        return dict(preferences)
    
    def _learn_time_preferences(self, bookings: List[MRBSEntry], 
                               preferences: Dict[PreferenceType, List[PreferenceScore]]):
        """Learn preferred time slots from booking history"""
        time_slots = []
        
        for booking in bookings:
            try:
                start_time = datetime.fromtimestamp(booking.start_time)
                hour = start_time.hour
                day_of_week = start_time.weekday()
                
                time_slots.append({
                    'hour': hour,
                    'day_of_week': day_of_week,
                    'timestamp': booking.timestamp
                })
            except (AttributeError, ValueError) as e:
                logger.warning(f"Invalid booking time data: {e}")
                continue
        
        if not time_slots:
            return
        
        # Analyze hourly preferences
        hour_counts = Counter([slot['hour'] for slot in time_slots])
        total_bookings = len(time_slots)
        
        for hour, count in hour_counts.most_common():
            score = count / total_bookings
            confidence = min(score * 2, 1.0)  
            
            if score > 0.1:  # Only include significant preferences
                preferences[PreferenceType.TIME_SLOT].append(
                    PreferenceScore(
                        attribute="preferred_hour",
                        value=hour,
                        score=score,
                        confidence=confidence,
                        last_updated=datetime.now(),
                        source="booking"
                    )
                )
        
        # Analyze day-of-week preferences
        dow_counts = Counter([slot['day_of_week'] for slot in time_slots])
        
        for dow, count in dow_counts.most_common():
            score = count / total_bookings
            confidence = min(score * 2, 1.0)
            
            if score > 0.1:
                preferences[PreferenceType.TIME_SLOT].append(
                    PreferenceScore(
                        attribute="preferred_day_of_week",
                        value=dow,
                        score=score,
                        confidence=confidence,
                        last_updated=datetime.now(),
                        source="booking"
                    )
                )
    
    def _learn_room_preferences(self, bookings: List[MRBSEntry], 
                               preferences: Dict[PreferenceType, List[PreferenceScore]]):
        """Learn room type and specific room preferences"""
        room_usage = defaultdict(int)
        room_types = defaultdict(int)
        
        for booking in bookings:
            try:
                room = booking.room
                if room:
                    room_usage[room.id] += 1
                    
                    capacity = getattr(room, 'capacity', 10)
                    if capacity <= 5:
                        room_type = "small"
                    elif capacity <= 15:
                        room_type = "medium"
                    else:
                        room_type = "large"
                    
                    room_types[room_type] += 1
            except AttributeError:
                continue
        
        total_bookings = len(bookings)
        
        if total_bookings == 0:
            return
        
        # Room type preferences
        for room_type, count in room_types.items():
            score = count / total_bookings
            confidence = min(score * 1.5, 1.0)
            
            if score > 0.15:
                preferences[PreferenceType.ROOM_TYPE].append(
                    PreferenceScore(
                        attribute="room_size_category",
                        value=room_type,
                        score=score,
                        confidence=confidence,
                        last_updated=datetime.now(),
                        source="booking"
                    )
                )
        
        # Specific room preferences
        for room_id, count in room_usage.items():
            score = count / total_bookings
            confidence = min(score * 3, 1.0)  # Higher confidence for specific rooms
            
            if score > 0.2:  # Higher threshold for specific rooms
                preferences[PreferenceType.ROOM_TYPE].append(
                    PreferenceScore(
                        attribute="preferred_room_id",
                        value=room_id,
                        score=score,
                        confidence=confidence,
                        last_updated=datetime.now(),
                        source="booking"
                    )
                )
    
    def _learn_duration_preferences(self, bookings: List[MRBSEntry], 
                                   preferences: Dict[PreferenceType, List[PreferenceScore]]):
        """Learn preferred booking durations"""
        durations = []
        
        for booking in bookings:
            try:
                duration = (booking.end_time - booking.start_time) / 3600  # Duration in hours
                if duration > 0:
                    durations.append(duration)
            except (AttributeError, TypeError):
                continue
        
        if not durations:
            return
        
        # Calculate common duration patterns
        duration_counts = Counter([round(d * 2) / 2 for d in durations])  # Round to nearest 0.5 hour
        total_bookings = len(durations)
        avg_duration = np.mean(durations)
        
        for duration, count in duration_counts.most_common():
            score = count / total_bookings
            confidence = min(score * 2, 1.0)
            
            if score > 0.1 and duration > 0:
                preferences[PreferenceType.DURATION].append(
                    PreferenceScore(
                        attribute="preferred_duration",
                        value=duration,
                        score=score,
                        confidence=confidence,
                        last_updated=datetime.now(),
                        source="booking"
                    )
                )
        
        # Add average duration preference
        preferences[PreferenceType.DURATION].append(
            PreferenceScore(
                attribute="average_duration",
                value=avg_duration,
                score=1.0,
                confidence=0.8,
                last_updated=datetime.now(),
                source="booking"
            )
        )
    
    def _learn_capacity_preferences(self, bookings: List[MRBSEntry], 
                                   preferences: Dict[PreferenceType, List[PreferenceScore]]):
        """Learn preferred room capacities"""
        capacities = []
        
        for booking in bookings:
            try:
                if booking.room and hasattr(booking.room, 'capacity') and booking.room.capacity:
                    capacities.append(booking.room.capacity)
            except AttributeError:
                continue
        
        if not capacities:
            return
        
        avg_capacity = np.mean(capacities)
        
        # Capacity range preferences
        capacity_ranges = {
            "small": (0, 10),
            "medium": (11, 25),
            "large": (26, 50),
            "extra_large": (51, float('inf'))
        }
        
        range_counts = defaultdict(int)
        for capacity in capacities:
            for range_name, (min_cap, max_cap) in capacity_ranges.items():
                if min_cap <= capacity <= max_cap:
                    range_counts[range_name] += 1
                    break
        
        total_bookings = len(capacities)
        
        for range_name, count in range_counts.items():
            score = count / total_bookings
            confidence = min(score * 1.5, 1.0)
            
            if score > 0.15:
                preferences[PreferenceType.CAPACITY].append(
                    PreferenceScore(
                        attribute="capacity_range",
                        value=range_name,
                        score=score,
                        confidence=confidence,
                        last_updated=datetime.now(),
                        source="booking"
                    )
                )
        
        # Average capacity preference
        preferences[PreferenceType.CAPACITY].append(
            PreferenceScore(
                attribute="average_capacity",
                value=avg_capacity,
                score=1.0,
                confidence=0.7,
                last_updated=datetime.now(),
                source="booking"
            )
        )
    
    def _learn_feature_preferences(self, bookings: List[MRBSEntry], 
                                  preferences: Dict[PreferenceType, List[PreferenceScore]]):
        """Learn preferences for room features (using room descriptions and custom_html)"""
        # Since we don't have a separate RoomFeature table, we'll analyze room descriptions
        room_features = {}
        
        for booking in bookings:
            try:
                if booking.room and hasattr(booking.room, 'description') and booking.room.description:
                    # Extract features from room description
                    description = booking.room.description.lower()
                    features = []
                    
                    # Common room features to look for
                    feature_keywords = {
                        'projector': ['projector', 'projection'],
                        'whiteboard': ['whiteboard', 'board'],
                        'tv': ['tv', 'television', 'screen'],
                        'ac': ['ac', 'air conditioning', 'aircon'],
                        'wifi': ['wifi', 'wireless'],
                        'video_conference': ['video', 'conference', 'zoom', 'teams'],
                        'phone': ['phone', 'telephone']
                    }
                    
                    for feature, keywords in feature_keywords.items():
                        if any(keyword in description for keyword in keywords):
                            features.append(feature)
                    
                    if features:
                        room_features[booking.room_id] = features
            except AttributeError:
                continue
        
        # Count feature usage
        feature_counts = defaultdict(int)
        total_bookings = len(bookings)
        
        for room_id, features in room_features.items():
            bookings_with_room = sum(1 for b in bookings if getattr(b, 'room_id', None) == room_id)
            for feature in features:
                feature_counts[feature] += bookings_with_room
        
        # Create preference scores
        for feature_name, count in feature_counts.items():
            if total_bookings > 0:
                score = count / total_bookings
                confidence = min(score * 1.5, 1.0)
                
                if score > 0.1:
                    preferences[PreferenceType.FEATURES].append(
                        PreferenceScore(
                            attribute="room_feature",
                            value=feature_name,
                            score=score,
                            confidence=confidence,
                            last_updated=datetime.now(),
                            source="booking"
                        )
                    )
    
    def _learn_recurrence_preferences(self, bookings: List[MRBSEntry], 
                                     preferences: Dict[PreferenceType, List[PreferenceScore]]):
        """Learn recurrence pattern preferences"""
        try:
            recurring_bookings = [b for b in bookings if getattr(b, 'repeat_id', None) is not None]
            one_time_bookings = [b for b in bookings if getattr(b, 'repeat_id', None) is None]
            
            total_bookings = len(bookings)
            
            if total_bookings == 0:
                return
            
            # Recurrence type preference
            recurring_ratio = len(recurring_bookings) / total_bookings
            one_time_ratio = len(one_time_bookings) / total_bookings
            
            if recurring_ratio > 0.1:
                preferences[PreferenceType.RECURRENCE].append(
                    PreferenceScore(
                        attribute="booking_type",
                        value="recurring",
                        score=recurring_ratio,
                        confidence=min(recurring_ratio * 2, 1.0),
                        last_updated=datetime.now(),
                        source="booking"
                    )
                )
            
            if one_time_ratio > 0.1:
                preferences[PreferenceType.RECURRENCE].append(
                    PreferenceScore(
                        attribute="booking_type",
                        value="one_time",
                        score=one_time_ratio,
                        confidence=min(one_time_ratio * 2, 1.0),
                        last_updated=datetime.now(),
                        source="booking"
                    )
                )
        except Exception as e:
            logger.warning(f"Error learning recurrence preferences: {e}")
    
    def _merge_preferences(self, implicit_prefs: Dict, explicit_prefs: Dict) -> Dict:
        """Merge implicit and explicit preferences with appropriate weighting"""
        merged = defaultdict(list)
        
        # Weight factors
        implicit_weight = 0.7
        explicit_weight = 0.3
        
        # Merge preferences for each type
        all_types = set(implicit_prefs.keys()) | set(explicit_prefs.keys())
        
        for pref_type in all_types:
            implicit_scores = implicit_prefs.get(pref_type, [])
            explicit_scores = explicit_prefs.get(pref_type, [])
            
            # Add weighted implicit preferences
            for score in implicit_scores:
                weighted_score = PreferenceScore(
                    attribute=score.attribute,
                    value=score.value,
                    score=score.score * implicit_weight,
                    confidence=score.confidence * implicit_weight,
                    last_updated=score.last_updated,
                    source=f"implicit_{score.source}"
                )
                merged[pref_type].append(weighted_score)
            
            # Add weighted explicit preferences
            for score in explicit_scores:
                weighted_score = PreferenceScore(
                    attribute=score.attribute,
                    value=score.value,
                    score=score.score * explicit_weight,
                    confidence=score.confidence * explicit_weight,
                    last_updated=score.last_updated,
                    source=f"explicit_{score.source}"
                )
                merged[pref_type].append(weighted_score)
        
        return dict(merged)
    
    def update_preferences_from_feedback(self, user_id: str, feedback: Dict[str, Any]):
        """Update user preferences based on new feedback"""
        try:
            # Get current profile
            profile = self.learn_user_preferences(user_id)
            
            # Process new feedback
            new_preferences = self._process_new_feedback(feedback)
            
            # Update preferences with feedback
            for pref_type, new_scores in new_preferences.items():
                if pref_type in profile.preferences:
                    profile.preferences[pref_type].extend(new_scores)
                else:
                    profile.preferences[pref_type] = new_scores
            
            profile.updated_at = datetime.now()
            
            # Update cache and database
            self._cache_profile(profile)
            self._update_user_preferences_db(profile)
            
            logger.info(f"Updated preferences for user {user_id} based on feedback")
            
        except Exception as e:
            logger.error(f"Error updating preferences from feedback: {str(e)}")
    
    def get_preference_strength(self, user_id: str, preference_type: PreferenceType, 
                               attribute: str, value: Any) -> float:
        """Get the strength of a specific preference for a user"""
        try:
            profile = self.learn_user_preferences(user_id)
            
            if preference_type not in profile.preferences:
                return 0.0
            
            for score in profile.preferences[preference_type]:
                if score.attribute == attribute and score.value == value:
                    return score.score * score.confidence
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error getting preference strength: {str(e)}")
            return 0.0
    
    def _extract_booking_patterns(self, user_id: str) -> Dict[str, Any]:
        """Extract high-level booking patterns"""
        if not self.tables_exist:
            return {
                'total_bookings': 5,
                'booking_frequency': {'per_week': 1.5, 'per_month': 6},
                'peak_usage_times': {'peak_hours': [10, 14], 'peak_days': [1, 2]},
                'typical_advance_booking': 24.0,
                'cancellation_rate': 0.1,
                'preferred_booking_duration': 2.0
            }
        
        try:
            bookings = self.db.query(MRBSEntry).filter(
                MRBSEntry.create_by == user_id
            ).order_by(MRBSEntry.timestamp.desc()).limit(200).all()
            
            if not bookings:
                return {}
            
            patterns = {
                'total_bookings': len(bookings),
                'booking_frequency': self._calculate_booking_frequency(bookings),
                'peak_usage_times': self._find_peak_usage_times(bookings),
                'typical_advance_booking': self._calculate_advance_booking_time(bookings),
                'cancellation_rate': self._calculate_cancellation_rate(user_id),
                'preferred_booking_duration': self._calculate_preferred_duration(bookings)
            }
            
            return patterns
        except Exception as e:
            logger.error(f"Error extracting booking patterns: {e}")
            return {}
    
    def _calculate_booking_frequency(self, bookings: List[MRBSEntry]) -> Dict[str, float]:
        """Calculate booking frequency patterns"""
        if not bookings:
            return {}
        
        try:
            # Calculate bookings per week/month
            dates = [datetime.fromtimestamp(b.start_time).date() for b in bookings if hasattr(b, 'start_time')]
            
            if not dates:
                return {}
            
            date_range = (max(dates) - min(dates)).days
            
            if date_range == 0:
                return {'per_week': len(bookings), 'per_month': len(bookings)}
            
            bookings_per_week = len(bookings) * 7 / date_range
            bookings_per_month = len(bookings) * 30 / date_range
            
            return {
                'per_week': round(bookings_per_week, 2),
                'per_month': round(bookings_per_month, 2)
            }
        except Exception as e:
            logger.warning(f"Error calculating booking frequency: {e}")
            return {}
    
    def _find_peak_usage_times(self, bookings: List[MRBSEntry]) -> Dict[str, List[int]]:
        """Find peak usage times by hour and day of week"""
        try:
            hours = []
            days = []
            
            for booking in bookings:
                try:
                    start_time = datetime.fromtimestamp(booking.start_time)
                    hours.append(start_time.hour)
                    days.append(start_time.weekday())
                except (AttributeError, ValueError):
                    continue
            
            if not hours or not days:
                return {'peak_hours': [], 'peak_days': []}
            
            hour_counts = Counter(hours)
            day_counts = Counter(days)
            
            peak_hours = [hour for hour, count in hour_counts.most_common(3)]
            peak_days = [day for day, count in day_counts.most_common(3)]
            
            return {
                'peak_hours': peak_hours,
                'peak_days': peak_days
            }
        except Exception as e:
            logger.warning(f"Error finding peak usage times: {e}")
            return {'peak_hours': [], 'peak_days': []}
    
    def _calculate_advance_booking_time(self, bookings: List[MRBSEntry]) -> float:
        """Calculate typical advance booking time in hours"""
        try:
            advance_times = []
            
            for booking in bookings:
                try:
                    booking_time = booking.timestamp
                    start_time = datetime.fromtimestamp(booking.start_time)
                    advance_hours = (start_time - booking_time).total_seconds() / 3600
                    
                    if advance_hours > 0:  # Only positive advance times
                        advance_times.append(advance_hours)
                except (AttributeError, ValueError):
                    continue
            
            return np.median(advance_times) if advance_times else 24.0
        except Exception as e:
            logger.warning(f"Error calculating advance booking time: {e}")
            return 24.0
    
    def _calculate_cancellation_rate(self, user_id: str) -> float:
        """Calculate user's booking cancellation rate"""
        # This would need to be implemented based on how cancellations are tracked
        # For now, return a default value
        return 0.1
    
    def _calculate_preferred_duration(self, bookings: List[MRBSEntry]) -> float:
        """Calculate user's preferred booking duration"""
        try:
            durations = []
            for booking in bookings:
                try:
                    duration = (booking.end_time - booking.start_time) / 3600
                    if duration > 0:
                        durations.append(duration)
                except (AttributeError, TypeError):
                    continue
            
            return np.median(durations) if durations else 2.0
        except Exception as e:
            logger.warning(f"Error calculating preferred duration: {e}")
            return 2.0
    
    def _get_interaction_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user interaction history - simplified for current models"""
        # Since we don't have UserInteraction table yet, return empty list
        # This can be populated when the table is added
        return []
    
    def _get_cached_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get cached user profile"""
        try:
            return self.cache_manager.get(f"user_profile:{user_id}")
        except Exception as e:
            logger.warning(f"Error getting cached profile: {e}")
            return None
    
    def _cache_profile(self, profile: UserProfile):
        """Cache user profile"""
        try:
            self.cache_manager.set(
                f"user_profile:{profile.user_id}", 
                profile, 
                ttl=3600  # Cache for 1 hour
            )
        except Exception as e:
            logger.warning(f"Error caching profile: {e}")
    
    def _is_profile_fresh(self, profile: UserProfile) -> bool:
        """Check if cached profile is still fresh"""
        try:
            return (datetime.now() - profile.updated_at).total_seconds() < 1800  # 30 minutes
        except Exception:
            return False
    
    def _update_user_preferences_db(self, profile: UserProfile):
        """Update user preferences in database - simplified for current models"""
        try:
            # Since we don't have UserPreference table yet, we'll store preferences
            # as JSON in a simple way or skip database storage for now
            # This method can be implemented when preference tables are added
            
            logger.debug(f"Preference storage skipped - no UserPreference table available")
            
        except Exception as e:
            logger.error(f"Error updating preferences in database: {str(e)}")
            if hasattr(self, 'db') and self.db:
                try:
                    self.db.rollback()
                except Exception:
                    pass
    
    def _get_default_profile(self, user_id: str) -> UserProfile:
        """Get default profile for new users"""
        return UserProfile(
            user_id=user_id,
            preferences={},
            booking_patterns={},
            interaction_history=[],
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    
    def _process_new_feedback(self, feedback: Dict[str, Any]) -> Dict[PreferenceType, List[PreferenceScore]]:
        """Process new feedback into preference scores"""
        preferences = defaultdict(list)
        
        # This would process the feedback based on its structure
        # Implementation depends on feedback format
        try:
            if 'time_preference' in feedback:
                time_pref = feedback['time_preference']
                preferences[PreferenceType.TIME_SLOT].append(
                    PreferenceScore(
                        attribute="feedback_time_preference",
                        value=time_pref,
                        score=0.8,
                        confidence=0.7,
                        last_updated=datetime.now(),
                        source="feedback"
                    )
                )
            
            if 'room_features' in feedback:
                for feature in feedback['room_features']:
                    preferences[PreferenceType.FEATURES].append(
                        PreferenceScore(
                            attribute="room_feature",
                            value=feature,
                            score=0.9,
                            confidence=0.8,
                            last_updated=datetime.now(),
                            source="feedback"
                        )
                    )
        except Exception as e:
            logger.error(f"Error processing feedback: {e}")
        
        return dict(preferences)