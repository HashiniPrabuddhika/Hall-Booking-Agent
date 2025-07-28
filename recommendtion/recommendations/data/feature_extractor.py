#recommendtion.recommendations.data.feature_extractor
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from collections import defaultdict, Counter
import re

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Extract and process features from booking and user data for recommendations"""
    
    def __init__(self):
        """Initialize feature extractor"""
        self.user_features_cache = {}
        self.room_features_cache = {}
        
        # Define feature categories
        self.time_slots = self._generate_time_slots()
        self.equipment_types = [
            'projector', 'whiteboard', 'tv', 'microphone', 'camera',
            'computer', 'phone', 'speakers', 'screen', 'flip_chart'
        ]
        self.amenity_types = [
            'wifi', 'ac', 'heating', 'natural_light', 'quiet',
            'private', 'accessible', 'parking', 'kitchen', 'printer'
        ]
        
        logger.info("FeatureExtractor initialized")
    
    def _generate_time_slots(self) -> List[str]:
        """Generate time slot labels"""
        slots = []
        for hour in range(6, 22):  # 6 AM to 10 PM
            for minute in [0, 30]:
                time_str = f"{hour:02d}:{minute:02d}"
                slots.append(time_str)
        return slots
    
    def extract_user_features(self, user_data: Dict[str, Any], 
                            booking_history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract comprehensive user features"""
        try:
            user_id = user_data.get('user_id') or user_data.get('id')
            
            # Check cache
            if user_id in self.user_features_cache:
                return self.user_features_cache[user_id]
            
            features = {
                'user_id': user_id,
                'demographic_features': self._extract_demographic_features(user_data),
                'preference_features': self._extract_preference_features(user_data),
                'behavioral_features': self._extract_behavioral_features(booking_history or []),
                'temporal_features': self._extract_temporal_features(booking_history or []),
                'satisfaction_features': self._extract_satisfaction_features(booking_history or []),
                'usage_patterns': self._extract_usage_patterns(booking_history or [])
            }
            
            # Create feature vector
            features['feature_vector'] = self._create_user_feature_vector(features)
            
            # Cache results
            self.user_features_cache[user_id] = features
            
            logger.info(f"Extracted features for user {user_id}")
            return features
            
        except Exception as e:
            logger.error(f"Failed to extract user features: {e}")
            return {'error': str(e)}
    
    def _extract_demographic_features(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract demographic features"""
        return {
            'department': user_data.get('department', 'unknown'),
            'role': user_data.get('role', 'unknown'),
            'seniority_level': user_data.get('seniority_level', 'unknown'),
            'team_size': user_data.get('team_size', 0),
            'location': user_data.get('location', 'unknown')
        }
    
    def _extract_preference_features(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract user preference features"""
        preferences = user_data.get('preferences', {})
        
        return {
            'preferred_capacity_min': preferences.get('capacity_min', 1),
            'preferred_capacity_max': preferences.get('capacity_max', 50),
            'preferred_duration_min': preferences.get('duration_min', 30),
            'preferred_duration_max': preferences.get('duration_max', 480),
            'required_equipment': preferences.get('equipment', []),
            'preferred_amenities': preferences.get('amenities', []),
            'preferred_floors': preferences.get('floors', []),
            'preferred_buildings': preferences.get('buildings', []),
            'avoid_rooms': preferences.get('avoid_rooms', []),
            'accessibility_needs': preferences.get('accessibility_needs', False)
        }
    
    def _extract_behavioral_features(self, booking_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract behavioral features from booking history"""
        if not booking_history:
            return self._get_default_behavioral_features()
        
        # Calculate booking patterns
        total_bookings = len(booking_history)
        
        # Room preferences
        room_counts = Counter(booking.get('room_id') for booking in booking_history)
        most_used_rooms = [room_id for room_id, _ in room_counts.most_common(5)]
        
        # Duration patterns
        durations = []
        for booking in booking_history:
            start_time = datetime.fromisoformat(booking.get('start_time', ''))
            end_time = datetime.fromisoformat(booking.get('end_time', ''))
            duration = (end_time - start_time).total_seconds() / 60  # minutes
            durations.append(duration)
        
        avg_duration = np.mean(durations) if durations else 60
        std_duration = np.std(durations) if len(durations) > 1 else 0
        
        # Advance booking patterns
        advance_days = []
        for booking in booking_history:
            booking_time = datetime.fromisoformat(booking.get('booking_time', ''))
            start_time = datetime.fromisoformat(booking.get('start_time', ''))
            advance = (start_time - booking_time).total_seconds() / (24 * 3600)  # days
            advance_days.append(max(0, advance))
        
        avg_advance_booking = np.mean(advance_days) if advance_days else 1
        
        # Cancellation patterns
        cancellations = sum(1 for booking in booking_history 
                          if booking.get('status') == 'cancelled')
        cancellation_rate = cancellations / total_bookings if total_bookings > 0 else 0
        
        # No-show patterns
        no_shows = sum(1 for booking in booking_history 
                      if booking.get('attended') == False)
        no_show_rate = no_shows / total_bookings if total_bookings > 0 else 0
        
        return {
            'total_bookings': total_bookings,
            'avg_duration': avg_duration,
            'std_duration': std_duration,
            'most_used_rooms': most_used_rooms,
            'room_diversity': len(room_counts),
            'avg_advance_booking_days': avg_advance_booking,
            'cancellation_rate': cancellation_rate,
            'no_show_rate': no_show_rate,
            'booking_frequency': total_bookings / max(1, len(set(
                booking.get('start_time', '')[:10] for booking in booking_history
            )))  # bookings per unique day
        }
    
    def _extract_temporal_features(self, booking_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract temporal usage patterns"""
        if not booking_history:
            return self._get_default_temporal_features()
        
        # Day of week patterns
        day_counts = defaultdict(int)
        hour_counts = defaultdict(int)
        
        for booking in booking_history:
            start_time = datetime.fromisoformat(booking.get('start_time', ''))
            day_of_week = start_time.strftime('%A')
            hour = start_time.hour
            
            day_counts[day_of_week] += 1
            hour_counts[hour] += 1
        
        # Find preferred patterns
        preferred_days = [day for day, _ in 
                         sorted(day_counts.items(), key=lambda x: x[1], reverse=True)[:3]]
        preferred_hours = [hour for hour, _ in 
                          sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)[:3]]
        
        # Calculate peak usage times
        morning_bookings = sum(1 for booking in booking_history
                              if 6 <= datetime.fromisoformat(booking.get('start_time', '')).hour < 12)
        afternoon_bookings = sum(1 for booking in booking_history
                                if 12 <= datetime.fromisoformat(booking.get('start_time', '')).hour < 18)
        evening_bookings = sum(1 for booking in booking_history
                              if 18 <= datetime.fromisoformat(booking.get('start_time', '')).hour < 22)
        
        total = len(booking_history)
        
        return {
            'preferred_days': preferred_days,
            'preferred_hours': preferred_hours,
            'morning_preference': morning_bookings / max(1, total),
            'afternoon_preference': afternoon_bookings / max(1, total),
            'evening_preference': evening_bookings / max(1, total),
            'weekend_usage': sum(1 for booking in booking_history
                               if datetime.fromisoformat(booking.get('start_time', '')).weekday() >= 5) / max(1, total)
        }
    
    def _extract_satisfaction_features(self, booking_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract satisfaction and feedback features"""
        if not booking_history:
            return self._get_default_satisfaction_features()
        
        # Extract ratings
        ratings = [booking.get('rating') for booking in booking_history 
                  if booking.get('rating') is not None]
        
        # Extract feedback sentiments
        positive_feedback = sum(1 for booking in booking_history
                               if booking.get('feedback_sentiment') == 'positive')
        negative_feedback = sum(1 for booking in booking_history
                               if booking.get('feedback_sentiment') == 'negative')
        
        # Common issues
        issues = []
        for booking in booking_history:
            if booking.get('issues'):
                issues.extend(booking['issues'])
        
        common_issues = [issue for issue, _ in Counter(issues).most_common(5)]
        
        return {
            'avg_rating': np.mean(ratings) if ratings else 3.0,
            'rating_std': np.std(ratings) if len(ratings) > 1 else 0.5,
            'total_ratings': len(ratings),
            'positive_feedback_rate': positive_feedback / max(1, len(booking_history)),
            'negative_feedback_rate': negative_feedback / max(1, len(booking_history)),
            'common_issues': common_issues
        }
    
    def _extract_usage_patterns(self, booking_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract detailed usage patterns"""
        if not booking_history:
            return self._get_default_usage_patterns()
        
        # Equipment usage patterns
        equipment_usage = defaultdict(int)
        for booking in booking_history:
            equipment = booking.get('equipment_used', [])
            for eq in equipment:
                equipment_usage[eq] += 1
        
        # Meeting purpose patterns
        purpose_counts = Counter(booking.get('purpose', 'unknown') 
                               for booking in booking_history)
        
        # Attendee patterns
        attendee_counts = [booking.get('attendee_count', 1) for booking in booking_history
                          if booking.get('attendee_count')]
        
        # Booking lead time patterns
        lead_times = []
        for booking in booking_history:
            if booking.get('booking_time') and booking.get('start_time'):
                booking_time = datetime.fromisoformat(booking['booking_time'])
                start_time = datetime.fromisoformat(booking['start_time'])
                lead_time = (start_time - booking_time).total_seconds() / 3600  # hours
                lead_times.append(max(0, lead_time))
        
        return {
            'equipment_preferences': dict(equipment_usage),
            'common_purposes': [purpose for purpose, _ in purpose_counts.most_common(5)],
            'avg_attendee_count': np.mean(attendee_counts) if attendee_counts else 1,
            'std_attendee_count': np.std(attendee_counts) if len(attendee_counts) > 1 else 0,
            'avg_lead_time_hours': np.mean(lead_times) if lead_times else 24,
            'planning_consistency': 1 - (np.std(lead_times) / max(1, np.mean(lead_times))) if lead_times else 0.5
        }
    
    def _create_user_feature_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """Create numerical feature vector from extracted features"""
        vector = np.zeros(100)  # 100-dimensional feature vector
        
        try:
            # Behavioral features (0-19)
            behavioral = features.get('behavioral_features', {})
            vector[0] = min(behavioral.get('total_bookings', 0) / 100, 1.0)
            vector[1] = min(behavioral.get('avg_duration', 60) / 480, 1.0)
            vector[2] = min(behavioral.get('room_diversity', 1) / 20, 1.0)
            vector[3] = min(behavioral.get('avg_advance_booking_days', 1) / 30, 1.0)
            vector[4] = behavioral.get('cancellation_rate', 0)
            vector[5] = behavioral.get('no_show_rate', 0)
            vector[6] = min(behavioral.get('booking_frequency', 1) / 10, 1.0)
            
            # Temporal features (7-19)
            temporal = features.get('temporal_features', {})
            vector[7] = temporal.get('morning_preference', 0)
            vector[8] = temporal.get('afternoon_preference', 0)
            vector[9] = temporal.get('evening_preference', 0)
            vector[10] = temporal.get('weekend_usage', 0)
            
            # Preferred days encoding (one-hot style)
            preferred_days = temporal.get('preferred_days', [])
            day_mapping = {'Monday': 11, 'Tuesday': 12, 'Wednesday': 13, 
                          'Thursday': 14, 'Friday': 15, 'Saturday': 16, 'Sunday': 17}
            for day in preferred_days[:2]:  # Top 2 preferred days
                if day in day_mapping:
                    vector[day_mapping[day]] = 1.0
            
            # Satisfaction features (18-29)
            satisfaction = features.get('satisfaction_features', {})
            vector[18] = satisfaction.get('avg_rating', 3.0) / 5.0
            vector[19] = min(satisfaction.get('rating_std', 0.5) / 2.0, 1.0)
            vector[20] = satisfaction.get('positive_feedback_rate', 0)
            vector[21] = satisfaction.get('negative_feedback_rate', 0)
            
            # Preference features (22-49)
            preferences = features.get('preference_features', {})
            vector[22] = min(preferences.get('preferred_capacity_min', 1) / 50, 1.0)
            vector[23] = min(preferences.get('preferred_capacity_max', 50) / 100, 1.0)
            vector[24] = min(preferences.get('preferred_duration_min', 30) / 480, 1.0)
            vector[25] = min(preferences.get('preferred_duration_max', 480) / 960, 1.0)
            
            # Equipment preferences (26-35)
            required_equipment = preferences.get('required_equipment', [])
            for i, eq_type in enumerate(self.equipment_types[:10]):
                if eq_type in required_equipment:
                    vector[26 + i] = 1.0
            
            # Amenity preferences (36-45)
            preferred_amenities = preferences.get('preferred_amenities', [])
            for i, amenity_type in enumerate(self.amenity_types[:10]):
                if amenity_type in preferred_amenities:
                    vector[36 + i] = 1.0
            
            # Usage patterns (46-59)
            usage = features.get('usage_patterns', {})
            vector[46] = min(usage.get('avg_attendee_count', 1) / 50, 1.0)
            vector[47] = min(usage.get('std_attendee_count', 0) / 20, 1.0)
            vector[48] = min(usage.get('avg_lead_time_hours', 24) / 168, 1.0)  # Max 1 week
            vector[49] = usage.get('planning_consistency', 0.5)
            
            # Equipment usage patterns (50-59)
            equipment_prefs = usage.get('equipment_preferences', {})
            for i, eq_type in enumerate(self.equipment_types[:10]):
                usage_count = equipment_prefs.get(eq_type, 0)
                vector[50 + i] = min(usage_count / max(1, behavioral.get('total_bookings', 1)), 1.0)
            
            # Demographic features (60-79)
            demographic = features.get('demographic_features', {})
            
            # Department encoding (simple hash-based)
            dept = demographic.get('department', 'unknown')
            dept_hash = hash(dept) % 10
            vector[60 + dept_hash % 10] = 1.0
            
            # Role encoding
            role = demographic.get('role', 'unknown')
            role_hash = hash(role) % 5
            vector[70 + role_hash] = 1.0
            
            # Team size
            vector[75] = min(demographic.get('team_size', 0) / 100, 1.0)
            
            # Reserved for future features (80-99)
            # Can be used for additional demographic or contextual features
            
        except Exception as e:
            logger.error(f"Error creating user feature vector: {e}")
        
        return vector
    
    def extract_room_features(self, room_data: Dict[str, Any], 
                            usage_history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract comprehensive room features"""
        try:
            room_id = room_data.get('room_id') or room_data.get('id')
            
            # Check cache
            if room_id in self.room_features_cache:
                return self.room_features_cache[room_id]
            
            features = {
                'room_id': room_id,
                'physical_features': self._extract_physical_features(room_data),
                'equipment_features': self._extract_equipment_features(room_data),
                'location_features': self._extract_location_features(room_data),
                'usage_features': self._extract_room_usage_features(usage_history or []),
                'availability_features': self._extract_availability_features(room_data, usage_history or []),
                'quality_features': self._extract_quality_features(usage_history or [])
            }
            
            # Create feature vector
            features['feature_vector'] = self._create_room_feature_vector(features)
            
            # Cache results
            self.room_features_cache[room_id] = features
            
            logger.info(f"Extracted features for room {room_id}")
            return features
            
        except Exception as e:
            logger.error(f"Failed to extract room features: {e}")
            return {'error': str(e)}
    
    def _extract_physical_features(self, room_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract physical room features"""
        return {
            'capacity': room_data.get('capacity', 0),
            'area_sqm': room_data.get('area_sqm', 0),
            'has_windows': room_data.get('has_windows', False),
            'natural_light_rating': room_data.get('natural_light_rating', 0),
            'noise_level': room_data.get('noise_level', 3),  # 1-5 scale
            'temperature_control': room_data.get('temperature_control', 'none'),
            'furniture_type': room_data.get('furniture_type', 'standard'),
            'room_shape': room_data.get('room_shape', 'rectangular'),
            'ceiling_height': room_data.get('ceiling_height', 2.5)
        }
    
    def _extract_equipment_features(self, room_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract equipment and technology features"""
        equipment = room_data.get('equipment', [])
        
        return {
            'available_equipment': equipment,
            'equipment_count': len(equipment),
            'has_av_equipment': any(eq in equipment for eq in ['projector', 'tv', 'speakers', 'microphone']),
            'has_presentation_tools': any(eq in equipment for eq in ['projector', 'whiteboard', 'flip_chart']),
            'has_video_conferencing': any(eq in equipment for eq in ['camera', 'microphone', 'speakers']),
            'tech_level': self._calculate_tech_level(equipment),
            'equipment_age': room_data.get('equipment_age', 'unknown'),
            'maintenance_status': room_data.get('maintenance_status', 'good')
        }
    
    def _extract_location_features(self, room_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract location and accessibility features"""
        return {
            'building': room_data.get('building', 'unknown'),
            'floor': room_data.get('floor', 0),
            'wing': room_data.get('wing', 'unknown'),
            'distance_to_elevator': room_data.get('distance_to_elevator', 0),
            'distance_to_restroom': room_data.get('distance_to_restroom', 0),
            'distance_to_kitchen': room_data.get('distance_to_kitchen', 0),
            'parking_nearby': room_data.get('parking_nearby', False),
            'public_transport_access': room_data.get('public_transport_access', 'unknown'),
            'accessibility_features': room_data.get('accessibility_features', []),
            'is_accessible': len(room_data.get('accessibility_features', [])) > 0
        }
    
    def _extract_room_usage_features(self, usage_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract room usage pattern features"""
        if not usage_history:
            return self._get_default_room_usage_features()
        
        # Basic usage stats
        total_bookings = len(usage_history)
        
        # Duration patterns
        durations = []
        for booking in usage_history:
            if booking.get('start_time') and booking.get('end_time'):
                start = datetime.fromisoformat(booking['start_time'])
                end = datetime.fromisoformat(booking['end_time'])
                duration = (end - start).total_seconds() / 60
                durations.append(duration)
        
        # Utilization patterns
        utilization_by_hour = defaultdict(int)
        utilization_by_day = defaultdict(int)
        
        for booking in usage_history:
            if booking.get('start_time'):
                start_time = datetime.fromisoformat(booking['start_time'])
                utilization_by_hour[start_time.hour] += 1
                utilization_by_day[start_time.strftime('%A')] += 1
        
        # Popular time slots
        peak_hours = [hour for hour, _ in 
                     sorted(utilization_by_hour.items(), key=lambda x: x[1], reverse=True)[:3]]
        
        # User diversity
        unique_users = len(set(booking.get('user_id') for booking in usage_history 
                              if booking.get('user_id')))
        
        return {
            'total_bookings': total_bookings,
            'avg_duration_minutes': np.mean(durations) if durations else 60,
            'std_duration_minutes': np.std(durations) if len(durations) > 1 else 0,
            'peak_hours': peak_hours,
            'utilization_rate': self._calculate_utilization_rate(usage_history),
            'user_diversity': unique_users,
            'repeat_user_rate': (total_bookings - unique_users) / max(1, total_bookings),
            'popular_purposes': self._get_popular_purposes(usage_history),
            'avg_attendee_count': self._get_avg_attendee_count(usage_history)
        }
    
    def _extract_availability_features(self, room_data: Dict[str, Any], 
                                     usage_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract availability pattern features"""
        # Calculate availability windows
        busy_periods = self._identify_busy_periods(usage_history)
        
        return {
            'typical_busy_hours': busy_periods.get('hours', []),
            'typical_busy_days': busy_periods.get('days', []),
            'availability_score': self._calculate_availability_score(usage_history),
            'booking_lead_time_avg': self._calculate_avg_booking_lead_time(usage_history),
            'cancellation_rate': self._calculate_room_cancellation_rate(usage_history),
            'overbooking_incidents': self._count_overbooking_incidents(usage_history)
        }
    
    def _extract_quality_features(self, usage_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract room quality and satisfaction features"""
        if not usage_history:
            return self._get_default_quality_features()
        
        # Satisfaction metrics
        ratings = [booking.get('rating') for booking in usage_history 
                  if booking.get('rating') is not None]
        
        # Issues and complaints
        issues = []
        for booking in usage_history:
            if booking.get('issues'):
                issues.extend(booking['issues'])
        
        issue_counts = Counter(issues)
        common_issues = [issue for issue, _ in issue_counts.most_common(5)]
        
        # Maintenance requests
        maintenance_requests = sum(1 for booking in usage_history
                                  if booking.get('maintenance_requested', False))
        
        return {
            'avg_rating': np.mean(ratings) if ratings else 3.0,
            'rating_std': np.std(ratings) if len(ratings) > 1 else 0.5,
            'total_ratings': len(ratings),
            'common_issues': common_issues,
            'issue_rate': len(issues) / max(1, len(usage_history)),
            'maintenance_request_rate': maintenance_requests / max(1, len(usage_history)),
            'satisfaction_trend': self._calculate_satisfaction_trend(usage_history)
        }
    
    def _create_room_feature_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """Create numerical feature vector from extracted room features"""
        vector = np.zeros(80)  # 80-dimensional feature vector
        
        try:
            # Physical features (0-19)
            physical = features.get('physical_features', {})
            vector[0] = min(physical.get('capacity', 0) / 100, 1.0)
            vector[1] = min(physical.get('area_sqm', 0) / 200, 1.0)
            vector[2] = 1.0 if physical.get('has_windows', False) else 0.0
            vector[3] = physical.get('natural_light_rating', 0) / 5.0
            vector[4] = physical.get('noise_level', 3) / 5.0
            vector[5] = min(physical.get('ceiling_height', 2.5) / 5.0, 1.0)
            
            # Equipment features (6-25)
            equipment = features.get('equipment_features', {})
            vector[6] = min(equipment.get('equipment_count', 0) / 20, 1.0)
            vector[7] = 1.0 if equipment.get('has_av_equipment', False) else 0.0
            vector[8] = 1.0 if equipment.get('has_presentation_tools', False) else 0.0
            vector[9] = 1.0 if equipment.get('has_video_conferencing', False) else 0.0
            vector[10] = equipment.get('tech_level', 0) / 5.0
            
            # Equipment type encoding (11-20)
            available_equipment = equipment.get('available_equipment', [])
            for i, eq_type in enumerate(self.equipment_types[:10]):
                if eq_type in available_equipment:
                    vector[11 + i] = 1.0
            
            # Location features (21-35)
            location = features.get('location_features', {})
            vector[21] = min(location.get('floor', 0) / 20, 1.0)
            vector[22] = 1.0 / max(1, location.get('distance_to_elevator', 1))
            vector[23] = 1.0 / max(1, location.get('distance_to_restroom', 1))
            vector[24] = 1.0 / max(1, location.get('distance_to_kitchen', 1))
            vector[25] = 1.0 if location.get('parking_nearby', False) else 0.0
            vector[26] = 1.0 if location.get('is_accessible', False) else 0.0
            
            # Usage features (27-49)
            usage = features.get('usage_features', {})
            vector[27] = min(usage.get('total_bookings', 0) / 1000, 1.0)
            vector[28] = min(usage.get('avg_duration_minutes', 60) / 480, 1.0)
            vector[29] = usage.get('utilization_rate', 0)
            vector[30] = min(usage.get('user_diversity', 1) / 100, 1.0)
            vector[31] = usage.get('repeat_user_rate', 0)
            vector[32] = min(usage.get('avg_attendee_count', 1) / 50, 1.0)
            
            # Quality features (33-49)
            quality = features.get('quality_features', {})
            vector[33] = quality.get('avg_rating', 3.0) / 5.0
            vector[34] = min(quality.get('rating_std', 0.5) / 2.0, 1.0)
            vector[35] = quality.get('issue_rate', 0)
            vector[36] = quality.get('maintenance_request_rate', 0)
            vector[37] = quality.get('satisfaction_trend', 0)  # -1 to 1
            
            # Availability features (38-49)
            availability = features.get('availability_features', {})
            vector[38] = availability.get('availability_score', 0.5)
            vector[39] = min(availability.get('booking_lead_time_avg', 24) / 168, 1.0)
            vector[40] = availability.get('cancellation_rate', 0)
            vector[41] = min(availability.get('overbooking_incidents', 0) / 10, 1.0)
            
            # Reserved for future features (50-79)
            
        except Exception as e:
            logger.error(f"Error creating room feature vector: {e}")
        
        return vector
    
    # Helper methods for default values
    def _get_default_behavioral_features(self) -> Dict[str, Any]:
        return {
            'total_bookings': 0,
            'avg_duration': 60,
            'std_duration': 0,
            'most_used_rooms': [],
            'room_diversity': 0,
            'avg_advance_booking_days': 1,
            'cancellation_rate': 0,
            'no_show_rate': 0,
            'booking_frequency': 0
        }
    
    def _get_default_temporal_features(self) -> Dict[str, Any]:
        return {
            'preferred_days': [],
            'preferred_hours': [],
            'morning_preference': 0.33,
            'afternoon_preference': 0.33,
            'evening_preference': 0.33,
            'weekend_usage': 0
        }
    
    def _get_default_satisfaction_features(self) -> Dict[str, Any]:
        return {
            'avg_rating': 3.0,
            'rating_std': 0.5,
            'total_ratings': 0,
            'positive_feedback_rate': 0,
            'negative_feedback_rate': 0,
            'common_issues': []
        }
    
    def _get_default_usage_patterns(self) -> Dict[str, Any]:
        return {
            'equipment_preferences': {},
            'common_purposes': [],
            'avg_attendee_count': 1,
            'std_attendee_count': 0,
            'avg_lead_time_hours': 24,
            'planning_consistency': 0.5
        }
    
    def _get_default_room_usage_features(self) -> Dict[str, Any]:
        return {
            'total_bookings': 0,
            'avg_duration_minutes': 60,
            'std_duration_minutes': 0,
            'peak_hours': [],
            'utilization_rate': 0,
            'user_diversity': 0,
            'repeat_user_rate': 0,
            'popular_purposes': [],
            'avg_attendee_count': 1
        }
    
    def _get_default_quality_features(self) -> Dict[str, Any]:
        return {
            'avg_rating': 3.0,
            'rating_std': 0.5,
            'total_ratings': 0,
            'common_issues': [],
            'issue_rate': 0,
            'maintenance_request_rate': 0,
            'satisfaction_trend': 0
        }
    
    # Utility methods
    def _calculate_tech_level(self, equipment: List[str]) -> float:
        """Calculate technology level score (0-5)"""
        tech_scores = {
            'projector': 1, 'tv': 1, 'speakers': 1, 'microphone': 2,
            'camera': 2, 'computer': 2, 'smartboard': 3, 'video_conference': 3,
            'wireless_display': 3, 'touch_screen': 4, 'ai_assistant': 5
        }
        
        total_score = sum(tech_scores.get(eq, 0) for eq in equipment)
        return min(total_score / len(equipment) if equipment else 0, 5.0)
    
    def _calculate_utilization_rate(self, usage_history: List[Dict[str, Any]]) -> float:
        """Calculate room utilization rate"""
        if not usage_history:
            return 0.0
        
        # Simple calculation: bookings per available time slots
        # Assuming 12 hours available per day (8 AM - 8 PM)
        days_in_history = 30  # Assume 30-day history
        available_slots_per_day = 24  # 30-minute slots
        total_available_slots = days_in_history * available_slots_per_day
        
        return min(len(usage_history) / total_available_slots, 1.0)
    
    def _get_popular_purposes(self, usage_history: List[Dict[str, Any]]) -> List[str]:
        """Get most popular meeting purposes"""
        purposes = [booking.get('purpose', 'unknown') for booking in usage_history]
        purpose_counts = Counter(purposes)
        return [purpose for purpose, _ in purpose_counts.most_common(3)]
    
    def _get_avg_attendee_count(self, usage_history: List[Dict[str, Any]]) -> float:
        """Get average attendee count"""
        attendee_counts = [booking.get('attendee_count', 1) for booking in usage_history
                          if booking.get('attendee_count')]
        return np.mean(attendee_counts) if attendee_counts else 1.0
    
    def _identify_busy_periods(self, usage_history: List[Dict[str, Any]]) -> Dict[str, List]:
        """Identify typically busy hours and days"""
        hour_counts = defaultdict(int)
        day_counts = defaultdict(int)
        
        for booking in usage_history:
            if booking.get('start_time'):
                start_time = datetime.fromisoformat(booking['start_time'])
                hour_counts[start_time.hour] += 1
                day_counts[start_time.strftime('%A')] += 1
        
        busy_threshold = max(1, len(usage_history) * 0.1)  # 10% threshold
        
        return {
            'hours': [hour for hour, count in hour_counts.items() if count >= busy_threshold],
            'days': [day for day, count in day_counts.items() if count >= busy_threshold]
        }
    
    def _calculate_availability_score(self, usage_history: List[Dict[str, Any]]) -> float:
        """Calculate availability score (0-1, higher = more available)"""
        if not usage_history:
            return 1.0
        
        # Simple inverse of utilization rate
        utilization = self._calculate_utilization_rate(usage_history)
        return 1.0 - utilization
    
    def _calculate_avg_booking_lead_time(self, usage_history: List[Dict[str, Any]]) -> float:
        """Calculate average booking lead time in hours"""
        lead_times = []
        
        for booking in usage_history:
            if booking.get('booking_time') and booking.get('start_time'):
                booking_time = datetime.fromisoformat(booking['booking_time'])
                start_time = datetime.fromisoformat(booking['start_time'])
                lead_time = (start_time - booking_time).total_seconds() / 3600
                lead_times.append(max(0, lead_time))
        
        return np.mean(lead_times) if lead_times else 24.0
    
    def _calculate_room_cancellation_rate(self, usage_history: List[Dict[str, Any]]) -> float:
        """Calculate cancellation rate for the room"""
        if not usage_history:
            return 0.0
        
        cancellations = sum(1 for booking in usage_history 
                           if booking.get('status') == 'cancelled')
        return cancellations / len(usage_history)
    
    def _count_overbooking_incidents(self, usage_history: List[Dict[str, Any]]) -> int:
        """Count overbooking incidents"""
        return sum(1 for booking in usage_history 
                  if booking.get('overbooking_incident', False))
    
    def _calculate_satisfaction_trend(self, usage_history: List[Dict[str, Any]]) -> float:
        """Calculate satisfaction trend (-1 to 1)"""
        ratings_with_time = []
        
        for booking in usage_history:
            if booking.get('rating') and booking.get('start_time'):
                start_time = datetime.fromisoformat(booking['start_time'])
                ratings_with_time.append((start_time, booking['rating']))
        
        if len(ratings_with_time) < 2:
            return 0.0
        
        # Sort by time
        ratings_with_time.sort(key=lambda x: x[0])
        
        # Calculate trend using simple linear regression
        ratings = [rating for _, rating in ratings_with_time]
        x = np.arange(len(ratings))
        
        if len(ratings) > 1:
            slope = np.corrcoef(x, ratings)[0, 1] if len(ratings) > 2 else (ratings[-1] - ratings[0])
            return np.clip(slope, -1.0, 1.0)
        
        return 0.0
    
    def get_feature_importance(self, feature_type: str = 'user') -> Dict[str, float]:
        """Get feature importance scores for interpretability"""
        if feature_type == 'user':
            return {
                'total_bookings': 0.15,
                'avg_duration': 0.12,
                'preferred_equipment': 0.20,
                'satisfaction_rating': 0.18,
                'temporal_patterns': 0.15,
                'cancellation_rate': 0.10,
                'room_diversity': 0.10
            }
        elif feature_type == 'room':
            return {
                'capacity': 0.20,
                'equipment_available': 0.25,
                'location_convenience': 0.15,
                'utilization_rate': 0.12,
                'satisfaction_rating': 0.18,
                'availability_score': 0.10
            }
        else:
            return {}
    
    def clear_cache(self):
        """Clear feature caches"""
        self.user_features_cache.clear()
        self.room_features_cache.clear()
        logger.info("Feature extractor caches cleared")