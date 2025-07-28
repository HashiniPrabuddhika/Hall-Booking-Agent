#recommendtion.recommendations.strategies.smart_scheduling.py
import asyncio
import sqlite3
import json
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum
import aiosqlite
import pickle
import base64

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from src.database import get_db
from src.models import MRBSRoom, MRBSEntry, MRBSRepeat
from ..core.pattern_analyzer import PatternAnalyzer
from ..core.preference_learner import PreferenceLearner
from ..data.analytics_processor import AnalyticsProcessor
from ..utils.time_utils import TimeUtils
from ..utils.metrics import RecommendationMetrics

logger = logging.getLogger(__name__)


class OptimizationGoal(Enum):
    """Optimization goals for smart scheduling"""
    MINIMIZE_CONFLICTS = "minimize_conflicts"
    MAXIMIZE_EFFICIENCY = "maximize_efficiency"
    REDUCE_TRAVEL_TIME = "reduce_travel_time"
    BALANCE_WORKLOAD = "balance_workload"
    ENERGY_SAVINGS = "energy_savings"
    USER_SATISFACTION = "user_satisfaction"


@dataclass
class SchedulingConstraint:
    """Represents a scheduling constraint"""
    constraint_type: str
    priority: int  # 1-10, 10 being highest priority
    parameters: Dict[str, Any]
    user_id: Optional[int] = None
    room_id: Optional[str] = None


@dataclass
class OptimizedSchedule:
    """Represents an optimized schedule recommendation"""
    original_request: Dict[str, Any]
    optimized_slot: Dict[str, Any]
    optimization_score: float
    improvements: List[str]
    trade_offs: List[str]
    confidence: float
    alternative_slots: List[Dict[str, Any]]


class SQLiteCacheManager:
    """SQLite-based cache manager"""
    
    def __init__(self, db_path: str = "cache.db"):
        self.db_path = db_path
        self.initialized = False
    
    async def initialize(self):
        """Initialize the SQLite cache database"""
        if self.initialized:
            return
            
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    expires_at REAL,
                    created_at REAL DEFAULT (julianday('now')),
                    updated_at REAL DEFAULT (julianday('now'))
                )
            """)
            
            # Create index for expiration cleanup
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_expires_at 
                ON cache_entries(expires_at)
            """)
            
            await db.commit()
        
        self.initialized = True
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        await self.initialize()
        
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT value, expires_at FROM cache_entries WHERE key = ?",
                (key,)
            )
            row = await cursor.fetchone()
            
            if not row:
                return None
            
            value_str, expires_at = row
            
            # Check if expired
            if expires_at and expires_at < datetime.now().timestamp():
                await self.delete(key)
                return None
            
            try:
                # Try to deserialize as JSON first
                return json.loads(value_str)
            except json.JSONDecodeError:
                try:
                    # If JSON fails, try pickle (for complex objects)
                    return pickle.loads(base64.b64decode(value_str))
                except:
                    # Return as string if all else fails
                    return value_str
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL (time to live) in seconds"""
        await self.initialize()
        
        # Serialize value
        try:
            # Try JSON serialization first (most efficient)
            value_str = json.dumps(value)
        except (TypeError, ValueError):
            try:
                # Use pickle for complex objects
                value_str = base64.b64encode(pickle.dumps(value)).decode('utf-8')
            except:
                # Convert to string as last resort
                value_str = str(value)
        
        expires_at = None
        if ttl:
            expires_at = (datetime.now() + timedelta(seconds=ttl)).timestamp()
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO cache_entries 
                (key, value, expires_at, updated_at) 
                VALUES (?, ?, ?, julianday('now'))
            """, (key, value_str, expires_at))
            await db.commit()
        
        return True
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        await self.initialize()
        
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "DELETE FROM cache_entries WHERE key = ?",
                (key,)
            )
            await db.commit()
            return cursor.rowcount > 0
    
    async def clear_expired(self):
        """Clear expired cache entries"""
        await self.initialize()
        
        current_time = datetime.now().timestamp()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "DELETE FROM cache_entries WHERE expires_at IS NOT NULL AND expires_at < ?",
                (current_time,)
            )
            await db.commit()
    
    async def clear_all(self):
        """Clear all cache entries"""
        await self.initialize()
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM cache_entries")
            await db.commit()
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        await self.initialize()
        
        async with aiosqlite.connect(self.db_path) as db:
            # Total entries
            cursor = await db.execute("SELECT COUNT(*) FROM cache_entries")
            total_entries = (await cursor.fetchone())[0]
            
            # Expired entries
            current_time = datetime.now().timestamp()
            cursor = await db.execute(
                "SELECT COUNT(*) FROM cache_entries WHERE expires_at IS NOT NULL AND expires_at < ?",
                (current_time,)
            )
            expired_entries = (await cursor.fetchone())[0]
            
            # Cache size (approximate)
            cursor = await db.execute("SELECT SUM(LENGTH(value)) FROM cache_entries")
            cache_size = (await cursor.fetchone())[0] or 0
            
            return {
                'total_entries': total_entries,
                'expired_entries': expired_entries,
                'active_entries': total_entries - expired_entries,
                'cache_size_bytes': cache_size
            }


class SmartSchedulingStrategy:
    """
    Advanced scheduling optimization strategy that uses ML and heuristics
    to suggest optimal meeting times and room assignments
    """
    
    def __init__(self, db: Session, db_session: Session = None, cache_db_path: str = "scheduling_cache.db"):
        """
        Initialize SmartSchedulingStrategy with proper error handling
        
        Args:
            db: Main database session (required)
            db_session: Alternative db session parameter (for backward compatibility)
            cache_db_path: Path to SQLite cache database
        """
        self.db = db or db_session  # Use whichever is provided
        
        # Initialize components with proper error handling
        try:
            # PatternAnalyzer expects db_session as positional argument
            self.pattern_analyzer = PatternAnalyzer(self.db)
        except TypeError as e:
            logger.warning(f"Could not initialize PatternAnalyzer with db_session: {e}")
            # Try without db_session parameter
            try:
                self.pattern_analyzer = PatternAnalyzer()
            except Exception as e2:
                logger.error(f"Could not initialize PatternAnalyzer: {e2}")
                self.pattern_analyzer = None
        
        try:
            self.preference_learner = PreferenceLearner()
        except Exception as e:
            logger.warning(f"Could not initialize PreferenceLearner: {e}")
            self.preference_learner = None
        
        try:
            # Pass only the database session to AnalyticsProcessor
            self.analytics_processor = AnalyticsProcessor(self.db)
        except Exception as e:
            logger.warning(f"Could not initialize AnalyticsProcessor: {e}")
            self.analytics_processor = None
        
        self.cache_manager = SQLiteCacheManager(cache_db_path)
        self.time_utils = TimeUtils()
        
        try:
            self.metrics = RecommendationMetrics()
        except Exception as e:
            logger.warning(f"Could not initialize RecommendationMetrics: {e}")
            self.metrics = None
           
        # Optimization weights (can be adjusted based on organization preferences)
        self.optimization_weights = {
            OptimizationGoal.MINIMIZE_CONFLICTS: 0.25,
            OptimizationGoal.MAXIMIZE_EFFICIENCY: 0.20,
            OptimizationGoal.REDUCE_TRAVEL_TIME: 0.15,
            OptimizationGoal.BALANCE_WORKLOAD: 0.15,
            OptimizationGoal.ENERGY_SAVINGS: 0.10,
            OptimizationGoal.USER_SATISFACTION: 0.15
        }
        
        # Schedule cache cleanup task
        asyncio.create_task(self._schedule_cache_cleanup())
    
    async def _schedule_cache_cleanup(self):
        """Schedule periodic cache cleanup"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self.cache_manager.clear_expired()
                logger.info("Cache cleanup completed")
            except Exception as e:
                logger.error(f"Error during cache cleanup: {e}")
    
    async def optimize_schedule(
        self,
        user_id: str,  # Changed from int to str to match your usage
        context: Dict[str, Any],
        current_bookings: List[Dict[str, Any]] = None,
        user_bookings: List[Dict[str, Any]] = None,
        availability_checker: callable = None
    ) -> List[Dict[str, Any]]:
        """
        Optimize schedule with simplified interface matching RecommendationEngine expectations
        
        Args:
            user_id: User identifier (string)
            context: Request context from RecommendationEngine
            current_bookings: Current bookings data from MySQL
            user_bookings: User's booking history from MySQL
            availability_checker: Function to check room availability
            
        Returns:
            List of optimization recommendations
        """
        try:
            logger.info(f"Starting schedule optimization for user {user_id}")
            
            # Create mock optimization results if components are not available
            if not self.pattern_analyzer or not self.analytics_processor:
                return self._create_mock_optimizations(user_id, context)
            
            # Extract information from context
            room_name = context.get("room_name", "")
            date = context.get("date", "")
            start_time = context.get("start_time", "")
            end_time = context.get("end_time", "")
            
            # Generate optimization recommendations
            optimizations = []
            
            # 1. Time optimization
            if self._can_optimize_time(context):
                time_opt = await self._generate_time_optimization(
                    user_id, context, user_bookings or []
                )
                if time_opt:
                    optimizations.append(time_opt)
            
            # 2. Room optimization
            if self._can_optimize_room(context):
                room_opt = await self._generate_room_optimization(
                    user_id, context, current_bookings or []
                )
                if room_opt:
                    optimizations.append(room_opt)
            
            # 3. Schedule efficiency optimization
            if user_bookings:
                efficiency_opt = await self._generate_efficiency_optimization(
                    user_id, context, user_bookings
                )
                if efficiency_opt:
                    optimizations.append(efficiency_opt)
            
            logger.info(f"Generated {len(optimizations)} optimization recommendations")
            return optimizations
            
        except Exception as e:
            logger.error(f"Error in schedule optimization: {e}")
            # Return mock data on error
            return self._create_mock_optimizations(user_id, context)
    
    async def _generate_room_optimization(
        self,
        user_id: str,
        context: Dict[str, Any],
        current_bookings: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate room-based optimization"""
        try:
            room_name = context.get('room_name', '')
            
            # Simple room optimization logic
            if 'large' in room_name.lower() or 'conference' in room_name.lower():
                return {
                    'type': 'room_optimization',
                    'score': 0.7,
                    'reason': 'Consider smaller room if fewer than 6 attendees',
                    'suggestion': {
                        'optimization_type': 'room_size',
                        'recommended_capacity': 'small',
                        'confidence': 0.7
                    }
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating room optimization: {e}")
            return None
    
    async def _generate_efficiency_optimization(
        self,
        user_id: str,
        context: Dict[str, Any],
        user_bookings: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate efficiency-based optimization"""
        try:
            # Analyze meeting frequency
            meeting_count = len(user_bookings)
            
            if meeting_count > 10:  # High meeting frequency
                return {
                    'type': 'efficiency_optimization',
                    'score': 0.65,
                    'reason': 'Consider batching meetings to create focus time blocks',
                    'suggestion': {
                        'optimization_type': 'meeting_batching',
                        'recommendation': 'group_meetings',
                        'confidence': 0.65
                    }
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating efficiency optimization: {e}")
            return None
    
    async def _analyze_time_patterns(
        self,
        user_id: str,
        bookings: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze user's time patterns from booking history"""
        try:
            if not bookings:
                return {'preferred_hours': [9, 10, 14, 15]}
            
            # Extract hours from bookings
            hours = []
            for booking in bookings:
                try:
                    start_time = booking.get('start_time', '')
                    if isinstance(start_time, str):
                        from datetime import datetime
                        dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                        hours.append(dt.hour)
                    elif hasattr(start_time, 'hour'):
                        hours.append(start_time.hour)
                except:
                    continue
            
            if not hours:
                return {'preferred_hours': [9, 10, 14, 15]}
            
            # Find most common hours
            from collections import Counter
            hour_counts = Counter(hours)
            preferred_hours = [hour for hour, count in hour_counts.most_common(4)]
            
            return {
                'preferred_hours': preferred_hours,
                'total_meetings': len(bookings),
                'hour_distribution': dict(hour_counts)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing time patterns: {e}")
            return {'preferred_hours': [9, 10, 14, 15]}

    def _create_mock_optimizations(
        self, 
        user_id: str, 
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Create mock optimization recommendations when components are unavailable"""
        room_name = context.get("room_name", "unknown_room")
        start_time = context.get("start_time", "")
        end_time = context.get("end_time", "")
        
        mock_optimizations = [
            {
                'type': 'time_optimization',
                'score': 0.75,
                'reason': f'Suggest moving meeting 30 minutes earlier for better focus time',
                'suggestion': {
                    'room_name': room_name,
                    'start_time': start_time,
                    'end_time': end_time,
                    'optimization_type': 'time_shift',
                    'confidence': 0.75
                },
                'data_source': 'mock_optimization'
            },
            {
                'type': 'efficiency_optimization',
                'score': 0.68,
                'reason': 'Combining with nearby meetings to reduce context switching',
                'suggestion': {
                    'room_name': room_name,
                    'start_time': start_time,
                    'end_time': end_time,
                    'optimization_type': 'batch_meetings',
                    'confidence': 0.68
                },
                'data_source': 'mock_optimization'
            }
        ]
        
        return mock_optimizations
    
    def _can_optimize_time(self, context: Dict[str, Any]) -> bool:
        """Check if time optimization is possible"""
        return all(key in context for key in ["start_time", "end_time", "date"])
    
    def _can_optimize_room(self, context: Dict[str, Any]) -> bool:
        """Check if room optimization is possible"""
        return "room_name" in context
    
    async def _generate_time_optimization(
        self,
        user_id: str,
        context: Dict[str, Any],
        user_bookings: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate time-based optimization"""
        try:
            # Analyze user's booking patterns if data is available
            if user_bookings and self.pattern_analyzer:
                patterns = await self._analyze_time_patterns(user_id, user_bookings)
                optimal_times = patterns.get('preferred_hours', [9, 10, 14, 15])
            else:
                optimal_times = [9, 10, 14, 15]  # Default optimal times
            
            current_hour = 12  # Default or parse from context
            try:
                if context.get('start_time'):
                    from datetime import datetime
                    current_hour = datetime.fromisoformat(context['start_time']).hour
            except:
                pass
            
            if current_hour not in optimal_times:
                suggested_hour = min(optimal_times, key=lambda x: abs(x - current_hour))
                return {
                    'type': 'time_optimization',
                    'score': 0.8,
                    'reason': f'Moving to {suggested_hour}:00 aligns with your productive hours',
                    'suggestion': {
                        'optimization_type': 'time_shift',
                        'suggested_hour': suggested_hour,
                        'confidence': 0.8
                    }
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating time optimization: {e}")
            return None
    
    async def suggest_optimal_meeting_time(
        self,
        attendees: List[int],
        duration_minutes: int,
        room_requirements: Dict[str, Any] = None,
        preferred_time_range: Tuple[datetime, datetime] = None,
        optimization_goals: List[OptimizationGoal] = None
    ) -> List[Dict[str, Any]]:
        """
        Suggest optimal meeting times considering all attendees' schedules
        
        Args:
            attendees: List of user IDs for meeting attendees
            duration_minutes: Meeting duration in minutes
            room_requirements: Requirements for the meeting room
            preferred_time_range: Preferred time range for the meeting
            optimization_goals: Optimization goals to consider
            
        Returns:
            List of optimal meeting time suggestions
        """
        try:
            logger.info(f"Finding optimal meeting time for {len(attendees)} attendees")
            
            # Create cache key for this request
            cache_key = self._generate_cache_key(
                "meeting_suggestions",
                attendees, duration_minutes, room_requirements, preferred_time_range
            )
            
            # Check cache first
            cached_result = await self.cache_manager.get(cache_key)
            if cached_result:
                logger.info("Returning cached meeting suggestions")
                return cached_result
            
            # Get availability for all attendees
            attendee_availability = {}
            attendee_preferences = {}
            
            for attendee_id in attendees:
                availability = await self._get_user_availability(
                    attendee_id, preferred_time_range
                )
                preferences = await self._load_user_preferences(attendee_id)
                
                attendee_availability[attendee_id] = availability
                attendee_preferences[attendee_id] = preferences
            
            # Find common available slots
            common_slots = self._find_common_available_slots(
                attendee_availability, duration_minutes, preferred_time_range
            )
            
            # Score and rank slots
            scored_slots = []
            for slot in common_slots:
                score = await self._score_meeting_slot(
                    slot=slot,
                    attendees=attendees,
                    attendee_preferences=attendee_preferences,
                    room_requirements=room_requirements,
                    optimization_goals=optimization_goals or []
                )
                scored_slots.append({
                    'start_time': slot['start_time'].isoformat(),
                    'end_time': slot['end_time'].isoformat(),
                    'score': score,
                    'reasoning': self._generate_slot_reasoning(slot, score)
                })
            
            # Sort by score and return top suggestions
            scored_slots.sort(key=lambda x: x['score'], reverse=True)
            result = scored_slots[:10]  # Return top 10 suggestions
            
            # Cache the result for 30 minutes
            await self.cache_manager.set(cache_key, result, ttl=1800)
            
            return result
            
        except Exception as e:
            logger.error(f"Error suggesting optimal meeting time: {e}")
            return []
    
    async def optimize_room_utilization(
        self,
        rooms: List[str] = None,
        time_period_days: int = 7,
        optimization_strategy: str = "balanced"
    ) -> Dict[str, Any]:
        """
        Optimize room utilization across the organization
        
        Args:
            rooms: List of room IDs to optimize (None for all rooms)
            time_period_days: Time period to consider for optimization
            optimization_strategy: Strategy for optimization
            
        Returns:
            Dictionary containing optimization recommendations
        """
        try:
            logger.info(f"Optimizing room utilization for {time_period_days} days")
            
            # Check cache first
            cache_key = self._generate_cache_key(
                "room_utilization", rooms, time_period_days, optimization_strategy
            )
            cached_result = await self.cache_manager.get(cache_key)
            if cached_result:
                logger.info("Returning cached room utilization analysis")
                return cached_result
            
            # Get room usage analytics
            room_analytics = await self._get_room_usage_analytics(
                rooms, time_period_days
            )
            
            # Identify optimization opportunities
            opportunities = await self._identify_utilization_opportunities(
                room_analytics, optimization_strategy
            )
            
            # Generate specific recommendations
            recommendations = await self._generate_utilization_recommendations(
                opportunities, room_analytics
            )
            
            result = {
                'current_utilization': room_analytics,
                'opportunities': opportunities,
                'recommendations': recommendations,
                'projected_improvements': self._calculate_projected_improvements(
                    recommendations
                )
            }
            
            # Cache for 2 hours
            await self.cache_manager.set(cache_key, result, ttl=7200)
            return result
            
        except Exception as e:
            logger.error(f"Error optimizing room utilization: {e}")
            return {}
    
    async def schedule_recurring_meetings_optimally(
        self,
        meeting_template: Dict[str, Any],
        recurrence_pattern: Dict[str, Any],
        duration_weeks: int = 12
    ) -> List[Dict[str, Any]]:
        """
        Optimize scheduling for recurring meetings
        
        Args:
            meeting_template: Template for the recurring meeting
            recurrence_pattern: Pattern for recurrence (daily, weekly, etc.)
            duration_weeks: How many weeks to schedule ahead
            
        Returns:
            List of optimally scheduled recurring meeting instances
        """
        try:
            logger.info("Optimizing recurring meeting schedule")
            
            # Check cache
            cache_key = self._generate_cache_key(
                "recurring_meetings", meeting_template, recurrence_pattern, duration_weeks
            )
            cached_result = await self.cache_manager.get(cache_key)
            if cached_result:
                return cached_result
            
            # Generate all required meeting instances
            meeting_instances = self._generate_recurring_instances(
                meeting_template, recurrence_pattern, duration_weeks
            )
            
            # Optimize each instance considering the series
            optimized_instances = []
            for instance in meeting_instances:
                # Consider the impact on the overall series
                optimized = await self._optimize_recurring_instance(
                    instance, meeting_instances, optimized_instances
                )
                optimized_instances.append(optimized)
            
            # Cache for 4 hours
            await self.cache_manager.set(cache_key, optimized_instances, ttl=14400)
            return optimized_instances
            
        except Exception as e:
            logger.error(f"Error optimizing recurring meetings: {e}")
            return []
    
    def _generate_cache_key(self, prefix: str, *args) -> str:
        """Generate a consistent cache key from arguments"""
        # Create a hash from the arguments
        serialized_args = json.dumps(args, sort_keys=True, default=str)
        hash_obj = hashlib.md5(serialized_args.encode())
        return f"{prefix}:{hash_obj.hexdigest()}"
    
    async def _load_user_patterns(self, user_id: int) -> Dict[str, Any]:
        """Load user booking patterns with caching"""
        cache_key = f"user_patterns:{user_id}"
        cached = await self.cache_manager.get(cache_key)
        
        if cached:
            return cached
        
        # Get user's booking history
        with next(get_db()) as db:
            bookings = db.query(MRBSEntry).filter(
                MRBSEntry.create_by == str(user_id),
                MRBSEntry.start_time >= datetime.now() - timedelta(days=90)
            ).all()
        
        patterns = self.pattern_analyzer.analyze_user_patterns(
            user_id, [booking.__dict__ for booking in bookings]
        )
        
        await self.cache_manager.set(cache_key, patterns, ttl=3600)
        return patterns
    
    async def _load_user_preferences(self, user_id: int) -> Dict[str, Any]:
        """Load user preferences with caching"""
        cache_key = f"user_preferences:{user_id}"
        cached = await self.cache_manager.get(cache_key)
        
        if cached:
            return cached
        
        preferences = self.preference_learner.learn_user_preferences(user_id)
        await self.cache_manager.set(cache_key, preferences, ttl=7200)
        return preferences
    
    async def _get_schedule_context(
        self,
        user_id: int,
        time_horizon_hours: int
    ) -> Dict[str, Any]:
        """Get current schedule context for optimization with caching"""
        cache_key = f"schedule_context:{user_id}:{time_horizon_hours}"
        cached = await self.cache_manager.get(cache_key)
        
        if cached:
            return cached
        
        end_time = datetime.now() + timedelta(hours=time_horizon_hours)
        
        with next(get_db()) as db:
            upcoming_bookings = db.query(MRBSEntry).filter(
                MRBSEntry.create_by == str(user_id),
                MRBSEntry.start_time >= datetime.now(),
                MRBSEntry.start_time <= end_time
            ).all()
        
        context = {
            'upcoming_meetings': len(upcoming_bookings),
            'meeting_density': self._calculate_meeting_density(upcoming_bookings),
            'available_slots': self._identify_available_slots(
                upcoming_bookings, time_horizon_hours
            ),
            'peak_hours': self._identify_peak_hours(upcoming_bookings)
        }
        
        # Cache for 15 minutes (schedule context changes frequently)
        await self.cache_manager.set(cache_key, context, ttl=900)
        return context
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring"""
        return await self.cache_manager.get_stats()
    
    async def clear_user_cache(self, user_id: int):
        """Clear all cached data for a specific user"""
        patterns = [
            f"user_patterns:{user_id}",
            f"user_preferences:{user_id}",
            f"schedule_context:{user_id}:*"
        ]
        
        for pattern in patterns:
            await self.cache_manager.delete(pattern)
    
    # ... [Rest of the methods remain the same as in the original code] ...
    
    async def _optimize_single_meeting(
        self,
        request: Dict[str, Any],
        user_patterns: Dict[str, Any],
        user_preferences: Dict[str, Any],
        schedule_context: Dict[str, Any],
        optimization_goals: List[OptimizationGoal],
        constraints: List[SchedulingConstraint]
    ) -> Optional[OptimizedSchedule]:
        """Optimize a single meeting request"""
        try:
            # Extract request parameters
            requested_time = datetime.fromisoformat(request['start_time'])
            duration = timedelta(minutes=request.get('duration_minutes', 60))
            room_requirements = request.get('requirements', {})
            
            # Generate alternative time slots
            alternative_slots = await self._generate_alternative_slots(
                requested_time, duration, user_patterns, constraints
            )
            
            # Score each alternative
            scored_alternatives = []
            for slot in alternative_slots:
                score = await self._calculate_optimization_score(
                    slot, user_preferences, schedule_context, optimization_goals
                )
                scored_alternatives.append({
                    'slot': slot,
                    'score': score
                })
            
            if not scored_alternatives:
                return None
            
            # Select best alternative
            best_alternative = max(scored_alternatives, key=lambda x: x['score'])
            
            # Generate optimization explanation
            improvements, trade_offs = self._analyze_optimization_impact(
                request, best_alternative['slot'], user_patterns
            )
            
            return OptimizedSchedule(
                original_request=request,
                optimized_slot=best_alternative['slot'],
                optimization_score=best_alternative['score'],
                improvements=improvements,
                trade_offs=trade_offs,
                confidence=self._calculate_confidence_score(best_alternative),
                alternative_slots=[alt['slot'] for alt in scored_alternatives[1:6]]
            )
            
        except Exception as e:
            logger.error(f"Error optimizing single meeting: {e}")
            return None
    
    def _find_common_available_slots(
        self,
        attendee_availability: Dict[int, List[Dict]],
        duration_minutes: int,
        time_range: Tuple[datetime, datetime] = None
    ) -> List[Dict[str, Any]]:
        """Find time slots where all attendees are available"""
        if not attendee_availability:
            return []
        
        # Get the intersection of all attendees' availability
        common_slots = []
        
        # Start with the first attendee's availability
        first_attendee = list(attendee_availability.keys())[0]
        base_slots = attendee_availability[first_attendee]
        
        for slot in base_slots:
            slot_start = slot['start_time']
            slot_end = slot['end_time']
            
            # Check if this slot works for all attendees
            available_for_all = True
            for attendee_id, availability in attendee_availability.items():
                if not self._is_time_available_for_user(
                    slot_start, slot_end, availability
                ):
                    available_for_all = False
                    break
            
            if available_for_all:
                # Check if slot is long enough for the meeting
                if (slot_end - slot_start).total_seconds() >= duration_minutes * 60:
                    common_slots.append({
                        'start_time': slot_start,
                        'end_time': slot_start + timedelta(minutes=duration_minutes)
                    })
        
        return common_slots
    
    async def _score_meeting_slot(
        self,
        slot: Dict[str, Any],
        attendees: List[int],
        attendee_preferences: Dict[int, Dict],
        room_requirements: Dict[str, Any],
        optimization_goals: List[OptimizationGoal]
    ) -> float:
        """Score a meeting slot based on various factors"""
        total_score = 0.0
        
        for goal in optimization_goals:
            goal_score = 0.0
            
            if goal == OptimizationGoal.USER_SATISFACTION:
                # Score based on attendee preferences
                for attendee_id in attendees:
                    prefs = attendee_preferences.get(attendee_id, {})
                    goal_score += self._score_against_preferences(slot, prefs)
                goal_score /= len(attendees)
            
            elif goal == OptimizationGoal.MINIMIZE_CONFLICTS:
                # Score based on conflict likelihood
                goal_score = await self._score_conflict_likelihood(slot, attendees)
            
            elif goal == OptimizationGoal.MAXIMIZE_EFFICIENCY:
                # Score based on productivity factors
                goal_score = self._score_productivity_factors(slot)
            
            elif goal == OptimizationGoal.REDUCE_TRAVEL_TIME:
                # Score based on travel time between meetings
                goal_score = await self._score_travel_time(slot, attendees)
            
            # Apply goal weight
            weight = self.optimization_weights.get(goal, 0.1)
            total_score += goal_score * weight
        
        return min(max(total_score, 0.0), 1.0)  # Normalize to 0-1
    
    # ... [Include all other methods from the original code with minimal changes] ...
    
    def _score_against_preferences(
        self,
        slot: Dict[str, Any],
        preferences: Dict[str, Any]
    ) -> float:
        """Score a time slot against user preferences"""
        score = 0.5  # Base score
        
        slot_time = slot['start_time']
        
        # Check preferred times of day
        preferred_hours = preferences.get('preferred_hours', [])
        if preferred_hours:
            if slot_time.hour in preferred_hours:
                score += 0.2
            else:
                score -= 0.1
        
        # Check preferred days of week
        preferred_days = preferences.get('preferred_days', [])
        if preferred_days:
            if slot_time.weekday() in preferred_days:
                score += 0.1
            else:
                score -= 0.05
        
        # Check against blocked times
        blocked_times = preferences.get('blocked_times', [])
        for blocked in blocked_times:
            if self.time_utils.times_overlap(
                slot_time, slot['end_time'],
                blocked['start'], blocked['end']
            ):
                score -= 0.3
        
        return max(0.0, min(1.0, score))
    
    # ... [Continue with all other original methods] ...
    
    def _calculate_meeting_density(self, bookings: List) -> Dict[str, float]:
        """Calculate meeting density by time period"""
        if not bookings:
            return {'daily': 0.0, 'hourly': 0.0}
        
        # Group bookings by day and hour
        daily_counts = {}
        hourly_counts = {}
        
        for booking in bookings:
            day_key = booking.start_time.date()
            hour_key = booking.start_time.hour
            
            daily_counts[day_key] = daily_counts.get(day_key, 0) + 1
            hourly_counts[hour_key] = hourly_counts.get(hour_key, 0) + 1
        
        avg_daily = sum(daily_counts.values()) / len(daily_counts) if daily_counts else 0
        avg_hourly = sum(hourly_counts.values()) / len(hourly_counts) if hourly_counts else 0
        
        return {'daily': avg_daily, 'hourly': avg_hourly}
    
    def _generate_slot_reasoning(
        self,
        slot: Dict[str, Any],
        score: float
    ) -> List[str]:
        """Generate human-readable reasoning for slot recommendation"""
        reasoning = []
        
        slot_time = slot['start_time']
        
        if 9 <= slot_time.hour <= 11:
            reasoning.append("Optimal morning focus time")
        elif 14 <= slot_time.hour <= 16:
            reasoning.append("Good afternoon collaboration time")
        
        if slot_time.weekday() < 5:  # Weekday
            reasoning.append("Scheduled during business days")
        
        if score > 0.8:
            reasoning.append("High compatibility with attendee preferences")
        elif score > 0.6:
            reasoning.append("Good compatibility with most attendees")
        
        return reasoning