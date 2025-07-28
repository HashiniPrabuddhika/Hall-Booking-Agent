# data/analytics/analytics_manager.py
import json
import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta, date
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
from collections import defaultdict
import csv
import gzip

logger = logging.getLogger(__name__)

@dataclass
class BookingEvent:
    """Represents a booking event for analytics."""
    event_id: str
    user_id: int
    room_id: int
    event_type: str  # 'booking_created', 'booking_cancelled', 'booking_modified'
    timestamp: str
    booking_date: str
    start_time: str
    end_time: str
    duration_minutes: int
    metadata: Dict[str, Any]


@dataclass
class RecommendationEvent:
    """Represents a recommendation event for analytics."""
    event_id: str
    user_id: int
    recommendation_type: str
    recommended_items: List[Dict[str, Any]]
    timestamp: str
    accepted: bool
    accepted_item_id: Optional[int]
    response_time_ms: int
    context: Dict[str, Any]


class AnalyticsManager:
    """Manages storage and analysis of booking and recommendation analytics data."""
    
    def __init__(self, base_path: str = "data/analytics"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize analytics directories
        self.events_path = self.base_path / "events"
        self.aggregates_path = self.base_path / "aggregates"
        self.reports_path = self.base_path / "reports"
        self.exports_path = self.base_path / "exports"
        
        for path in [self.events_path, self.aggregates_path, self.reports_path, self.exports_path]:
            path.mkdir(exist_ok=True)
        
        # Initialize analytics database
        self._init_analytics_db()
    
    def _init_analytics_db(self):
        """Initialize SQLite database for analytics data."""
        db_path = self.base_path / "analytics.db"
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        
        self.conn.executescript("""
            -- Booking events table
            CREATE TABLE IF NOT EXISTS booking_events (
                event_id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                room_id INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                booking_date DATE NOT NULL,
                start_time TIME NOT NULL,
                end_time TIME NOT NULL,
                duration_minutes INTEGER NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Recommendation events table
            CREATE TABLE IF NOT EXISTS recommendation_events (
                event_id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                recommendation_type TEXT NOT NULL,
                recommended_items TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                accepted BOOLEAN DEFAULT FALSE,
                accepted_item_id INTEGER,
                response_time_ms INTEGER,
                context TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- User behavior patterns
            CREATE TABLE IF NOT EXISTS user_behavior_patterns (
                user_id INTEGER PRIMARY KEY,
                preferred_times TEXT,  -- JSON array of preferred time slots
                preferred_rooms TEXT,  -- JSON array of preferred room IDs
                booking_frequency REAL,  -- Average bookings per week
                advance_booking_days REAL,  -- Average days in advance
                cancellation_rate REAL,
                modification_rate REAL,
                recommendation_acceptance_rate REAL,
                last_updated TIMESTAMP,
                pattern_confidence REAL  -- 0-1 confidence score
            );
            
            -- Room utilization patterns
            CREATE TABLE IF NOT EXISTS room_utilization (
                room_id INTEGER,
                date DATE,
                hour INTEGER,
                utilization_rate REAL,  -- 0-1 utilization rate for that hour
                booking_count INTEGER,
                total_duration_minutes INTEGER,
                avg_booking_duration REAL,
                PRIMARY KEY (room_id, date, hour)
            );
            
            -- Daily aggregates
            CREATE TABLE IF NOT EXISTS daily_aggregates (
                date DATE PRIMARY KEY,
                total_bookings INTEGER DEFAULT 0,
                total_cancellations INTEGER DEFAULT 0,
                total_modifications INTEGER DEFAULT 0,
                unique_users INTEGER DEFAULT 0,
                unique_rooms INTEGER DEFAULT 0,
                avg_booking_duration REAL DEFAULT 0,
                peak_hour INTEGER,
                recommendation_requests INTEGER DEFAULT 0,
                recommendation_acceptances INTEGER DEFAULT 0
            );
            
            -- Recommendation performance
            CREATE TABLE IF NOT EXISTS recommendation_performance (
                recommendation_type TEXT,
                date DATE,
                total_requests INTEGER DEFAULT 0,
                total_acceptances INTEGER DEFAULT 0,
                acceptance_rate REAL DEFAULT 0,
                avg_response_time_ms REAL DEFAULT 0,
                PRIMARY KEY (recommendation_type, date)
            );
            
            CREATE INDEX IF NOT EXISTS idx_booking_events_user_time ON booking_events(user_id, timestamp);
            CREATE INDEX IF NOT EXISTS idx_booking_events_room_time ON booking_events(room_id, timestamp);
            CREATE INDEX IF NOT EXISTS idx_booking_events_date ON booking_events(booking_date);
            CREATE INDEX IF NOT EXISTS idx_recommendation_events_user ON recommendation_events(user_id);
            CREATE INDEX IF NOT EXISTS idx_recommendation_events_type ON recommendation_events(recommendation_type);
        """)
        self.conn.commit()
    
    def log_booking_event(self, booking_event: BookingEvent) -> bool:
        """Log a booking-related event."""
        try:
            self.conn.execute("""
                INSERT OR REPLACE INTO booking_events 
                (event_id, user_id, room_id, event_type, timestamp, booking_date,
                 start_time, end_time, duration_minutes, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                booking_event.event_id,
                booking_event.user_id,
                booking_event.room_id,
                booking_event.event_type,
                booking_event.timestamp,
                booking_event.booking_date,
                booking_event.start_time,
                booking_event.end_time,
                booking_event.duration_minutes,
                json.dumps(booking_event.metadata)
            ))
            self.conn.commit()
            
            # Update daily aggregates
            self._update_daily_aggregates(booking_event)
            
            # Update room utilization
            self._update_room_utilization(booking_event)
            
            logger.debug(f"Logged booking event: {booking_event.event_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error logging booking event: {e}")
            return False
    
    def log_recommendation_event(self, rec_event: RecommendationEvent) -> bool:
        """Log a recommendation-related event."""
        try:
            self.conn.execute("""
                INSERT OR REPLACE INTO recommendation_events
                (event_id, user_id, recommendation_type, recommended_items, timestamp,
                 accepted, accepted_item_id, response_time_ms, context)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                rec_event.event_id,
                rec_event.user_id,
                rec_event.recommendation_type,
                json.dumps(rec_event.recommended_items),
                rec_event.timestamp,
                rec_event.accepted,
                rec_event.accepted_item_id,
                rec_event.response_time_ms,
                json.dumps(rec_event.context)
            ))
            self.conn.commit()
            
            # Update recommendation performance
            self._update_recommendation_performance(rec_event)
            
            logger.debug(f"Logged recommendation event: {rec_event.event_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error logging recommendation event: {e}")
            return False
    
    def _update_daily_aggregates(self, booking_event: BookingEvent):
        """Update daily aggregate statistics."""
        try:
            event_date = booking_event.booking_date
            
            # Initialize record if it doesn't exist
            self.conn.execute("""
                INSERT OR IGNORE INTO daily_aggregates (date) VALUES (?)
            """, (event_date,))
            
            # Update based on event type
            if booking_event.event_type == 'booking_created':
                self.conn.execute("""
                    UPDATE daily_aggregates SET 
                        total_bookings = total_bookings + 1,
                        unique_users = (
                            SELECT COUNT(DISTINCT user_id) 
                            FROM booking_events 
                            WHERE booking_date = ? AND event_type = 'booking_created'
                        ),
                        unique_rooms = (
                            SELECT COUNT(DISTINCT room_id) 
                            FROM booking_events 
                            WHERE booking_date = ? AND event_type = 'booking_created'
                        ),
                        avg_booking_duration = (
                            SELECT AVG(duration_minutes) 
                            FROM booking_events 
                            WHERE booking_date = ? AND event_type = 'booking_created'
                        )
                    WHERE date = ?
                """, (event_date, event_date, event_date, event_date))
                
            elif booking_event.event_type == 'booking_cancelled':
                self.conn.execute("""
                    UPDATE daily_aggregates SET 
                        total_cancellations = total_cancellations + 1
                    WHERE date = ?
                """, (event_date,))
                
            elif booking_event.event_type == 'booking_modified':
                self.conn.execute("""
                    UPDATE daily_aggregates SET 
                        total_modifications = total_modifications + 1
                    WHERE date = ?
                """, (event_date,))
            
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Error updating daily aggregates: {e}")
    
    def _update_room_utilization(self, booking_event: BookingEvent):
        """Update room utilization statistics."""
        try:
            if booking_event.event_type != 'booking_created':
                return
                
            # Parse start time to get hour
            start_hour = int(booking_event.start_time.split(':')[0])
            event_date = booking_event.booking_date
            
            # Initialize record if it doesn't exist
            self.conn.execute("""
                INSERT OR IGNORE INTO room_utilization 
                (room_id, date, hour, utilization_rate, booking_count, 
                 total_duration_minutes, avg_booking_duration)
                VALUES (?, ?, ?, 0, 0, 0, 0)
            """, (booking_event.room_id, event_date, start_hour))
            
            # Update utilization data
            self.conn.execute("""
                UPDATE room_utilization SET 
                    booking_count = booking_count + 1,
                    total_duration_minutes = total_duration_minutes + ?,
                    avg_booking_duration = total_duration_minutes / booking_count,
                    utilization_rate = CAST(total_duration_minutes AS REAL) / 60.0
                WHERE room_id = ? AND date = ? AND hour = ?
            """, (booking_event.duration_minutes, booking_event.room_id, 
                  event_date, start_hour))
            
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Error updating room utilization: {e}")
    
    def _update_recommendation_performance(self, rec_event: RecommendationEvent):
        """Update recommendation performance metrics."""
        try:
            event_date = datetime.fromisoformat(rec_event.timestamp).date()
            rec_type = rec_event.recommendation_type
            
            # Initialize record if it doesn't exist
            self.conn.execute("""
                INSERT OR IGNORE INTO recommendation_performance 
                (recommendation_type, date, total_requests, total_acceptances, 
                 acceptance_rate, avg_response_time_ms)
                VALUES (?, ?, 0, 0, 0, 0)
            """, (rec_type, event_date))
            
            # Update metrics
            if rec_event.accepted:
                self.conn.execute("""
                    UPDATE recommendation_performance SET 
                        total_requests = total_requests + 1,
                        total_acceptances = total_acceptances + 1,
                        acceptance_rate = CAST(total_acceptances AS REAL) / total_requests,
                        avg_response_time_ms = (
                            (avg_response_time_ms * (total_requests - 1) + ?) / total_requests
                        )
                    WHERE recommendation_type = ? AND date = ?
                """, (rec_event.response_time_ms, rec_type, event_date))
            else:
                self.conn.execute("""
                    UPDATE recommendation_performance SET 
                        total_requests = total_requests + 1,
                        acceptance_rate = CAST(total_acceptances AS REAL) / total_requests,
                        avg_response_time_ms = (
                            (avg_response_time_ms * (total_requests - 1) + ?) / total_requests
                        )
                    WHERE recommendation_type = ? AND date = ?
                """, (rec_event.response_time_ms, rec_type, event_date))
            
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Error updating recommendation performance: {e}")
    
    def analyze_user_behavior(self, user_id: int) -> Dict[str, Any]:
        """Analyze behavior patterns for a specific user."""
        try:
            # Get user's booking events
            cursor = self.conn.execute("""
                SELECT * FROM booking_events 
                WHERE user_id = ? AND timestamp > ?
                ORDER BY timestamp DESC
            """, (user_id, (datetime.now() - timedelta(days=90)).isoformat()))
            
            events = cursor.fetchall()
            if not events:
                return {}
            
            # Analyze patterns
            booking_times = []
            room_preferences = defaultdict(int)
            advance_days = []
            cancellation_count = 0
            modification_count = 0
            total_bookings = 0
            
            for event in events:
                event_type = event[3]  # event_type column
                timestamp = datetime.fromisoformat(event[4])
                booking_date = datetime.fromisoformat(event[5]).date()
                start_time = event[6]
                room_id = event[2]
                
                if event_type == 'booking_created':
                    total_bookings += 1
                    booking_times.append(start_time)
                    room_preferences[room_id] += 1
                    
                    # Calculate advance booking days
                    advance_days.append((booking_date - timestamp.date()).days)
                    
                elif event_type == 'booking_cancelled':
                    cancellation_count += 1
                elif event_type == 'booking_modified':
                    modification_count += 1
            
            # Get recommendation acceptance rate
            cursor = self.conn.execute("""
                SELECT COUNT(*), SUM(CASE WHEN accepted THEN 1 ELSE 0 END)
                FROM recommendation_events 
                WHERE user_id = ? AND timestamp > ?
            """, (user_id, (datetime.now() - timedelta(days=90)).isoformat()))
            
            rec_total, rec_accepted = cursor.fetchone()
            
            # Calculate patterns
            patterns = {
                'user_id': user_id,
                'total_bookings': total_bookings,
                'booking_frequency': total_bookings / 13 if total_bookings > 0 else 0,  # per week
                'preferred_times': self._find_preferred_times(booking_times),
                'preferred_rooms': dict(sorted(room_preferences.items(), 
                                             key=lambda x: x[1], reverse=True)[:5]),
                'advance_booking_days': np.mean(advance_days) if advance_days else 0,
                'cancellation_rate': cancellation_count / total_bookings if total_bookings > 0 else 0,
                'modification_rate': modification_count / total_bookings if total_bookings > 0 else 0,
                'recommendation_acceptance_rate': rec_accepted / rec_total if rec_total > 0 else 0,
                'pattern_confidence': min(1.0, total_bookings / 10.0)  # Higher confidence with more data
            }
            
            # Store in user behavior patterns table
            self._store_user_patterns(patterns)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing user behavior: {e}")
            return {}
    
    def _find_preferred_times(self, booking_times: List[str]) -> List[str]:
        """Find preferred booking time slots."""
        if not booking_times:
            return []
        
        # Group times into time slots
        time_slots = defaultdict(int)
        
        for time_str in booking_times:
            hour = int(time_str.split(':')[0])
            
            if 6 <= hour < 9:
                time_slots['morning'] += 1
            elif 9 <= hour < 12:
                time_slots['late_morning'] += 1
            elif 12 <= hour < 14:
                time_slots['lunch'] += 1
            elif 14 <= hour < 17:
                time_slots['afternoon'] += 1
            elif 17 <= hour < 20:
                time_slots['evening'] += 1
            else:
                time_slots['other'] += 1
        
        # Return top 3 preferred time slots
        sorted_slots = sorted(time_slots.items(), key=lambda x: x[1], reverse=True)
        return [slot[0] for slot in sorted_slots[:3]]
    
    def _store_user_patterns(self, patterns: Dict[str, Any]):
        """Store user behavior patterns in database."""
        try:
            self.conn.execute("""
                INSERT OR REPLACE INTO user_behavior_patterns
                (user_id, preferred_times, preferred_rooms, booking_frequency,
                 advance_booking_days, cancellation_rate, modification_rate,
                 recommendation_acceptance_rate, last_updated, pattern_confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                patterns['user_id'],
                json.dumps(patterns['preferred_times']),
                json.dumps(patterns['preferred_rooms']),
                patterns['booking_frequency'],
                patterns['advance_booking_days'],
                patterns['cancellation_rate'],
                patterns['modification_rate'],
                patterns['recommendation_acceptance_rate'],
                datetime.now().isoformat(),
                patterns['pattern_confidence']
            ))
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Error storing user patterns: {e}")
    
    def get_room_utilization_report(self, start_date: date, end_date: date) -> Dict[str, Any]:
        """Generate room utilization report for date range."""
        try:
            cursor = self.conn.execute("""
                SELECT room_id, 
                       AVG(utilization_rate) as avg_utilization,
                       SUM(booking_count) as total_bookings,
                       AVG(avg_booking_duration) as avg_duration,
                       MAX(utilization_rate) as peak_utilization
                FROM room_utilization 
                WHERE date BETWEEN ? AND ?
                GROUP BY room_id
                ORDER BY avg_utilization DESC
            """, (start_date, end_date))
            
            room_data = []
            for row in cursor.fetchall():
                room_data.append({
                    'room_id': row[0],
                    'avg_utilization': row[1],
                    'total_bookings': row[2],
                    'avg_duration_minutes': row[3],
                    'peak_utilization': row[4]
                })
            
            # Get hourly patterns
            cursor = self.conn.execute("""
                SELECT hour, AVG(utilization_rate) as avg_utilization
                FROM room_utilization 
                WHERE date BETWEEN ? AND ?
                GROUP BY hour
                ORDER BY hour
            """, (start_date, end_date))
            
            hourly_pattern = {row[0]: row[1] for row in cursor.fetchall()}
            
            return {
                'date_range': {'start': str(start_date), 'end': str(end_date)},
                'room_utilization': room_data,
                'hourly_pattern': hourly_pattern,
                'total_rooms': len(room_data),
                'avg_system_utilization': np.mean([r['avg_utilization'] for r in room_data]) if room_data else 0
            }
            
        except Exception as e:
            logger.error(f"Error generating utilization report: {e}")
            return {}
    
    def get_recommendation_performance_report(self, days: int = 30) -> Dict[str, Any]:
        """Generate recommendation performance report."""
        try:
            start_date = (datetime.now() - timedelta(days=days)).date()
            
            cursor = self.conn.execute("""
                SELECT recommendation_type,
                       SUM(total_requests) as total_requests,
                       SUM(total_acceptances) as total_acceptances,
                       AVG(acceptance_rate) as avg_acceptance_rate,
                       AVG(avg_response_time_ms) as avg_response_time
                FROM recommendation_performance 
                WHERE date >= ?
                GROUP BY recommendation_type
                ORDER BY avg_acceptance_rate DESC
            """, (start_date,))
            
            performance_data = []
            for row in cursor.fetchall():
                performance_data.append({
                    'recommendation_type': row[0],
                    'total_requests': row[1],
                    'total_acceptances': row[2],
                    'acceptance_rate': row[3],
                    'avg_response_time_ms': row[4]
                })
            
            # Get daily trend
            cursor = self.conn.execute("""
                SELECT date, 
                       SUM(total_requests) as daily_requests,
                       SUM(total_acceptances) as daily_acceptances
                FROM recommendation_performance 
                WHERE date >= ?
                GROUP BY date
                ORDER BY date
            """, (start_date,))
            
            daily_trend = []
            for row in cursor.fetchall():
                daily_trend.append({
                    'date': row[0],
                    'requests': row[1],
                    'acceptances': row[2],
                    'acceptance_rate': row[2] / row[1] if row[1] > 0 else 0
                })
            
            overall_acceptance_rate = (
                sum(p['total_acceptances'] for p in performance_data) /
                sum(p['total_requests'] for p in performance_data)
                if sum(p['total_requests'] for p in performance_data) > 0 else 0
            )
            
            return {
                'period_days': days,
                'performance_by_type': performance_data,
                'daily_trend': daily_trend,
                'overall_acceptance_rate': overall_acceptance_rate,
                'total_requests': sum(p['total_requests'] for p in performance_data),
                'total_acceptances': sum(p['total_acceptances'] for p in performance_data)
            }
            
        except Exception as e:
            logger.error(f"Error generating recommendation performance report: {e}")
            return {}
    
    def export_data(self, data_type: str, start_date: date, end_date: date, 
                   format: str = 'csv') -> Optional[str]:
        """Export analytics data to file."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if data_type == 'booking_events':
                filename = f"booking_events_{start_date}_{end_date}_{timestamp}.{format}"
                file_path = self.exports_path / filename
                
                cursor = self.conn.execute("""
                    SELECT * FROM booking_events 
                    WHERE booking_date BETWEEN ? AND ?
                    ORDER BY timestamp
                """, (start_date, end_date))
                
            elif data_type == 'recommendation_events':
                filename = f"recommendation_events_{start_date}_{end_date}_{timestamp}.{format}"
                file_path = self.exports_path / filename
                
                cursor = self.conn.execute("""
                    SELECT * FROM recommendation_events 
                    WHERE DATE(timestamp) BETWEEN ? AND ?
                    ORDER BY timestamp
                """, (start_date, end_date))
                
            elif data_type == 'user_patterns':
                filename = f"user_patterns_{timestamp}.{format}"
                file_path = self.exports_path / filename
                
                cursor = self.conn.execute("""
                    SELECT * FROM user_behavior_patterns 
                    ORDER BY user_id
                """)
                
            else:
                logger.error(f"Unknown data type: {data_type}")
                return None
            
            # Write data based on format
            if format == 'csv':
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    
                    # Write header
                    columns = [description[0] for description in cursor.description]
                    writer.writerow(columns)
                    
                    # Write data
                    for row in cursor.fetchall():
                        writer.writerow(row)
                        
            elif format == 'json':
                data = []
                columns = [description[0] for description in cursor.description]
                
                for row in cursor.fetchall():
                    data.append(dict(zip(columns, row)))
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Exported {data_type} data to {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return None
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get overall analytics summary."""
        try:
            # Get recent activity (last 30 days)
            thirty_days_ago = (datetime.now() - timedelta(days=30)).date()
            
            cursor = self.conn.execute("""
                SELECT 
                    SUM(total_bookings) as total_bookings,
                    SUM(total_cancellations) as total_cancellations, 
                    SUM(unique_users) as unique_users,
                    AVG(avg_booking_duration) as avg_duration
                FROM daily_aggregates 
                WHERE date >= ?
            """, (thirty_days_ago,))
            
            booking_stats = cursor.fetchone()
            
            cursor = self.conn.execute("""
                SELECT COUNT(*) as total_events 
                FROM booking_events 
                WHERE timestamp >= ?
            """, ((datetime.now() - timedelta(days=30)).isoformat(),))
            
            total_events = cursor.fetchone()[0]
            
            cursor = self.conn.execute("""
                SELECT COUNT(*) as total_rec_events 
                FROM recommendation_events 
                WHERE timestamp >= ?
            """, ((datetime.now() - timedelta(days=30)).isoformat(),))
            
            total_rec_events = cursor.fetchone()[0]
            
            # Get database size info
            cursor = self.conn.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
            db_size = cursor.fetchone()[0]
            
            return {
                'last_30_days': {
                    'total_bookings': booking_stats[0] or 0,
                    'total_cancellations': booking_stats[1] or 0,
                    'unique_users': booking_stats[2] or 0,
                    'avg_booking_duration': booking_stats[3] or 0,
                    'total_events': total_events,
                    'total_recommendation_events': total_rec_events
                },
                'database_info': {
                    'size_bytes': db_size,
                    'size_mb': db_size / (1024 * 1024)
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting analytics summary: {e}")
            return {}
    
    def cleanup_old_data(self, days_to_keep: int = 365):
        """Clean up old analytics data."""
        try:
            cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).isoformat()
            
            # Delete old booking events
            cursor = self.conn.execute(
                "SELECT COUNT(*) FROM booking_events WHERE timestamp < ?", 
                (cutoff_date,)
            )
            old_booking_count = cursor.fetchone()[0]
            
            self.conn.execute("DELETE FROM booking_events WHERE timestamp < ?", (cutoff_date,))
            
            # Delete old recommendation events
            cursor = self.conn.execute(
                "SELECT COUNT(*) FROM recommendation_events WHERE timestamp < ?", 
                (cutoff_date,)
            )
            old_rec_count = cursor.fetchone()[0]
            
            self.conn.execute("DELETE FROM recommendation_events WHERE timestamp < ?", (cutoff_date,))
            
            # Clean up aggregated data (keep longer)
            old_aggregate_date = (datetime.now() - timedelta(days=days_to_keep * 2)).date()
            self.conn.execute("DELETE FROM daily_aggregates WHERE date < ?", (old_aggregate_date,))
            self.conn.execute("DELETE FROM room_utilization WHERE date < ?", (old_aggregate_date,))
            self.conn.execute("DELETE FROM recommendation_performance WHERE date < ?", (old_aggregate_date,))
            
            self.conn.commit()
            
            # Vacuum database to reclaim space
            self.conn.execute("VACUUM")
            
            logger.info(f"Cleaned up {old_booking_count} booking events and {old_rec_count} recommendation events")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def close(self):
        """Close database connection."""
        if hasattr(self, 'conn'):
            self.conn.close()


