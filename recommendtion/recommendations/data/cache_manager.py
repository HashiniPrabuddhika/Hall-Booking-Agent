import json
import pickle
import sqlite3
import logging
import threading
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import numpy as np
from contextlib import contextmanager
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class CacheKeyType(Enum):
    """Enumeration of cache key types for better organization"""
    USER_PREFERENCES = "user_pref"
    ROOM_SIMILARITIES = "room_sim"
    BOOKING_PATTERNS = "booking_pat"
    RECOMMENDATIONS = "recommendations"
    ANALYTICS = "analytics"
    EMBEDDINGS = "embeddings"
    ML_MODELS = "ml_models"
    AVAILABILITY = "availability"
    TIME_SERIES = "time_series"
    COLLABORATIVE_FILTER = "collab_filter"

@dataclass
class CacheConfig:
    """Configuration for SQLite cache settings"""
    database_path: str = "cache.db"
    max_connections: int = 10
    timeout: float = 30.0
    check_same_thread: bool = False
    isolation_level: Optional[str] = None  
    
    # TTL settings 
    default_ttl: int = 3600  # 1 hour
    user_preferences_ttl: int = 86400  # 24 hours
    room_similarities_ttl: int = 43200  # 12 hours
    recommendations_ttl: int = 1800  # 30 minutes
    analytics_ttl: int = 7200  # 2 hours
    availability_ttl: int = 300  # 5 minutes
    embeddings_ttl: int = 604800  # 1 week
    ml_models_ttl: int = 259200  # 3 days
    
    # Maintenance settings
    cleanup_interval: int = 3600  # Clean expired entries every hour
    vacuum_interval: int = 86400  # Vacuum database daily

class CacheManager:
    """
    Comprehensive SQLite cache manager for the recommendation system
    """
    
    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
        self._local = threading.local()
        self._lock = threading.RLock()
        self._last_cleanup = datetime.now()
        self._last_vacuum = datetime.now()
        self._initialize_database()
        
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection"""
        if not hasattr(self._local, 'connection'):
            try:
                db_path = Path(self.config.database_path)
                db_path.parent.mkdir(parents=True, exist_ok=True)
                
                self._local.connection = sqlite3.connect(
                    self.config.database_path,
                    timeout=self.config.timeout,
                    check_same_thread=self.config.check_same_thread,
                    isolation_level=self.config.isolation_level
                )
                
                self._local.connection.execute("PRAGMA journal_mode=WAL")
                self._local.connection.execute("PRAGMA synchronous=NORMAL")
                self._local.connection.execute("PRAGMA cache_size=10000")
                self._local.connection.execute("PRAGMA temp_store=MEMORY")
                
            except sqlite3.Error as e:
                logger.error(f"Failed to create SQLite connection: {e}")
                raise
                
        return self._local.connection
        
    def _initialize_database(self):
        """Initialize database schema"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Create cache table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    cache_key TEXT PRIMARY KEY,
                    key_type TEXT NOT NULL,
                    value_data BLOB NOT NULL,
                    value_type TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_key_type ON cache_entries(key_type)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_expires_at ON cache_entries(expires_at)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_last_accessed ON cache_entries(last_accessed)
            """)
            
            conn.commit()
            logger.info("SQLite cache database initialized successfully")
            
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    @contextmanager
    def _handle_db_errors(self):
        """Context manager for handling database errors gracefully"""
        try:
            yield
        except sqlite3.Error as e:
            logger.error(f"SQLite operation failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in cache operation: {e}")
            raise
    
    def _generate_cache_key(self, key_type: CacheKeyType, *args, **kwargs) -> str:
        """
        Generate standardized cache keys
        
        Args:
            key_type: Type of cache key
            *args: Positional arguments to include in key
            **kwargs: Keyword arguments to include in key
            
        Returns:
            Formatted cache key string
        """
        key_parts = [key_type.value]
        
        for arg in args:
            if isinstance(arg, (dict, list)):
                arg_str = json.dumps(arg, sort_keys=True)
                arg_hash = hashlib.md5(arg_str.encode()).hexdigest()[:8]
                key_parts.append(arg_hash)
            else:
                key_parts.append(str(arg))
        
        if kwargs:
            kwargs_str = json.dumps(kwargs, sort_keys=True)
            kwargs_hash = hashlib.md5(kwargs_str.encode()).hexdigest()[:8]
            key_parts.append(kwargs_hash)
        
        return ":".join(key_parts)
    
    def _get_ttl_for_key_type(self, key_type: CacheKeyType) -> int:
        """Get appropriate TTL for different key types"""
        ttl_mapping = {
            CacheKeyType.USER_PREFERENCES: self.config.user_preferences_ttl,
            CacheKeyType.ROOM_SIMILARITIES: self.config.room_similarities_ttl,
            CacheKeyType.RECOMMENDATIONS: self.config.recommendations_ttl,
            CacheKeyType.ANALYTICS: self.config.analytics_ttl,
            CacheKeyType.AVAILABILITY: self.config.availability_ttl,
            CacheKeyType.EMBEDDINGS: self.config.embeddings_ttl,
            CacheKeyType.ML_MODELS: self.config.ml_models_ttl,
        }
        return ttl_mapping.get(key_type, self.config.default_ttl)
    
    def _serialize_value(self, value: Any) -> Tuple[bytes, str]:
        """Serialize value and return data with type indicator"""
        if isinstance(value, (dict, list)):
            return json.dumps(value, default=str).encode('utf-8'), 'json'
        elif isinstance(value, np.ndarray):
            return pickle.dumps(value), 'pickle'
        elif isinstance(value, (int, float)):
            return str(value).encode('utf-8'), 'numeric'
        elif isinstance(value, str):
            return value.encode('utf-8'), 'string'
        elif isinstance(value, bool):
            return str(value).encode('utf-8'), 'boolean'
        else:
            return pickle.dumps(value), 'pickle'
    
    def _deserialize_value(self, data: bytes, value_type: str) -> Any:
        """Deserialize value based on type indicator"""
        if value_type == 'json':
            return json.loads(data.decode('utf-8'))
        elif value_type == 'pickle':
            return pickle.loads(data)
        elif value_type == 'numeric':
            value_str = data.decode('utf-8')
            return int(value_str) if '.' not in value_str else float(value_str)
        elif value_type == 'string':
            return data.decode('utf-8')
        elif value_type == 'boolean':
            return data.decode('utf-8') == 'True'
        else:
            # Fallback to pickle
            return pickle.loads(data)
    
    def _cleanup_expired_entries(self):
        """Remove expired entries from cache"""
        if (datetime.now() - self._last_cleanup).seconds < self.config.cleanup_interval:
            return
            
        try:
            with self._handle_db_errors():
                conn = self._get_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    DELETE FROM cache_entries 
                    WHERE expires_at < CURRENT_TIMESTAMP
                """)
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                if deleted_count > 0:
                    logger.debug(f"Cleaned up {deleted_count} expired cache entries")
                
                self._last_cleanup = datetime.now()
                
        except Exception as e:
            logger.error(f"Failed to cleanup expired entries: {e}")
    
    def _vacuum_database(self):
        """Vacuum database to reclaim space"""
        if (datetime.now() - self._last_vacuum).seconds < self.config.vacuum_interval:
            return
            
        try:
            with self._handle_db_errors():
                conn = self._get_connection()
                conn.execute("VACUUM")
                logger.debug("Database vacuumed successfully")
                self._last_vacuum = datetime.now()
                
        except Exception as e:
            logger.error(f"Failed to vacuum database: {e}")
    
    def set(self, key_type: CacheKeyType, value: Any, ttl: Optional[int] = None, *args, **kwargs) -> bool:
        """
        Set a value in cache
        
        Args:
            key_type: Type of cache key
            value: Value to cache
            ttl: Time to live in seconds (optional)
            *args: Additional key components
            **kwargs: Additional key components
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self._handle_db_errors():
                cache_key = self._generate_cache_key(key_type, *args, **kwargs)
                ttl = ttl or self._get_ttl_for_key_type(key_type)
                
                # Serialize value
                value_data, value_type = self._serialize_value(value)
                
                # Calculate expiration time
                expires_at = datetime.now() + timedelta(seconds=ttl)
                
                conn = self._get_connection()
                cursor = conn.cursor()
                
                # Insert or replace entry
                cursor.execute("""
                    INSERT OR REPLACE INTO cache_entries 
                    (cache_key, key_type, value_data, value_type, expires_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (cache_key, key_type.value, value_data, value_type, expires_at))
                
                conn.commit()
                
                logger.debug(f"Cached key: {cache_key} with TTL: {ttl}")
                
                # Periodic cleanup
                self._cleanup_expired_entries()
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to set cache key {key_type.value}: {e}")
            return False
    
    def get(self, key_type: CacheKeyType, default: Any = None, *args, **kwargs) -> Any:
        """
        Get a value from cache
        
        Args:
            key_type: Type of cache key
            default: Default value if key not found
            *args: Additional key components
            **kwargs: Additional key components
            
        Returns:
            Cached value or default
        """
        try:
            with self._handle_db_errors():
                cache_key = self._generate_cache_key(key_type, *args, **kwargs)
                
                conn = self._get_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT value_data, value_type, expires_at 
                    FROM cache_entries 
                    WHERE cache_key = ? AND expires_at > CURRENT_TIMESTAMP
                """, (cache_key,))
                
                result = cursor.fetchone()
                
                if result is None:
                    return default
                
                value_data, value_type, expires_at = result
                
                cursor.execute("""
                    UPDATE cache_entries 
                    SET access_count = access_count + 1, last_accessed = CURRENT_TIMESTAMP
                    WHERE cache_key = ?
                """, (cache_key,))
                
                conn.commit()
                
                return self._deserialize_value(value_data, value_type)
                        
        except Exception as e:
            logger.error(f"Failed to get cache key {key_type.value}: {e}")
            return default
    
    def delete(self, key_type: CacheKeyType, *args, **kwargs) -> bool:
        """Delete a key from cache"""
        try:
            with self._handle_db_errors():
                cache_key = self._generate_cache_key(key_type, *args, **kwargs)
                
                conn = self._get_connection()
                cursor = conn.cursor()
                
                cursor.execute("DELETE FROM cache_entries WHERE cache_key = ?", (cache_key,))
                
                deleted = cursor.rowcount > 0
                conn.commit()
                
                if deleted:
                    logger.debug(f"Deleted cache key: {cache_key}")
                
                return deleted
                
        except Exception as e:
            logger.error(f"Failed to delete cache key {key_type.value}: {e}")
            return False
    
    def exists(self, key_type: CacheKeyType, *args, **kwargs) -> bool:
        """Check if a key exists in cache"""
        try:
            with self._handle_db_errors():
                cache_key = self._generate_cache_key(key_type, *args, **kwargs)
                
                conn = self._get_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT 1 FROM cache_entries 
                    WHERE cache_key = ? AND expires_at > CURRENT_TIMESTAMP
                """, (cache_key,))
                
                return cursor.fetchone() is not None
                
        except Exception as e:
            logger.error(f"Failed to check cache key existence {key_type.value}: {e}")
            return False
    
    def increment(self, key_type: CacheKeyType, amount: int = 1, *args, **kwargs) -> Optional[int]:
        """Increment a numeric value in cache"""
        try:
            with self._handle_db_errors():
                cache_key = self._generate_cache_key(key_type, *args, **kwargs)
                
                with self._lock:
                    current_value = self.get(key_type, 0, *args, **kwargs)
                    
                    if not isinstance(current_value, (int, float)):
                        current_value = 0
                    
                    new_value = int(current_value) + amount
                    
                    if self.set(key_type, new_value, None, *args, **kwargs):
                        return new_value
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to increment cache key {key_type.value}: {e}")
            return None
    
    def set_multiple(self, key_type: CacheKeyType, mapping: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set multiple key-value pairs at once"""
        try:
            with self._handle_db_errors():
                ttl = ttl or self._get_ttl_for_key_type(key_type)
                expires_at = datetime.now() + timedelta(seconds=ttl)
                
                conn = self._get_connection()
                cursor = conn.cursor()
                
                batch_data = []
                for suffix, value in mapping.items():
                    cache_key = self._generate_cache_key(key_type, suffix)
                    value_data, value_type = self._serialize_value(value)
                    batch_data.append((cache_key, key_type.value, value_data, value_type, expires_at))
                
                cursor.executemany("""
                    INSERT OR REPLACE INTO cache_entries 
                    (cache_key, key_type, value_data, value_type, expires_at)
                    VALUES (?, ?, ?, ?, ?)
                """, batch_data)
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to set multiple cache keys {key_type.value}: {e}")
            return False
    
    def get_multiple(self, key_type: CacheKeyType, suffixes: List[str]) -> Dict[str, Any]:
        """Get multiple values at once"""
        try:
            with self._handle_db_errors():
                cache_keys = [self._generate_cache_key(key_type, suffix) for suffix in suffixes]
                
                conn = self._get_connection()
                cursor = conn.cursor()
                
                # Build query with placeholders
                placeholders = ','.join('?' * len(cache_keys))
                cursor.execute(f"""
                    SELECT cache_key, value_data, value_type 
                    FROM cache_entries 
                    WHERE cache_key IN ({placeholders}) 
                    AND expires_at > CURRENT_TIMESTAMP
                """, cache_keys)
                
                results = cursor.fetchall()
                
                # Map results back to suffixes
                key_to_suffix = {self._generate_cache_key(key_type, suffix): suffix 
                               for suffix in suffixes}
                
                result = {}
                for cache_key, value_data, value_type in results:
                    if cache_key in key_to_suffix:
                        suffix = key_to_suffix[cache_key]
                        result[suffix] = self._deserialize_value(value_data, value_type)
                
                return result
                
        except Exception as e:
            logger.error(f"Failed to get multiple cache keys {key_type.value}: {e}")
            return {}
    
    def flush_by_pattern(self, pattern: str) -> int:
        """Delete all keys matching a pattern (using LIKE)"""
        try:
            with self._handle_db_errors():
                sql_pattern = pattern.replace('*', '%').replace('?', '_')
                
                conn = self._get_connection()
                cursor = conn.cursor()
                
                cursor.execute("DELETE FROM cache_entries WHERE cache_key LIKE ?", (sql_pattern,))
                
                deleted = cursor.rowcount
                conn.commit()
                
                if deleted > 0:
                    logger.info(f"Deleted {deleted} keys matching pattern: {pattern}")
                
                return deleted
                
        except Exception as e:
            logger.error(f"Failed to flush keys by pattern {pattern}: {e}")
            return 0
    
    def flush_by_key_type(self, key_type: CacheKeyType) -> int:
        """Delete all keys of a specific type"""
        try:
            with self._handle_db_errors():
                conn = self._get_connection()
                cursor = conn.cursor()
                
                cursor.execute("DELETE FROM cache_entries WHERE key_type = ?", (key_type.value,))
                
                deleted = cursor.rowcount
                conn.commit()
                
                if deleted > 0:
                    logger.info(f"Deleted {deleted} keys of type: {key_type.value}")
                
                return deleted
                
        except Exception as e:
            logger.error(f"Failed to flush keys by type {key_type.value}: {e}")
            return 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            with self._handle_db_errors():
                conn = self._get_connection()
                cursor = conn.cursor()
                
                # Get total entries
                cursor.execute("SELECT COUNT(*) FROM cache_entries")
                total_entries = cursor.fetchone()[0]
                
                # Get expired entries
                cursor.execute("SELECT COUNT(*) FROM cache_entries WHERE expires_at <= CURRENT_TIMESTAMP")
                expired_entries = cursor.fetchone()[0]
                
                # Get entries by type
                cursor.execute("""
                    SELECT key_type, COUNT(*) 
                    FROM cache_entries 
                    WHERE expires_at > CURRENT_TIMESTAMP
                    GROUP BY key_type
                """)
                entries_by_type = dict(cursor.fetchall())
                
                # Get database size
                cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
                db_size = cursor.fetchone()[0]
                
                # Calculate hit rate (simplified - based on access patterns)
                cursor.execute("SELECT AVG(access_count) FROM cache_entries WHERE access_count > 0")
                avg_access = cursor.fetchone()[0] or 0
                
                return {
                    'total_entries': total_entries,
                    'active_entries': total_entries - expired_entries,
                    'expired_entries': expired_entries,
                    'entries_by_type': entries_by_type,
                    'database_size_bytes': db_size,
                    'database_size_mb': round(db_size / (1024 * 1024), 2),
                    'average_access_count': round(avg_access, 2)
                }
                
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {}
    
    def health_check(self) -> bool:
        """Check if SQLite database is healthy"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            return cursor.fetchone()[0] == 1
        except Exception as e:
            logger.error(f"SQLite health check failed: {e}")
            return False
    
    def optimize_database(self):
        """Optimize database performance"""
        try:
            with self._handle_db_errors():
                conn = self._get_connection()
                
                # Clean expired entries
                self._cleanup_expired_entries()
                
                # Analyze tables for query optimization
                conn.execute("ANALYZE")
                
                # Vacuum if needed
                self._vacuum_database()
                
                logger.info("Database optimization completed")
                
        except Exception as e:
            logger.error(f"Failed to optimize database: {e}")
    
    def close(self):
        """Close database connections"""
        try:
            if hasattr(self._local, 'connection'):
                self._local.connection.close()
                delattr(self._local, 'connection')
            logger.info("SQLite connections closed")
        except Exception as e:
            logger.error(f"Error closing SQLite connection: {e}")

# Convenience methods for specific recommendation system use cases

class RecommendationCacheManager(CacheManager):
    """Extended cache manager with recommendation-specific methods"""
    
    def cache_user_preferences(self, user_id: str, preferences: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Cache user preferences"""
        return self.set(CacheKeyType.USER_PREFERENCES, preferences, ttl, user_id)
    
    def get_user_preferences(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get cached user preferences"""
        return self.get(CacheKeyType.USER_PREFERENCES, None, user_id)
    
    def cache_room_similarities(self, room_id: str, similarities: Dict[str, float], ttl: Optional[int] = None) -> bool:
        """Cache room similarity scores"""
        return self.set(CacheKeyType.ROOM_SIMILARITIES, similarities, ttl, room_id)
    
    def get_room_similarities(self, room_id: str) -> Optional[Dict[str, float]]:
        """Get cached room similarities"""
        return self.get(CacheKeyType.ROOM_SIMILARITIES, None, room_id)
    
    def cache_recommendations(self, user_id: str, context: str, recommendations: List[Dict], ttl: Optional[int] = None) -> bool:
        """Cache user recommendations"""
        return self.set(CacheKeyType.RECOMMENDATIONS, recommendations, ttl, user_id, context)
    
    def get_recommendations(self, user_id: str, context: str) -> Optional[List[Dict]]:
        """Get cached recommendations"""
        return self.get(CacheKeyType.RECOMMENDATIONS, None, user_id, context)
    
    def cache_booking_patterns(self, pattern_key: str, patterns: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Cache booking patterns"""
        return self.set(CacheKeyType.BOOKING_PATTERNS, patterns, ttl, pattern_key)
    
    def get_booking_patterns(self, pattern_key: str) -> Optional[Dict[str, Any]]:
        """Get cached booking patterns"""
        return self.get(CacheKeyType.BOOKING_PATTERNS, None, pattern_key)
    
    def cache_room_availability(self, room_id: str, date: str, availability: List[Dict], ttl: Optional[int] = None) -> bool:
        """Cache room availability"""
        return self.set(CacheKeyType.AVAILABILITY, availability, ttl, room_id, date)
    
    def get_room_availability(self, room_id: str, date: str) -> Optional[List[Dict]]:
        """Get cached room availability"""
        return self.get(CacheKeyType.AVAILABILITY, None, room_id, date)
    
    def invalidate_user_cache(self, user_id: str) -> int:
        """Invalidate all cache entries for a user"""
        patterns = [
            f"{CacheKeyType.USER_PREFERENCES.value}:{user_id}:%",
            f"{CacheKeyType.RECOMMENDATIONS.value}:{user_id}:%"
        ]
        
        total_deleted = 0
        for pattern in patterns:
            total_deleted += self.flush_by_pattern(pattern)
        
        return total_deleted
    
    def invalidate_room_cache(self, room_id: str) -> int:
        """Invalidate all cache entries for a room"""
        patterns = [
            f"{CacheKeyType.ROOM_SIMILARITIES.value}:{room_id}:%",
            f"{CacheKeyType.AVAILABILITY.value}:{room_id}:%"
        ]
        
        total_deleted = 0
        for pattern in patterns:
            total_deleted += self.flush_by_pattern(pattern)
        
        return total_deleted

_cache_manager_instance = None

def get_cache_manager(config: CacheConfig = None) -> RecommendationCacheManager:
    """Get or create global cache manager instance"""
    global _cache_manager_instance
    
    if _cache_manager_instance is None or not _cache_manager_instance.health_check():
        _cache_manager_instance = RecommendationCacheManager(config)
    
    return _cache_manager_instance