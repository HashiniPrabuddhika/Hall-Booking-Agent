# recommendtion/recommendations/utils/cache_manager.py
import sqlite3
import json
import pickle
import threading
from typing import Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
from recommendtion.config.recommendation_config import RecommendationConfig

class CacheManager:
    """SQLite-based cache manager for recommendations"""
    
    def __init__(self):
        try:
            # Get cache database path from config or use default
            self.db_path = getattr(RecommendationConfig, 'CACHE_DB_PATH', 'cache/recommendations_cache.db')
            
            # Ensure directory exists
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Thread-local storage for database connections
            self._local = threading.local()
            
            # Initialize database schema
            self._init_db()
            self.use_sqlite = True
            
            # Clean expired entries on startup
            self._cleanup_expired()
            
        except Exception as e:
            # Fallback to in-memory cache if SQLite unavailable
            self.use_sqlite = False
            self._memory_cache = {}
            print(f"Warning: SQLite unavailable ({e}), using in-memory cache")
    
    def _get_connection(self):
        """Get thread-local database connection"""
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(
                self.db_path,
                timeout=30.0,
                check_same_thread=False
            )
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection
    
    def _init_db(self):
        """Initialize cache database schema"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cache_entries (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                expires_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create index for expiration cleanup
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_expires_at 
            ON cache_entries(expires_at)
        ''')
        
        conn.commit()
    
    def _cleanup_expired(self):
        """Remove expired cache entries"""
        if not self.use_sqlite:
            return
            
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            current_time = datetime.now()
            cursor.execute(
                'DELETE FROM cache_entries WHERE expires_at < ?',
                (current_time,)
            )
            conn.commit()
            
        except Exception as e:
            print(f"Cache cleanup error: {e}")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        try:
            if self.use_sqlite:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                current_time = datetime.now()
                cursor.execute('''
                    SELECT value FROM cache_entries 
                    WHERE key = ? AND (expires_at IS NULL OR expires_at > ?)
                ''', (f"rec:{key}", current_time))
                
                row = cursor.fetchone()
                if row:
                    return json.loads(row['value'])
            else:
                return self._memory_cache.get(key)
                
        except Exception as e:
            print(f"Cache get error: {e}")
        return None
    
    async def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set cached value with TTL"""
        try:
            ttl = ttl or getattr(RecommendationConfig, 'CACHE_TTL', 3600)
            
            if self.use_sqlite:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                serialized = json.dumps(value, default=str)
                expires_at = datetime.now() + timedelta(seconds=ttl) if ttl > 0 else None
                
                cursor.execute('''
                    INSERT OR REPLACE INTO cache_entries (key, value, expires_at)
                    VALUES (?, ?, ?)
                ''', (f"rec:{key}", serialized, expires_at))
                
                conn.commit()
                return True
            else:
                self._memory_cache[key] = value
                # Note: In-memory cache doesn't implement TTL for simplicity
                return True
                
        except Exception as e:
            print(f"Cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete cached value"""
        try:
            if self.use_sqlite:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                cursor.execute('DELETE FROM cache_entries WHERE key = ?', (f"rec:{key}",))
                conn.commit()
                
                return cursor.rowcount > 0
            else:
                if key in self._memory_cache:
                    del self._memory_cache[key]
                    return True
                    
        except Exception as e:
            print(f"Cache delete error: {e}")
        return False
    
    async def clear_user_cache(self, user_id: str):
        """Clear all cached data for a user"""
        try:
            if self.use_sqlite:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                # Find and delete all keys containing user_id
                cursor.execute(
                    'DELETE FROM cache_entries WHERE key LIKE ?',
                    (f"%{user_id}%",)
                )
                conn.commit()
                
            else:
                # Clear memory cache entries containing user_id
                keys_to_delete = [k for k in self._memory_cache.keys() if user_id in k]
                for key in keys_to_delete:
                    del self._memory_cache[key]
                    
        except Exception as e:
            print(f"Cache clear error: {e}")
    
    async def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        try:
            if self.use_sqlite:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                # Get total entries
                cursor.execute('SELECT COUNT(*) as total FROM cache_entries')
                total = cursor.fetchone()['total']
                
                # Get expired entries
                current_time = datetime.now()
                cursor.execute(
                    'SELECT COUNT(*) as expired FROM cache_entries WHERE expires_at < ?',
                    (current_time,)
                )
                expired = cursor.fetchone()['expired']
                
                return {
                    'total_entries': total,
                    'expired_entries': expired,
                    'active_entries': total - expired,
                    'cache_type': 'SQLite'
                }
            else:
                return {
                    'total_entries': len(self._memory_cache),
                    'expired_entries': 0,
                    'active_entries': len(self._memory_cache),
                    'cache_type': 'Memory'
                }
                
        except Exception as e:
            print(f"Cache stats error: {e}")
            return {'error': str(e)}
    
    async def cleanup_expired_entries(self):
        """Manually trigger cleanup of expired entries"""
        self._cleanup_expired()
    
    def close(self):
        """Close database connections"""
        if hasattr(self._local, 'connection'):
            self._local.connection.close()
            delattr(self._local, 'connection')