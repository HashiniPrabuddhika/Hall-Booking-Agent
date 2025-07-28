# data/cache/cache_manager.py
import json
import pickle
import hashlib
import gzip
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import logging
import sqlite3
from dataclasses import dataclass, asdict
import threading
import time

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Metadata for cached recommendations."""
    cache_key: str
    user_id: int
    request_type: str
    created_at: str
    expires_at: str
    hit_count: int
    data_size_bytes: int
    compression_ratio: float


class CacheManager:
    """Advanced caching system using SQLite with file-based fallback for large data."""
    
    def __init__(self, 
                 base_path: str = "data/cache",
                 sqlite_memory_limit_mb: int = 100,
                 use_file_fallback: bool = True):
        
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.sqlite_memory_limit = sqlite_memory_limit_mb * 1024 * 1024  # Convert to bytes
        self.use_file_fallback = use_file_fallback
        
        # Initialize cache directories (for large data fallback)
        if self.use_file_fallback:
            self.recommendations_path = self.base_path / "recommendations"
            self.user_profiles_path = self.base_path / "user_profiles"
            self.room_similarities_path = self.base_path / "room_similarities"
            self.analytics_path = self.base_path / "analytics"
            
            for path in [self.recommendations_path, self.user_profiles_path,
                        self.room_similarities_path, self.analytics_path]:
                path.mkdir(exist_ok=True)
        
        # Initialize SQLite database
        self._init_sqlite_cache()
        
        # Start cleanup thread
        self._start_cleanup_thread()
    
    def _init_sqlite_cache(self):
        """Initialize SQLite database for cache data and metadata."""
        db_path = self.base_path / "cache.db"
        
        # Use WAL mode for better concurrent access
        self.conn = sqlite3.connect(
            str(db_path), 
            check_same_thread=False,
            timeout=30.0
        )
        
        # Enable WAL mode and other optimizations
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA cache_size=10000")  # 10MB cache
        self.conn.execute("PRAGMA temp_store=MEMORY")
        
        # Create tables
        self.conn.executescript("""
            -- Cache data table (for small to medium sized data)
            CREATE TABLE IF NOT EXISTS cache_data (
                cache_key TEXT PRIMARY KEY,
                data BLOB,
                compressed_size INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Cache metadata table
            CREATE TABLE IF NOT EXISTS cache_entries (
                cache_key TEXT PRIMARY KEY,
                user_id INTEGER,
                request_type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                hit_count INTEGER DEFAULT 0,
                data_size_bytes INTEGER,
                compression_ratio REAL,
                storage_type TEXT DEFAULT 'sqlite',
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Cache statistics table
            CREATE TABLE IF NOT EXISTS cache_stats (
                date TEXT PRIMARY KEY,
                total_requests INTEGER DEFAULT 0,
                cache_hits INTEGER DEFAULT 0,
                cache_misses INTEGER DEFAULT 0,
                evictions INTEGER DEFAULT 0,
                storage_used_mb REAL DEFAULT 0
            );
            
            -- Indexes for performance
            CREATE INDEX IF NOT EXISTS idx_user_id ON cache_entries(user_id);
            CREATE INDEX IF NOT EXISTS idx_request_type ON cache_entries(request_type);
            CREATE INDEX IF NOT EXISTS idx_expires_at ON cache_entries(expires_at);
            CREATE INDEX IF NOT EXISTS idx_last_accessed ON cache_entries(last_accessed);
            CREATE INDEX IF NOT EXISTS idx_storage_type ON cache_entries(storage_type);
        """)
        
        self.conn.commit()
        
        # Create a thread lock for database operations
        self._db_lock = threading.Lock()
    
    def _generate_cache_key(self, user_id: int, request_type: str, 
                          params: Dict[str, Any]) -> str:
        """Generate a unique cache key for the request."""
        # Sort parameters for consistent hashing
        sorted_params = json.dumps(params, sort_keys=True)
        key_string = f"{user_id}:{request_type}:{sorted_params}"
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def _compress_data(self, data: Any) -> Tuple[bytes, float]:
        """Compress data and return compression ratio."""
        original_data = pickle.dumps(data)
        compressed_data = gzip.compress(original_data)
        compression_ratio = len(compressed_data) / len(original_data)
        return compressed_data, compression_ratio
    
    def _decompress_data(self, compressed_data: bytes) -> Any:
        """Decompress and deserialize data."""
        original_data = gzip.decompress(compressed_data)
        return pickle.loads(original_data)
    
    def _should_store_in_file(self, data_size: int) -> bool:
        """Determine if data should be stored in file vs SQLite."""
        return self.use_file_fallback and data_size > (self.sqlite_memory_limit // 10)  # 10% of limit
    
    def _store_data(self, cache_key: str, compressed_data: bytes, category: str) -> str:
        """Store data in SQLite or file based on size."""
        storage_type = "sqlite"
        
        if self._should_store_in_file(len(compressed_data)):
            # Store in file for large data
            self._store_in_file(cache_key, compressed_data, category)
            storage_type = "file"
        else:
            # Store in SQLite for smaller data
            with self._db_lock:
                try:
                    self.conn.execute("""
                        INSERT OR REPLACE INTO cache_data 
                        (cache_key, data, compressed_size) 
                        VALUES (?, ?, ?)
                    """, (cache_key, compressed_data, len(compressed_data)))
                    self.conn.commit()
                except sqlite3.Error as e:
                    logger.warning(f"SQLite storage failed, falling back to file: {e}")
                    if self.use_file_fallback:
                        self._store_in_file(cache_key, compressed_data, category)
                        storage_type = "file"
                    else:
                        raise
        
        return storage_type
    
    def _retrieve_data(self, cache_key: str, storage_type: str, category: str) -> Optional[bytes]:
        """Retrieve data from SQLite or file based on storage type."""
        if storage_type == "sqlite":
            with self._db_lock:
                cursor = self.conn.execute(
                    "SELECT data FROM cache_data WHERE cache_key = ?",
                    (cache_key,)
                )
                result = cursor.fetchone()
                return result[0] if result else None
        else:
            return self._load_from_file(cache_key, category)
    
    def _store_in_file(self, cache_key: str, data: bytes, category: str):
        """Store compressed data in file system."""
        if category == "recommendations":
            file_path = self.recommendations_path / f"{cache_key}.gz"
        elif category == "user_profiles":
            file_path = self.user_profiles_path / f"{cache_key}.gz"
        elif category == "room_similarities":
            file_path = self.room_similarities_path / f"{cache_key}.gz"
        else:
            file_path = self.analytics_path / f"{cache_key}.gz"
        
        with open(file_path, 'wb') as f:
            f.write(data)
    
    def _load_from_file(self, cache_key: str, category: str) -> Optional[bytes]:
        """Load compressed data from file system."""
        if category == "recommendations":
            file_path = self.recommendations_path / f"{cache_key}.gz"
        elif category == "user_profiles":
            file_path = self.user_profiles_path / f"{cache_key}.gz"
        elif category == "room_similarities":
            file_path = self.room_similarities_path / f"{cache_key}.gz"
        else:
            file_path = self.analytics_path / f"{cache_key}.gz"
        
        if file_path.exists():
            with open(file_path, 'rb') as f:
                return f.read()
        return None
    
    def set_recommendation(self, user_id: int, request_type: str, 
                         params: Dict[str, Any], data: Any,
                         ttl_hours: int = 24) -> str:
        """Cache recommendation data."""
        try:
            cache_key = self._generate_cache_key(user_id, request_type, params)
            compressed_data, compression_ratio = self._compress_data(data)
            
            expires_at = datetime.now() + timedelta(hours=ttl_hours)
            
            # Store the data
            storage_type = self._store_data(cache_key, compressed_data, "recommendations")
            
            # Update metadata
            with self._db_lock:
                self.conn.execute("""
                    INSERT OR REPLACE INTO cache_entries 
                    (cache_key, user_id, request_type, created_at, expires_at, 
                     data_size_bytes, compression_ratio, storage_type, last_accessed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (cache_key, user_id, request_type, datetime.now(), expires_at,
                      len(compressed_data), compression_ratio, storage_type, datetime.now()))
                self.conn.commit()
            
            logger.debug(f"Cached recommendation for user {user_id}, key: {cache_key[:8]}... ({storage_type})")
            return cache_key
            
        except Exception as e:
            logger.error(f"Error caching recommendation: {e}")
            return ""
    
    def get_recommendation(self, user_id: int, request_type: str, 
                         params: Dict[str, Any]) -> Optional[Any]:
        """Retrieve cached recommendation data."""
        try:
            cache_key = self._generate_cache_key(user_id, request_type, params)
            
            # Check if entry exists and is not expired
            with self._db_lock:
                cursor = self.conn.execute("""
                    SELECT storage_type, expires_at FROM cache_entries 
                    WHERE cache_key = ? AND expires_at > ?
                """, (cache_key, datetime.now()))
                
                result = cursor.fetchone()
                if not result:
                    self._record_cache_miss()
                    return None
                
                storage_type, expires_at = result
            
            # Retrieve data
            compressed_data = self._retrieve_data(cache_key, storage_type, "recommendations")
            
            if compressed_data is None:
                self._record_cache_miss()
                return None
            
            # Decompress and return data
            data = self._decompress_data(compressed_data)
            
            # Update hit count and last accessed
            with self._db_lock:
                self.conn.execute("""
                    UPDATE cache_entries SET 
                        hit_count = hit_count + 1,
                        last_accessed = ?
                    WHERE cache_key = ?
                """, (datetime.now(), cache_key))
                self.conn.commit()
            
            self._record_cache_hit()
            logger.debug(f"Cache hit for user {user_id}, key: {cache_key[:8]}...")
            return data
            
        except Exception as e:
            logger.error(f"Error retrieving cached recommendation: {e}")
            self._record_cache_miss()
            return None
    
    def cache_user_profile(self, user_id: int, profile_data: Dict[str, Any], 
                          ttl_hours: int = 168) -> str:  # 1 week default
        """Cache user profile data."""
        cache_key = f"user_profile_{user_id}"
        compressed_data, compression_ratio = self._compress_data(profile_data)
        
        storage_type = self._store_data(cache_key, compressed_data, "user_profiles")
        
        expires_at = datetime.now() + timedelta(hours=ttl_hours)
        with self._db_lock:
            self.conn.execute("""
                INSERT OR REPLACE INTO cache_entries 
                (cache_key, user_id, request_type, created_at, expires_at, 
                 data_size_bytes, compression_ratio, storage_type, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (cache_key, user_id, "user_profile", datetime.now(), expires_at,
                  len(compressed_data), compression_ratio, storage_type, datetime.now()))
            self.conn.commit()
        
        return cache_key
    
    def get_user_profile(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve cached user profile."""
        cache_key = f"user_profile_{user_id}"
        
        # Check expiration
        with self._db_lock:
            cursor = self.conn.execute("""
                SELECT storage_type FROM cache_entries 
                WHERE cache_key = ? AND expires_at > ?
            """, (cache_key, datetime.now()))
            
            result = cursor.fetchone()
            if not result:
                return None
            
            storage_type = result[0]
        
        compressed_data = self._retrieve_data(cache_key, storage_type, "user_profiles")
        
        if compressed_data:
            # Update last accessed
            with self._db_lock:
                self.conn.execute("""
                    UPDATE cache_entries SET last_accessed = ? WHERE cache_key = ?
                """, (datetime.now(), cache_key))
                self.conn.commit()
            
            return self._decompress_data(compressed_data)
        return None
    
    def cache_room_similarities(self, similarities: Dict[int, Dict[int, float]], 
                              ttl_hours: int = 72) -> str:  # 3 days default
        """Cache room similarity matrix."""
        cache_key = f"room_similarities_{datetime.now().strftime('%Y%m%d')}"
        compressed_data, compression_ratio = self._compress_data(similarities)
        
        storage_type = self._store_data(cache_key, compressed_data, "room_similarities")
        
        expires_at = datetime.now() + timedelta(hours=ttl_hours)
        with self._db_lock:
            self.conn.execute("""
                INSERT OR REPLACE INTO cache_entries 
                (cache_key, user_id, request_type, created_at, expires_at, 
                 data_size_bytes, compression_ratio, storage_type, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (cache_key, 0, "room_similarities", datetime.now(), expires_at,
                  len(compressed_data), compression_ratio, storage_type, datetime.now()))
            self.conn.commit()
        
        return cache_key
    
    def get_room_similarities(self) -> Optional[Dict[int, Dict[int, float]]]:
        """Retrieve cached room similarities."""
        # Get the most recent similarity cache
        with self._db_lock:
            cursor = self.conn.execute("""
                SELECT cache_key, storage_type FROM cache_entries 
                WHERE request_type = 'room_similarities' AND expires_at > ?
                ORDER BY created_at DESC LIMIT 1
            """, (datetime.now(),))
            
            result = cursor.fetchone()
            if not result:
                return None
            
            cache_key, storage_type = result
        
        compressed_data = self._retrieve_data(cache_key, storage_type, "room_similarities")
        
        if compressed_data:
            with self._db_lock:
                self.conn.execute("""
                    UPDATE cache_entries SET last_accessed = ? WHERE cache_key = ?
                """, (datetime.now(), cache_key))
                self.conn.commit()
            
            return self._decompress_data(compressed_data)
        return None
    
    def invalidate_user_cache(self, user_id: int):
        """Invalidate all cache entries for a specific user."""
        try:
            with self._db_lock:
                # Get all cache keys for the user
                cursor = self.conn.execute(
                    "SELECT cache_key, storage_type FROM cache_entries WHERE user_id = ?",
                    (user_id,)
                )
                
                entries_to_delete = cursor.fetchall()
                
                for cache_key, storage_type in entries_to_delete:
                    # Delete from SQLite cache_data table
                    if storage_type == "sqlite":
                        self.conn.execute("DELETE FROM cache_data WHERE cache_key = ?", (cache_key,))
                    
                    # Delete file-based cache
                    if self.use_file_fallback:
                        for category in ["recommendations", "user_profiles"]:
                            if hasattr(self, f"{category}_path"):
                                file_path = getattr(self, f"{category}_path") / f"{cache_key}.gz"
                                file_path.unlink(missing_ok=True)
                
                # Remove from metadata
                self.conn.execute("DELETE FROM cache_entries WHERE user_id = ?", (user_id,))
                self.conn.commit()
            
            logger.info(f"Invalidated cache for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error invalidating user cache: {e}")
    
    def _record_cache_hit(self):
        """Record cache hit statistics."""
        today = datetime.now().strftime('%Y-%m-%d')
        with self._db_lock:
            self.conn.execute("""
                INSERT OR IGNORE INTO cache_stats (date) VALUES (?)
            """, (today,))
            self.conn.execute("""
                UPDATE cache_stats SET 
                    total_requests = total_requests + 1,
                    cache_hits = cache_hits + 1
                WHERE date = ?
            """, (today,))
            self.conn.commit()
    
    def _record_cache_miss(self):
        """Record cache miss statistics."""
        today = datetime.now().strftime('%Y-%m-%d')
        with self._db_lock:
            self.conn.execute("""
                INSERT OR IGNORE INTO cache_stats (date) VALUES (?)
            """, (today,))
            self.conn.execute("""
                UPDATE cache_stats SET 
                    total_requests = total_requests + 1,
                    cache_misses = cache_misses + 1
                WHERE date = ?
            """, (today,))
            self.conn.commit()
    
    def _start_cleanup_thread(self):
        """Start background thread for cache cleanup."""
        def cleanup_worker():
            while True:
                try:
                    time.sleep(3600)  # Run every hour
                    self.cleanup_expired_entries()
                    self._vacuum_database()  # Periodic database maintenance
                except Exception as e:
                    logger.error(f"Error in cleanup thread: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
    
    def cleanup_expired_entries(self):
        """Remove expired cache entries."""
        try:
            with self._db_lock:
                # Find expired entries
                cursor = self.conn.execute("""
                    SELECT cache_key, storage_type FROM cache_entries 
                    WHERE expires_at < ?
                """, (datetime.now(),))
                
                expired_entries = cursor.fetchall()
                
                for cache_key, storage_type in expired_entries:
                    # Delete from SQLite cache_data
                    if storage_type == "sqlite":
                        self.conn.execute("DELETE FROM cache_data WHERE cache_key = ?", (cache_key,))
                    
                    # Delete files
                    if self.use_file_fallback:
                        for category_name in ["recommendations", "user_profiles", "room_similarities", "analytics"]:
                            if hasattr(self, f"{category_name}_path"):
                                file_path = getattr(self, f"{category_name}_path") / f"{cache_key}.gz"
                                file_path.unlink(missing_ok=True)
                
                # Remove from metadata
                self.conn.execute("DELETE FROM cache_entries WHERE expires_at < ?", (datetime.now(),))
                
                # Update eviction stats
                if expired_entries:
                    today = datetime.now().strftime('%Y-%m-%d')
                    self.conn.execute("""
                        INSERT OR IGNORE INTO cache_stats (date) VALUES (?)
                    """, (today,))
                    self.conn.execute("""
                        UPDATE cache_stats SET evictions = evictions + ? WHERE date = ?
                    """, (len(expired_entries), today))
                
                self.conn.commit()
            
            if expired_entries:
                logger.info(f"Cleaned up {len(expired_entries)} expired cache entries")
                
        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")
    
    def _vacuum_database(self):
        """Perform database maintenance."""
        try:
            with self._db_lock:
                # Only vacuum if we have significant deleted data
                cursor = self.conn.execute("PRAGMA freelist_count")
                free_pages = cursor.fetchone()[0]
                
                if free_pages > 1000:  # Only vacuum if we have > 1000 free pages
                    logger.info("Performing database vacuum...")
                    self.conn.execute("VACUUM")
                    logger.info("Database vacuum completed")
        except Exception as e:
            logger.error(f"Error during database vacuum: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            with self._db_lock:
                # Current cache size
                cursor = self.conn.execute("""
                    SELECT COUNT(*), SUM(data_size_bytes), AVG(hit_count), AVG(compression_ratio),
                           SUM(CASE WHEN storage_type = 'sqlite' THEN 1 ELSE 0 END),
                           SUM(CASE WHEN storage_type = 'file' THEN 1 ELSE 0 END)
                    FROM cache_entries WHERE expires_at > ?
                """, (datetime.now(),))
                
                count, total_size, avg_hits, avg_compression, sqlite_count, file_count = cursor.fetchone()
                
                # SQLite database size
                cursor = self.conn.execute("SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()")
                db_size = cursor.fetchone()[0]
                
                # Recent performance stats
                cursor = self.conn.execute("""
                    SELECT SUM(total_requests), SUM(cache_hits), SUM(cache_misses), SUM(evictions)
                    FROM cache_stats WHERE date >= ?
                """, ((datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),))
                
                total_req, total_hits, total_misses, total_evictions = cursor.fetchone()
                
                hit_rate = (total_hits / total_req) * 100 if total_req else 0
                
                return {
                    'active_entries': count or 0,
                    'total_size_mb': (total_size or 0) / (1024 * 1024),
                    'database_size_mb': db_size / (1024 * 1024),
                    'sqlite_entries': sqlite_count or 0,
                    'file_entries': file_count or 0,
                    'avg_hit_count': avg_hits or 0,
                    'avg_compression_ratio': avg_compression or 0,
                    'hit_rate_percent': hit_rate,
                    'total_requests_7d': total_req or 0,
                    'total_evictions_7d': total_evictions or 0,
                    'storage_backend': 'sqlite'
                }
                
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}
    
    def close(self):
        """Close connections and cleanup."""
        if hasattr(self, 'conn'):
            try:
                # Final vacuum before closing
                self.conn.execute("VACUUM")
                self.conn.close()
            except:
                pass


# Example usage and configuration
if __name__ == "__main__":
    # Initialize cache manager
    cache_manager = CacheManager(
        base_path="data/cache",
        sqlite_memory_limit_mb=50,  # Store up to 50MB in SQLite, larger data in files
        use_file_fallback=True
    )
    
    # Test the cache
    test_data = {"recommendations": [1, 2, 3, 4, 5], "scores": [0.9, 0.8, 0.7, 0.6, 0.5]}
    
    # Cache some data
    cache_key = cache_manager.set_recommendation(
        user_id=123,
        request_type="similar_rooms",
        params={"location": "Sri Lanka", "budget": 5000},
        data=test_data,
        ttl_hours=24
    )
    
    print(f"Cached data with key: {cache_key}")
    
    # Retrieve the data
    retrieved_data = cache_manager.get_recommendation(
        user_id=123,
        request_type="similar_rooms",
        params={"location": "Sri Lanka", "budget": 5000}
    )
    
    print(f"Retrieved data: {retrieved_data}")
    
    # Get cache statistics
    stats = cache_manager.get_cache_stats()
    print(f"Cache stats: {stats}")
    
    # Close the cache manager
    cache_manager.close()