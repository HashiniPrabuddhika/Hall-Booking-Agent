# data/embeddings/embedding_manager.py
import os
import json
import pickle
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from pathlib import Path
import sqlite3
import hashlib

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Manages storage and retrieval of embeddings for rooms, users, and booking patterns."""
    
    def __init__(self, base_path: str = "data/embeddings"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize subdirectories
        self.room_embeddings_path = self.base_path / "rooms"
        self.user_embeddings_path = self.base_path / "users"
        self.booking_embeddings_path = self.base_path / "bookings"
        self.metadata_path = self.base_path / "metadata"
        
        for path in [self.room_embeddings_path, self.user_embeddings_path, 
                    self.booking_embeddings_path, self.metadata_path]:
            path.mkdir(exist_ok=True)
        
        # Initialize metadata database
        self._init_metadata_db()
    
    def _init_metadata_db(self):
        """Initialize SQLite database for embedding metadata."""
        db_path = self.metadata_path / "embeddings_metadata.db"
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        
        # Create tables for different embedding types
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS room_embeddings (
                room_id INTEGER PRIMARY KEY,
                embedding_hash TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                feature_version TEXT,
                dimensions INTEGER,
                file_path TEXT
            );
            
            CREATE TABLE IF NOT EXISTS user_embeddings (
                user_id INTEGER PRIMARY KEY,
                embedding_hash TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                feature_version TEXT,
                dimensions INTEGER,
                file_path TEXT
            );
            
            CREATE TABLE IF NOT EXISTS booking_embeddings (
                booking_id INTEGER PRIMARY KEY,
                embedding_hash TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                feature_version TEXT,
                dimensions INTEGER,
                file_path TEXT
            );
            
            CREATE TABLE IF NOT EXISTS embedding_versions (
                version_id TEXT PRIMARY KEY,
                model_name TEXT,
                created_at TIMESTAMP,
                description TEXT
            );
        """)
        self.conn.commit()
    
    def save_room_embedding(self, room_id: int, embedding: np.ndarray, 
                          features: Dict[str, Any], version: str = "v1.0") -> bool:
        """Save room embedding with metadata."""
        try:
            # Generate hash for embedding
            embedding_hash = hashlib.md5(embedding.tobytes()).hexdigest()
            
            # Save embedding to file
            file_path = self.room_embeddings_path / f"room_{room_id}_{embedding_hash}.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump({
                    'embedding': embedding,
                    'features': features,
                    'room_id': room_id,
                    'created_at': datetime.now().isoformat()
                }, f)
            
            # Update metadata
            self.conn.execute("""
                INSERT OR REPLACE INTO room_embeddings 
                (room_id, embedding_hash, created_at, updated_at, feature_version, dimensions, file_path)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (room_id, embedding_hash, datetime.now(), datetime.now(), 
                  version, len(embedding), str(file_path)))
            self.conn.commit()
            
            logger.info(f"Saved room embedding for room {room_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving room embedding: {e}")
            return False
    
    def load_room_embedding(self, room_id: int) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Load room embedding and features."""
        try:
            cursor = self.conn.execute(
                "SELECT file_path FROM room_embeddings WHERE room_id = ?", 
                (room_id,)
            )
            result = cursor.fetchone()
            
            if not result:
                return None
            
            file_path = Path(result[0])
            if not file_path.exists():
                logger.warning(f"Embedding file not found: {file_path}")
                return None
            
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                return data['embedding'], data['features']
                
        except Exception as e:
            logger.error(f"Error loading room embedding: {e}")
            return None
    
    def save_user_embedding(self, user_id: int, embedding: np.ndarray, 
                          preferences: Dict[str, Any], version: str = "v1.0") -> bool:
        """Save user embedding with preferences."""
        try:
            embedding_hash = hashlib.md5(embedding.tobytes()).hexdigest()
            
            file_path = self.user_embeddings_path / f"user_{user_id}_{embedding_hash}.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump({
                    'embedding': embedding,
                    'preferences': preferences,
                    'user_id': user_id,
                    'created_at': datetime.now().isoformat()
                }, f)
            
            self.conn.execute("""
                INSERT OR REPLACE INTO user_embeddings 
                (user_id, embedding_hash, created_at, updated_at, feature_version, dimensions, file_path)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (user_id, embedding_hash, datetime.now(), datetime.now(), 
                  version, len(embedding), str(file_path)))
            self.conn.commit()
            
            logger.info(f"Saved user embedding for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving user embedding: {e}")
            return False
    
    def load_user_embedding(self, user_id: int) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Load user embedding and preferences."""
        try:
            cursor = self.conn.execute(
                "SELECT file_path FROM user_embeddings WHERE user_id = ?", 
                (user_id,)
            )
            result = cursor.fetchone()
            
            if not result:
                return None
            
            file_path = Path(result[0])
            if not file_path.exists():
                return None
            
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                return data['embedding'], data['preferences']
                
        except Exception as e:
            logger.error(f"Error loading user embedding: {e}")
            return None
    
    def save_booking_embedding(self, booking_id: int, embedding: np.ndarray, 
                             context: Dict[str, Any], version: str = "v1.0") -> bool:
        """Save booking pattern embedding."""
        try:
            embedding_hash = hashlib.md5(embedding.tobytes()).hexdigest()
            
            file_path = self.booking_embeddings_path / f"booking_{booking_id}_{embedding_hash}.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump({
                    'embedding': embedding,
                    'context': context,
                    'booking_id': booking_id,
                    'created_at': datetime.now().isoformat()
                }, f)
            
            self.conn.execute("""
                INSERT OR REPLACE INTO booking_embeddings 
                (booking_id, embedding_hash, created_at, updated_at, feature_version, dimensions, file_path)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (booking_id, embedding_hash, datetime.now(), datetime.now(), 
                  version, len(embedding), str(file_path)))
            self.conn.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving booking embedding: {e}")
            return False
    
    def get_all_room_embeddings(self) -> Dict[int, np.ndarray]:
        """Get all room embeddings for similarity calculations."""
        embeddings = {}
        try:
            cursor = self.conn.execute(
                "SELECT room_id, file_path FROM room_embeddings"
            )
            
            for room_id, file_path in cursor.fetchall():
                try:
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                        embeddings[room_id] = data['embedding']
                except Exception as e:
                    logger.warning(f"Could not load embedding for room {room_id}: {e}")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error getting all room embeddings: {e}")
            return {}
    
    def get_all_user_embeddings(self) -> Dict[int, np.ndarray]:
        """Get all user embeddings for collaborative filtering."""
        embeddings = {}
        try: 
            cursor = self.conn.execute(
                "SELECT user_id, file_path FROM user_embeddings"
            )
            
            for user_id, file_path in cursor.fetchall():
                try:
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                        embeddings[user_id] = data['embedding']
                except Exception as e:
                    logger.warning(f"Could not load embedding for user {user_id}: {e}")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error getting all user embeddings: {e}")
            return {}
    
    def cleanup_old_embeddings(self, days_old: int = 30):
        """Remove embeddings older than specified days."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            # Find old embeddings
            cursor = self.conn.execute("""
                SELECT file_path FROM room_embeddings WHERE created_at < ?
                UNION ALL
                SELECT file_path FROM user_embeddings WHERE created_at < ?
                UNION ALL  
                SELECT file_path FROM booking_embeddings WHERE created_at < ?
            """, (cutoff_date, cutoff_date, cutoff_date))
            
            old_files = cursor.fetchall()
            
            # Delete files and database records
            for (file_path,) in old_files:
                try:
                    Path(file_path).unlink(missing_ok=True)
                except Exception as e:
                    logger.warning(f"Could not delete file {file_path}: {e}")
            
            # Clean up database records
            self.conn.execute("DELETE FROM room_embeddings WHERE created_at < ?", (cutoff_date,))
            self.conn.execute("DELETE FROM user_embeddings WHERE created_at < ?", (cutoff_date,))
            self.conn.execute("DELETE FROM booking_embeddings WHERE created_at < ?", (cutoff_date,))
            self.conn.commit()
            
            logger.info(f"Cleaned up {len(old_files)} old embedding files")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get statistics about stored embeddings."""
        try:
            stats = {}
            
            # Count embeddings by type
            cursor = self.conn.execute("SELECT COUNT(*) FROM room_embeddings")
            stats['room_embeddings'] = cursor.fetchone()[0]
            
            cursor = self.conn.execute("SELECT COUNT(*) FROM user_embeddings")
            stats['user_embeddings'] = cursor.fetchone()[0]
            
            cursor = self.conn.execute("SELECT COUNT(*) FROM booking_embeddings")
            stats['booking_embeddings'] = cursor.fetchone()[0]
            
            # Get total disk usage
            total_size = 0
            for path in [self.room_embeddings_path, self.user_embeddings_path, self.booking_embeddings_path]:
                for file_path in path.rglob("*.pkl"):
                    total_size += file_path.stat().st_size
            
            stats['total_disk_usage_mb'] = total_size / (1024 * 1024)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting embedding stats: {e}")
            return {}
    
    def close(self):
        """Close database connection."""
        if hasattr(self, 'conn'):
            self.conn.close()
