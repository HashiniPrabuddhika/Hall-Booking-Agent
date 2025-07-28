# data/models/model_manager.py
import os
import json
import pickle
import joblib
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
import logging
import sqlite3
import hashlib
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class ModelMetadata:
    """Metadata for stored ML models."""
    model_id: str
    model_type: str
    version: str
    created_at: str
    updated_at: str
    performance_metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    training_data_hash: str
    file_path: str
    file_size_mb: float
    description: str = ""


class ModelManager:
    """Manages storage and versioning of trained ML models."""
    
    def __init__(self, base_path: str = "data/models"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize model type directories
        self.clustering_path = self.base_path / "clustering"
        self.embedding_path = self.base_path / "embedding"
        self.time_series_path = self.base_path / "time_series"
        self.collaborative_path = self.base_path / "collaborative"
        self.metadata_path = self.base_path / "metadata"
        
        for path in [self.clustering_path, self.embedding_path, 
                    self.time_series_path, self.collaborative_path, self.metadata_path]:
            path.mkdir(exist_ok=True)
        
        # Initialize metadata database
        self._init_metadata_db()
    
    def _init_metadata_db(self):
        """Initialize SQLite database for model metadata."""
        db_path = self.metadata_path / "models_metadata.db"
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS models (
                model_id TEXT PRIMARY KEY,
                model_type TEXT NOT NULL,
                version TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                performance_metrics TEXT,
                hyperparameters TEXT,
                training_data_hash TEXT,
                file_path TEXT NOT NULL,
                file_size_mb REAL,
                description TEXT,
                is_active BOOLEAN DEFAULT 1
            );
            
            CREATE TABLE IF NOT EXISTS model_versions (
                model_type TEXT,
                version TEXT,
                model_id TEXT,
                created_at TIMESTAMP,
                is_latest BOOLEAN DEFAULT 0,
                PRIMARY KEY (model_type, version),
                FOREIGN KEY (model_id) REFERENCES models (model_id)
            );
            
            CREATE TABLE IF NOT EXISTS training_logs (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT,
                training_started TIMESTAMP,
                training_completed TIMESTAMP,
                status TEXT,
                error_message TEXT,
                FOREIGN KEY (model_id) REFERENCES models (model_id)
            );
        """)
        self.conn.commit()
    
    def save_model(self, model: Any, model_type: str, version: str,
                   performance_metrics: Dict[str, float],
                   hyperparameters: Dict[str, Any],
                   training_data_hash: str,
                   description: str = "") -> str:
        """Save a trained model with metadata."""
        try:
            # Generate unique model ID
            model_id = f"{model_type}_{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Determine file path based on model type
            if model_type == "clustering":
                file_path = self.clustering_path / f"{model_id}.pkl"
            elif model_type == "embedding":
                file_path = self.embedding_path / f"{model_id}.pkl"
            elif model_type == "time_series":
                file_path = self.time_series_path / f"{model_id}.pkl"
            elif model_type == "collaborative":
                file_path = self.collaborative_path / f"{model_id}.pkl"
            else:
                file_path = self.base_path / f"{model_id}.pkl"
            
            # Save model using joblib for better scikit-learn compatibility
            if hasattr(model, 'fit'):  # sklearn-like models
                joblib.dump(model, file_path)
            else:  # other models
                with open(file_path, 'wb') as f:
                    pickle.dump(model, f)
            
            # Get file size
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            
            # Create metadata
            metadata = ModelMetadata(
                model_id=model_id,
                model_type=model_type,
                version=version,
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
                performance_metrics=performance_metrics,
                hyperparameters=hyperparameters,
                training_data_hash=training_data_hash,
                file_path=str(file_path),
                file_size_mb=file_size_mb,
                description=description
            )
            
            # Save to database
            self.conn.execute("""
                INSERT INTO models 
                (model_id, model_type, version, created_at, updated_at, 
                 performance_metrics, hyperparameters, training_data_hash, 
                 file_path, file_size_mb, description)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metadata.model_id, metadata.model_type, metadata.version,
                metadata.created_at, metadata.updated_at,
                json.dumps(metadata.performance_metrics),
                json.dumps(metadata.hyperparameters),
                metadata.training_data_hash, metadata.file_path,
                metadata.file_size_mb, metadata.description
            ))
            
            # Update version tracking
            self.conn.execute("""
                INSERT OR REPLACE INTO model_versions 
                (model_type, version, model_id, created_at, is_latest)
                VALUES (?, ?, ?, ?, 1)
            """, (model_type, version, model_id, datetime.now()))
            
            # Mark previous versions as not latest
            self.conn.execute("""
                UPDATE model_versions 
                SET is_latest = 0 
                WHERE model_type = ? AND version != ? AND model_id != ?
            """, (model_type, version, model_id))
            
            self.conn.commit()
            
            logger.info(f"Saved model {model_id} of type {model_type}")
            return model_id
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, model_id: str) -> Optional[Any]:
        """Load a model by its ID."""
        try:
            cursor = self.conn.execute(
                "SELECT file_path FROM models WHERE model_id = ? AND is_active = 1",
                (model_id,)
            )
            result = cursor.fetchone()
            
            if not result:
                logger.warning(f"Model {model_id} not found")
                return None
            
            file_path = Path(result[0])
            if not file_path.exists():
                logger.error(f"Model file not found: {file_path}")
                return None
            
            # Try joblib first, then pickle
            try:
                return joblib.load(file_path)
            except:
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
                    
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {e}")
            return None
    
    def load_latest_model(self, model_type: str) -> Optional[Any]:
        """Load the latest version of a model type."""
        try:
            cursor = self.conn.execute("""
                SELECT m.model_id, m.file_path 
                FROM models m
                JOIN model_versions mv ON m.model_id = mv.model_id
                WHERE m.model_type = ? AND mv.is_latest = 1 AND m.is_active = 1
                ORDER BY m.created_at DESC
                LIMIT 1
            """, (model_type,))
            
            result = cursor.fetchone()
            if not result:
                logger.warning(f"No latest model found for type {model_type}")
                return None
            
            model_id, file_path = result
            file_path = Path(file_path)
            
            if not file_path.exists():
                logger.error(f"Model file not found: {file_path}")
                return None
            
            try:
                return joblib.load(file_path)
            except:
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
                    
        except Exception as e:
            logger.error(f"Error loading latest model for type {model_type}: {e}")
            return None
    
    def get_model_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """Get metadata for a specific model."""
        try:
            cursor = self.conn.execute("""
                SELECT model_id, model_type, version, created_at, updated_at,
                       performance_metrics, hyperparameters, training_data_hash,
                       file_path, file_size_mb, description
                FROM models WHERE model_id = ?
            """, (model_id,))
            
            result = cursor.fetchone()
            if not result:
                return None
                
            return ModelMetadata(
                model_id=result[0],
                model_type=result[1],
                version=result[2],
                created_at=result[3],
                updated_at=result[4],
                performance_metrics=json.loads(result[5] or "{}"),
                hyperparameters=json.loads(result[6] or "{}"),
                training_data_hash=result[7],
                file_path=result[8],
                file_size_mb=result[9],
                description=result[10] or ""
            )
            
        except Exception as e:
            logger.error(f"Error getting model metadata: {e}")
            return None
    
    def list_models(self, model_type: Optional[str] = None) -> List[ModelMetadata]:
        """List all models, optionally filtered by type."""
        try:
            if model_type:
                cursor = self.conn.execute("""
                    SELECT model_id, model_type, version, created_at, updated_at,
                           performance_metrics, hyperparameters, training_data_hash,
                           file_path, file_size_mb, description
                    FROM models WHERE model_type = ? AND is_active = 1
                    ORDER BY created_at DESC
                """, (model_type,))
            else:
                cursor = self.conn.execute("""
                    SELECT model_id, model_type, version, created_at, updated_at,
                           performance_metrics, hyperparameters, training_data_hash,
                           file_path, file_size_mb, description
                    FROM models WHERE is_active = 1
                    ORDER BY created_at DESC
                """)
            
            models = []
            for row in cursor.fetchall():
                models.append(ModelMetadata(
                    model_id=row[0],
                    model_type=row[1],
                    version=row[2],
                    created_at=row[3],
                    updated_at=row[4],
                    performance_metrics=json.loads(row[5] or "{}"),
                    hyperparameters=json.loads(row[6] or "{}"),
                    training_data_hash=row[7],
                    file_path=row[8],
                    file_size_mb=row[9],
                    description=row[10] or ""
                ))
            
            return models
            
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    def delete_model(self, model_id: str) -> bool:
        """Soft delete a model (mark as inactive)."""
        try:
            # Mark as inactive instead of deleting
            self.conn.execute(
                "UPDATE models SET is_active = 0 WHERE model_id = ?",
                (model_id,)
            )
            self.conn.commit()
            
            logger.info(f"Marked model {model_id} as inactive")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting model {model_id}: {e}")
            return False
    
    def cleanup_old_models(self, days_old: int = 90, keep_latest: int = 3):
        """Clean up old models, keeping the latest N versions of each type."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            # Get model types
            cursor = self.conn.execute("SELECT DISTINCT model_type FROM models")
            model_types = [row[0] for row in cursor.fetchall()]
            
            for model_type in model_types:
                # Get models to keep (latest N)
                cursor = self.conn.execute("""
                    SELECT model_id FROM models 
                    WHERE model_type = ? AND is_active = 1
                    ORDER BY created_at DESC 
                    LIMIT ?
                """, (model_type, keep_latest))
                
                keep_models = {row[0] for row in cursor.fetchall()}
                
                # Find old models to delete
                cursor = self.conn.execute("""
                    SELECT model_id, file_path FROM models 
                    WHERE model_type = ? AND created_at < ? AND is_active = 1
                """, (model_type, cutoff_date))
                
                for model_id, file_path in cursor.fetchall():
                    if model_id not in keep_models:
                        # Delete file
                        try:
                            Path(file_path).unlink(missing_ok=True)
                        except Exception as e:
                            logger.warning(f"Could not delete model file {file_path}: {e}")
                        
                        # Mark as inactive
                        self.conn.execute(
                            "UPDATE models SET is_active = 0 WHERE model_id = ?",
                            (model_id,)
                        )
            
            self.conn.commit()
            logger.info("Completed model cleanup")
            
        except Exception as e:
            logger.error(f"Error during model cleanup: {e}")
    
    def get_model_performance_history(self, model_type: str) -> List[Dict[str, Any]]:
        """Get performance history for a model type."""
        try:
            cursor = self.conn.execute("""
                SELECT version, created_at, performance_metrics 
                FROM models 
                WHERE model_type = ? AND is_active = 1
                ORDER BY created_at ASC
            """, (model_type,))
            
            history = []
            for version, created_at, metrics_json in cursor.fetchall():
                metrics = json.loads(metrics_json or "{}")
                history.append({
                    'version': version,
                    'created_at': created_at,
                    'metrics': metrics
                })
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting performance history: {e}")
            return []
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        try:
            cursor = self.conn.execute("""
                SELECT model_type, COUNT(*), SUM(file_size_mb), AVG(file_size_mb)
                FROM models WHERE is_active = 1
                GROUP BY model_type
            """)
            
            stats = {}
            total_models = 0
            total_size = 0
            
            for model_type, count, total_mb, avg_mb in cursor.fetchall():
                stats[model_type] = {
                    'count': count,
                    'total_size_mb': total_mb or 0,
                    'avg_size_mb': avg_mb or 0
                }
                total_models += count
                total_size += (total_mb or 0)
            
            stats['totals'] = {
                'total_models': total_models,
                'total_size_mb': total_size
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting storage stats: {e}")
            return {}
    
    def close(self):
        """Close database connection."""
        if hasattr(self, 'conn'):
            self.conn.close()


