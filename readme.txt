 .\venv\Scripts\activate
 uvicorn main:app --reload    
 pip install -r requirements.txt

 
# data/README.md
# Data Storage System

This directory contains the data storage components for the room booking recommendation system.

## Structure

```
data/
├── embeddings/          # Vector embeddings storage
│   ├── rooms/          # Room feature embeddings
│   ├── users/          # User preference embeddings  
│   ├── bookings/       # Booking pattern embeddings
│   └── metadata/       # Embedding metadata database
├── models/             # Trained ML models
│   ├── clustering/     # User/room clustering models
│   ├── embedding/      # Embedding generation models
│   ├── time_series/    # Demand forecasting models
│   ├── collaborative/  # Collaborative filtering models
│   └── metadata/       # Model metadata database
├── cache/              # Cached recommendations
│   ├── recommendations/ # Cached recommendation results
│   ├── user_profiles/  # Cached user profiles
│   ├── room_similarities/ # Cached similarity matrices
│   ├── analytics/      # Cached analytics data
│   └── metadata/       # Cache metadata database
└── analytics/          # Analytics and reporting data
    ├── events/         # Raw event data
    ├── aggregates/     # Aggregated statistics
    ├── reports/        # Generated reports
    └── exports/        # Data exports
```

## Components

### EmbeddingManager
- Stores and retrieves vector embeddings for rooms, users, and booking patterns
- Supports versioning and metadata tracking
- Automatic cleanup of old embeddings
- Efficient similarity search capabilities

### ModelManager  
- Manages trained ML models with versioning
- Supports multiple model formats (pickle, joblib)
- Performance tracking and comparison
- Automatic model lifecycle management

### CacheManager
- High-performance caching with Redis primary and file fallback
- Compression and expiration management
- Cache hit/miss analytics
- User-specific and global cache invalidation

### AnalyticsManager
- Comprehensive event logging and analytics
- Real-time aggregation of booking patterns
- User behavior analysis
- Performance reporting and exports

## Usage Examples

### Embeddings
```python
from data import EmbeddingManager

embedding_mgr = EmbeddingManager()

# Save room embedding
embedding_mgr.save_room_embedding(
    room_id=101,
    embedding=room_vector,
    features={'capacity': 10, 'has_projector': True}
)

# Load user embedding  
user_embedding, preferences = embedding_mgr.load_user_embedding(user_id=1)
```

### Models
```python
from data import ModelManager

model_mgr = ModelManager()

# Save trained model
model_id = model_mgr.save_model(
    model=trained_classifier,
    model_type="clustering", 
    version="v2.1",
    performance_metrics={'accuracy': 0.85, 'f1_score': 0.82},
    hyperparameters={'n_clusters': 5, 'random_state': 42}
)

# Load latest model
latest_model = model_mgr.load_latest_model("clustering")
```

### Caching
```python
from data import CacheManager

cache_mgr = CacheManager()

# Cache recommendations
cache_key = cache_mgr.set_recommendation(
    user_id=1,
    request_type="alternative_rooms",
    params={'date': '2024-03-15', 'time': '14:00'},
    data=recommendations,
    ttl_hours=24
)

# Retrieve cached data
cached_recs = cache_mgr.get_recommendation(
    user_id=1,
    request_type="alternative_rooms", 
    params={'date': '2024-03-15', 'time': '14:00'}
)
```

### Analytics  
```python
from data import AnalyticsManager, BookingEvent

analytics_mgr = AnalyticsManager()

# Log booking event
booking_event = BookingEvent(
    event_id="booking_12345",
    user_id=1,
    room_id=101,
    event_type="booking_created",
    timestamp="2024-03-15T14:30:00",
    booking_date="2024-03-20",
    start_time="09:00",
    end_time="10:00", 
    duration_minutes=60,
    metadata={'source': 'web_app'}
)

analytics_mgr.log_booking_event(booking_event)

# Generate reports
utilization_report = analytics_mgr.get_room_utilization_report(
    start_date=date(2024, 3, 1),
    end_date=date(2024, 3, 31)
)
```

## Configuration

The data storage system can be configured through environment variables:

```bash
# Redis configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Storage paths
DATA_BASE_PATH=/app/data
EMBEDDINGS_PATH=/app/data/embeddings
MODELS_PATH=/app/data/models  
CACHE_PATH=/app/data/cache
ANALYTICS_PATH=/app/data/analytics

# Retention policies
MODEL_RETENTION_DAYS=90
CACHE_TTL_HOURS=24
ANALYTICS_RETENTION_DAYS=365
EMBEDDING_RETENTION_DAYS=30
```

## Performance Considerations

- **Embeddings**: Use memory mapping for large embedding matrices
- **Models**: Lazy loading with joblib for better memory efficiency  
- **Cache**: Redis for hot data, compressed files for cold data
- **Analytics**: Batch processing and aggregation to minimize database load

## Backup and Recovery

Regular backups should include:
- SQLite metadata databases
- Trained model files
- Critical embedding data
- Analytics aggregates

The system supports incremental backups and point-in-time recovery for all components.

