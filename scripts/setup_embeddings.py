import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy.orm import sessionmaker
from src.database import engine
from src.models import MRBSRoom
from recommendtion.recommendations.models.embedding_model import EmbeddingModel
from recommendtion.recommendations.data.analytics_processor import AnalyticsProcessor

async def initialize_embeddings():
    """Initialize embeddings for all rooms in the database."""
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    
    try:
        embedding_model = EmbeddingModel()
        analytics = AnalyticsProcessor(db)
        
        # Get all rooms
        rooms = db.query(MRBSRoom).filter(MRBSRoom.disabled == False).all()
        
        print(f"Initializing embeddings for {len(rooms)} rooms...")
        
        for room in rooms:
            print(f"Processing room: {room.room_name}")
            
            # Get room features
            room_features = await analytics.get_room_features(room.room_name)
            
            # Create room data for embedding
            room_data = {
                'name': room.room_name,
                'capacity': room.capacity,
                'description': room.description or '',
                'features': [],  # Could be extracted from description
                'location': f'Area {room.area_id}'
            }
            
            # Store embedding
            embedding_model.store_room_embedding(str(room.id), room_data)
            
        print("Embedding initialization complete!")
        
    finally:
        db.close()

if __name__ == "__main__":
    asyncio.run(initialize_embeddings())
