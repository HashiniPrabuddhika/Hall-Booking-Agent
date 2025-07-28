# tests/test_recommendations/test_recommendation_system.py
import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from sqlalchemy.orm import Session
from typing import List, Dict, Any

from src.models import MRBSRoom, MRBSEntry, MRBSRepeat
from recommendtion.recommendations.core.recommendation_engine import RecommendationEngine
from recommendtion.recommendations.strategies.alternative_time import AlternativeTimeStrategy
from recommendtion.recommendations.strategies.alternative_room import AlternativeRoomStrategy
from recommendtion.recommendations.models.embedding_model import EmbeddingModel
from recommendtion.recommendations.data.analytics_processor import AnalyticsProcessor

class TestRecommendationSystem:
    """Comprehensive test suite for the recommendation system"""
    
    @pytest.fixture
    def sample_rooms(self) -> List[MRBSRoom]:
        """Create sample rooms for testing"""
        return [
            MRBSRoom(
                id=1, room_name="Conference Room A", capacity=10,
                description="Small meeting room with projector"
            ),
            MRBSRoom(
                id=2, room_name="Conference Room B", capacity=20,
                description="Large meeting room with video conferencing"
            ),
            MRBSRoom(
                id=3, room_name="Training Room", capacity=30,
                description="Training room with whiteboards"
            )
        ]
    
    @pytest.fixture
    def sample_bookings(self) -> List[MRBSEntry]:
        """Create sample bookings for testing"""
        now = int(datetime.now().timestamp())
        return [
            MRBSEntry(
                id=1, room_id=1, start_time=now + 3600, end_time=now + 7200,
                create_by="user1", name="Team Meeting",
                description="Weekly team sync"
            ),
            MRBSEntry(
                id=2, room_id=2, start_time=now + 1800, end_time=now + 5400,
                create_by="user2", name="Client Presentation",
                description="Q4 results presentation"
            )
        ]
    
    @pytest.fixture
    def recommendation_engine(self, sample_rooms, sample_bookings):
        """Initialize recommendation engine with test data"""
        with patch('src.recommendations.core.recommendation_engine.get_db') as mock_db:
            mock_session = Mock(spec=Session)
            mock_session.query.return_value.all.return_value = sample_rooms
            mock_db.return_value = mock_session
            
            engine = RecommendationEngine()
            return engine

class TestCoreRecommendationEngine:
    """Test the core recommendation engine functionality"""
    
    def test_engine_initialization(self, recommendation_engine):
        """Test if recommendation engine initializes correctly"""
        assert recommendation_engine is not None
        assert hasattr(recommendation_engine, 'get_recommendations')
        assert hasattr(recommendation_engine, 'pattern_analyzer')
        assert hasattr(recommendation_engine, 'similarity_engine')
    
    @pytest.mark.asyncio
    async def test_basic_recommendations(self, recommendation_engine):
        """Test basic recommendation generation"""
        request = {
            'user_id': 'test_user',
            'room_preferences': ['Conference Room A'],
            'time_slot': {'start': '2024-01-15T10:00:00', 'end': '2024-01-15T11:00:00'},
            'meeting_type': 'team_meeting'
        }
        
        recommendations = await recommendation_engine.get_recommendations(request)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Check recommendation structure
        for rec in recommendations:
            assert 'room_id' in rec
            assert 'confidence_score' in rec
            assert 'reason' in rec
            assert 0 <= rec['confidence_score'] <= 1
    
    def test_fallback_recommendations(self, recommendation_engine):
        """Test fallback when no ML recommendations available"""
        with patch.object(recommendation_engine, '_get_ml_recommendations', return_value=[]):
            request = {'user_id': 'new_user', 'room_preferences': []}
            
            recommendations = recommendation_engine.get_recommendations(request)
            
            # Should still return some recommendations (rule-based fallback)
            assert len(recommendations) > 0

class TestAlternativeStrategies:
    """Test alternative time/room recommendation strategies"""
    
    def test_alternative_time_suggestions(self, sample_bookings):
        """Test alternative time slot recommendations"""
        strategy = AlternativeTimeStrategy()
        
        # Simulate conflict scenario
        requested_time = {
            'start': sample_bookings[0].start_time,
            'end': sample_bookings[0].end_time,
            'room_id': sample_bookings[0].room_id
        }
        
        alternatives = strategy.get_alternatives(requested_time, sample_bookings)
        
        assert isinstance(alternatives, list)
        assert len(alternatives) > 0
        
        # Check that alternatives don't conflict
        for alt in alternatives:
            assert alt['start_time'] != requested_time['start']
            assert alt['room_id'] == requested_time['room_id']
    
    def test_alternative_room_suggestions(self, sample_rooms, sample_bookings):
        """Test alternative room recommendations"""
        strategy = AlternativeRoomStrategy()
        
        request = {
            'original_room_id': 1,
            'capacity_needed': 10,
            'time_slot': {'start': sample_bookings[0].start_time, 'end': sample_bookings[0].end_time}
        }
        
        alternatives = strategy.get_alternatives(request, sample_rooms, sample_bookings)
        
        assert isinstance(alternatives, list)
        
        # Check room alternatives meet capacity requirements
        for alt in alternatives:
            assert alt['room_id'] != request['original_room_id']
            # Should have adequate capacity
            room = next(r for r in sample_rooms if r.id == alt['room_id'])
            assert room.capacity >= request['capacity_needed']

class TestMLModels:
    """Test machine learning model components"""
    
    @pytest.mark.asyncio
    async def test_embedding_model(self):
        """Test embedding model functionality"""
        embedding_model = EmbeddingModel()
        
        # Test room description embedding
        room_descriptions = [
            "Small meeting room with projector",
            "Large conference room with video equipment"
        ]
        
        embeddings = await embedding_model.get_embeddings(room_descriptions)
        
        assert len(embeddings) == len(room_descriptions)
        assert all(isinstance(emb, list) for emb in embeddings)
        assert all(len(emb) > 0 for emb in embeddings)
    
    def test_similarity_calculation(self):
        """Test similarity calculation between rooms/users"""
        from recommendtion.recommendations.core.similarity_engine import SimilarityEngine
        
        similarity_engine = SimilarityEngine()
        
        # Mock embedding vectors
        room1_vector = [0.1, 0.2, 0.3, 0.4]
        room2_vector = [0.2, 0.3, 0.4, 0.5]
        
        similarity = similarity_engine.calculate_similarity(room1_vector, room2_vector)
        
        assert 0 <= similarity <= 1
        assert isinstance(similarity, float)

class TestDataProcessing:
    """Test data processing and analytics components"""
    
    def test_booking_analytics(self, sample_bookings):
        """Test booking pattern analysis"""
        processor = AnalyticsProcessor()
        
        analytics = processor.analyze_booking_patterns(sample_bookings)
        
        assert 'peak_hours' in analytics
        assert 'popular_rooms' in analytics
        assert 'user_patterns' in analytics
        assert isinstance(analytics['peak_hours'], list)
    
    def test_feature_extraction(self, sample_bookings):
        """Test feature extraction from booking data"""
        from recommendtion.recommendations.data.feature_extractor import FeatureExtractor
        
        extractor = FeatureExtractor()
        features = extractor.extract_booking_features(sample_bookings[0])
        
        assert 'time_of_day' in features
        assert 'day_of_week' in features
        assert 'duration' in features
        assert 'room_capacity' in features

class TestPerformanceMetrics:
    """Test recommendation system performance metrics"""
    
    def test_recommendation_accuracy(self):
        """Test recommendation accuracy measurement"""
        from recommendtion.recommendations.utils.metrics import RecommendationMetrics
        
        metrics = RecommendationMetrics()
        
        # Mock actual vs predicted recommendations
        actual_bookings = [{'room_id': 1, 'user_id': 'user1'}]
        recommended = [
            {'room_id': 1, 'confidence_score': 0.8},
            {'room_id': 2, 'confidence_score': 0.6}
        ]
        
        accuracy = metrics.calculate_accuracy(actual_bookings, recommended)
        
        assert 0 <= accuracy <= 1
        assert isinstance(accuracy, float)
    
    def test_response_time_measurement(self):
        """Test recommendation response time"""
        import time
        from recommendtion.recommendations.utils.metrics import RecommendationMetrics
        
        metrics = RecommendationMetrics()
        
        start_time = time.time()
        # Simulate recommendation generation
        time.sleep(0.1)
        response_time = metrics.measure_response_time(start_time)
        
        assert response_time >= 0.1
        assert isinstance(response_time, float)

class TestIntegrationScenarios:
    """End-to-end integration tests"""
    
    @pytest.mark.asyncio
    async def test_full_recommendation_flow(self, recommendation_engine, sample_rooms, sample_bookings):
        """Test complete recommendation flow"""
        # Simulate API request
        request_data = {
            'user_id': 'test_user',
            'meeting_title': 'Team Standup',
            'description': 'Daily team meeting',
            'duration': 30,  # minutes
            'preferred_time': '2024-01-15T10:00:00',
            'attendees': 5,
            'equipment_needed': ['projector']
        }
        
        # Process through recommendation engine
        recommendations = await recommendation_engine.get_recommendations(request_data)
        
        # Validate response
        assert len(recommendations) > 0
        assert all('room_id' in rec for rec in recommendations)
        assert all('confidence_score' in rec for rec in recommendations)
        assert all('available_times' in rec for rec in recommendations)
    
    def test_concurrent_recommendations(self, recommendation_engine):
        """Test system under concurrent load"""
        import concurrent.futures
        
        def get_recommendation(user_id):
            request = {
                'user_id': f'user_{user_id}',
                'meeting_type': 'meeting',
                'duration': 60
            }
            return recommendation_engine.get_recommendations(request)
        
        # Simulate concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(get_recommendation, i) for i in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should complete successfully
        assert len(results) == 10
        assert all(isinstance(result, list) for result in results)

# Utility functions for testing
class TestDataGenerator:
    """Generate test data for comprehensive testing"""
    
    @staticmethod
    def generate_booking_history(num_bookings: int = 100) -> List[Dict[str, Any]]:
        """Generate synthetic booking history for testing"""
        import random
        from datetime import datetime, timedelta
        
        bookings = []
        base_time = datetime.now()
        
        for i in range(num_bookings):
            booking = {
                'id': i + 1,
                'user_id': f'user_{random.randint(1, 20)}',
                'room_id': random.randint(1, 5),
                'start_time': base_time + timedelta(days=random.randint(-30, 30), 
                                                 hours=random.randint(8, 18)),
                'duration': random.choice([30, 60, 90, 120]),
                'meeting_type': random.choice(['team_meeting', 'client_call', 'training', 'presentation']),
                'attendees': random.randint(2, 15)
            }
            bookings.append(booking)
        
        return bookings
    
    @staticmethod
    def create_test_scenarios() -> List[Dict[str, Any]]:
        """Create various test scenarios"""
        return [
            {
                'name': 'Peak Hour Conflict',
                'request': {
                    'time': '2024-01-15T14:00:00',  # Peak hour
                    'duration': 60,
                    'attendees': 10
                },
                'expected': 'alternative_time_suggestions'
            },
            {
                'name': 'Large Meeting',
                'request': {
                    'attendees': 50,
                    'duration': 120,
                    'equipment': ['projector', 'video_conf']
                },
                'expected': 'large_room_recommendation'
            },
            {
                'name': 'New User',
                'request': {
                    'user_id': 'brand_new_user',
                    'meeting_type': 'team_meeting'
                },
                'expected': 'fallback_to_popular_rooms'
            }
        ]

# Performance benchmarking
class TestPerformanceBenchmarks:
    """Performance and load testing"""
    
    def test_recommendation_latency(self, recommendation_engine):
        """Test recommendation generation latency"""
        import time
        
        request = {
            'user_id': 'test_user',
            'meeting_type': 'meeting',
            'duration': 60
        }
        
        latencies = []
        for _ in range(100):
            start = time.time()
            recommendation_engine.get_recommendations(request)
            latencies.append(time.time() - start)
        
        avg_latency = sum(latencies) / len(latencies)
        p95_latency = sorted(latencies)[95]
        
        # Assert reasonable performance
        assert avg_latency < 0.5  # Average < 500ms
        assert p95_latency < 1.0   # 95th percentile < 1s
    
    def test_memory_usage(self, recommendation_engine):
        """Test memory usage during recommendations"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Generate many recommendations
        for i in range(100):
            request = {'user_id': f'user_{i}', 'meeting_type': 'meeting'}
            recommendation_engine.get_recommendations(request)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 100MB)
        assert memory_increase < 100 * 1024 * 1024

if __name__ == "__main__":
    # Run specific test categories
    pytest.main([
        "tests/test_recommendations/",
        "-v",
        "--tb=short",
        "--cov=recommendtion/recommendations",
        "--cov-report=html"
    ])