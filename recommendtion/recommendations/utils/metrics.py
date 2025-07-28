# recommendtion/recommendations/utils/metrics.py
from typing import Dict, Any, List
from datetime import datetime
import json
from .cache_manager import CacheManager

class RecommendationMetrics:
    """Track and analyze recommendation system performance"""
    
    def __init__(self):
        self.cache = CacheManager()
        self.metrics_store = []  # In-memory store for now
    
    def track_recommendation_request(
        self, 
        user_id: str, 
        request_type: str, 
        context: Dict[str, Any]
    ):
        """Track recommendation requests"""
        metric = {
            'timestamp': datetime.now().isoformat(),
            'type': 'request',
            'user_id': user_id,
            'request_type': request_type,
            'context': context
        }
        self.metrics_store.append(metric)
    
    def track_booking_outcome(
        self, 
        user_id: str, 
        booking_data: Dict[str, Any], 
        outcome: str
    ):
        """Track the outcome of recommendations"""
        metric = {
            'timestamp': datetime.now().isoformat(),
            'type': 'outcome',
            'user_id': user_id,
            'booking_data': booking_data,
            'outcome': outcome  # 'success', 'rejected', 'modified'
        }
        self.metrics_store.append(metric)
    
    async def get_recommendation_explanation(
        self, 
        recommendation_id: str
    ) -> Dict[str, Any]:
        """Get explanation for a specific recommendation"""
        # This would provide detailed reasoning for transparency
        return {
            'recommendation_id': recommendation_id,
            'explanation': 'Based on your booking history and similar user patterns',
            'factors': [
                'User booking frequency',
                'Room similarity',
                'Time preferences',
                'Availability patterns'
            ],
            'confidence_breakdown': {
                'user_pattern_match': 0.8,
                'room_similarity': 0.7,
                'time_preference': 0.6,
                'availability': 0.9
            }
        }
    
    def get_system_performance(self) -> Dict[str, Any]:
        """Get overall system performance metrics"""
        total_requests = len([m for m in self.metrics_store if m['type'] == 'request'])
        successful_outcomes = len([m for m in self.metrics_store if m['type'] == 'outcome' and m['outcome'] == 'success'])
        
        success_rate = successful_outcomes / total_requests if total_requests > 0 else 0
        
        return {
            'total_requests': total_requests,
            'successful_recommendations': successful_outcomes,
            'success_rate': success_rate,
            'avg_response_time': 0.5,  # Would be calculated from actual metrics
            'user_satisfaction': 0.85  # Would be calculated from feedback
        }
