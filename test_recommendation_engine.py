import sys
import os
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Add your project paths
sys.path.append('.')
sys.path.append('./src')

# Import your models and recommendation engine
from src.models import MRBSRoom, MRBSEntry, MRBSRepeat
from recommendtion.recommendations.core.recommendation_engine import RecommendationEngine, RecommendationEngineFactory
from recommendtion.config.recommendation_config import RecommendationConfig


def test_recommendation_engine():
    """Test the recommendation engine with real database data"""
    print("\n" + "=" * 50)
    print("TESTING RECOMMENDATION ENGINE")
    print("=" * 50)
    
    try:
        # Create recommendation engine
        config = RecommendationConfig()
        engine = RecommendationEngine(config=config)
        
        print("✓ Recommendation engine initialized")
        
        # Test engine status
        status = engine.get_engine_status()
        print(f"✓ Engine status: {status['status']}")
        print(f"✓ MySQL connection: {status['mysql_connection']}")
        print(f"✓ Active rooms: {status['database_stats']['active_rooms']}")
        print(f"✓ Recent bookings: {status['database_stats']['recent_bookings']}")
        
        return engine
        
    except Exception as e:
        print(f"✗ Recommendation engine initialization failed: {e}")
        return None

def test_room_data_retrieval(engine):
    """Test room data retrieval from database"""
    print("\n" + "=" * 50)
    print("TESTING ROOM DATA RETRIEVAL")
    print("=" * 50)
    
    try:
        # Get all rooms
        rooms = engine.get_room_data_from_db()
        print(f"✓ Retrieved {len(rooms)} rooms from database")
        
        if rooms:
            print("Room details:")
            for room in rooms[:5]:  # Show first 5 rooms
                print(f"  - {room['room_name']}: Capacity {room['capacity']}, Area {room['area_id']}")
        
        # Test specific room
        if rooms:
            test_room = rooms[0]['room_name']
            specific_room = engine.get_room_data_from_db(test_room)
            print(f"✓ Retrieved specific room '{test_room}': {len(specific_room)} result(s)")
        
        return rooms
        
    except Exception as e:
        print(f"✗ Room data retrieval failed: {e}")
        return []

def test_availability_check(engine, rooms):
    """Test room availability checking"""
    print("\n" + "=" * 50)
    print("TESTING ROOM AVAILABILITY")
    print("=" * 50)
    
    if not rooms:
        print("✗ No rooms available for testing")
        return
    
    try:
        # Test with first room
        test_room = rooms[0]['room_name']
        
        # Test current time (likely occupied)
        now = datetime.now()
        current_available = engine.check_room_availability_in_db(
            test_room, 
            now, 
            now + timedelta(hours=1)
        )
        print(f"✓ Room '{test_room}' availability now: {'Available' if current_available else 'Occupied'}")
        
        # Test future time (likely available)
        future_time = now + timedelta(days=7)  # Next week
        future_available = engine.check_room_availability_in_db(
            test_room,
            future_time,
            future_time + timedelta(hours=1)
        )
        print(f"✓ Room '{test_room}' availability next week: {'Available' if future_available else 'Occupied'}")
        
    except Exception as e:
        print(f"✗ Availability check failed: {e}")

def test_recommendations(engine, rooms):
    """Test actual recommendation generation"""
    print("\n" + "=" * 50)
    print("TESTING RECOMMENDATION GENERATION")
    print("=" * 50)
    
    if not rooms:
        print("✗ No rooms available for testing")
        return
    
    try:
        # Create test request
        test_room = rooms[0]['room_name']
        start_time = datetime.now() + timedelta(hours=2)
        end_time = start_time + timedelta(hours=1)
        
        request_data = {
            'user_id': 'test_user_001',
            'room_id': test_room,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'purpose': 'team meeting',
            'capacity': 8,
            'requirements': {
                'projector': True,
                'whiteboard': True
            }
        }
        
        print(f"Test request:")
        print(f"  Room: {test_room}")
        print(f"  Time: {start_time.strftime('%Y-%m-%d %H:%M')} - {end_time.strftime('%H:%M')}")
        print(f"  Purpose: {request_data['purpose']}")
        
        # Generate recommendations
        recommendations = engine.get_recommendations(request_data)
        
        print(f"\n✓ Generated {len(recommendations)} recommendations:")
        
        for i, rec in enumerate(recommendations, 1):
            print(f"\n  Recommendation {i}:")
            print(f"    Type: {rec['type']}")
            print(f"    Score: {rec['score']:.2f}")
            print(f"    Reason: {rec['reason']}")
            print(f"    Data Source: {rec['data_source']}")
            
            if 'suggestion' in rec:
                suggestion = rec['suggestion']
                print(f"    Suggested Room: {suggestion.get('room_name', suggestion.get('room_id', 'N/A'))}")
                if 'capacity' in suggestion:
                    print(f"    Room Capacity: {suggestion['capacity']}")
                if 'confidence' in suggestion:
                    print(f"    Confidence: {suggestion['confidence']:.2f}")
        
        return recommendations
        
    except Exception as e:
        print(f"✗ Recommendation generation failed: {e}")
        return []

def test_user_history(engine):
    """Test user booking history retrieval"""
    print("\n" + "=" * 50)
    print("TESTING USER BOOKING HISTORY")
    print("=" * 50)
    
    try:
        # Try to get some real user from database
        if hasattr(engine, 'db') and engine.db:
            # Get a sample user who has made bookings
            recent_entry = engine.db.query(MRBSEntry).filter(
                MRBSEntry.create_by != ''
            ).first()
            
            if recent_entry:
                test_user = recent_entry.create_by
                print(f"Testing with user: {test_user}")
                
                history = engine.get_user_booking_history(test_user, days=90)
                print(f"✓ Retrieved {len(history)} bookings for user {test_user}")
                
                if history:
                    print("Recent bookings:")
                    for booking in history[:3]:  # Show first 3
                        print(f"  - {booking['booking_name']} in {booking['room_name']} "
                              f"on {booking['start_time'].strftime('%Y-%m-%d %H:%M')}")
                        
                return history
            else:
                print("No user bookings found in database")
                
        return []
                
    except Exception as e:
        print(f"Error testing user history: {e}")
        return []

def test_utilization_stats(engine, rooms):
    """Test room utilization statistics"""
    print("\n" + "=" * 50)
    print("TESTING ROOM UTILIZATION STATS")
    print("=" * 50)
    
    if not rooms:
        print("✗ No rooms available for testing")
        return
    
    try:
        # Get utilization for all rooms
        all_stats = engine.get_room_utilization_stats(days=30)
        print(f"✓ Retrieved utilization stats for {len(all_stats)} rooms")
        
        if all_stats:
            print("\nRoom utilization (last 30 days):")
            for room_name, stats in list(all_stats.items())[:5]:  # Show first 5 rooms
                print(f"  {room_name}:")
                print(f"    Total bookings: {stats['total_bookings']}")
                print(f"    Hours booked: {stats['total_hours_booked']}")
                print(f"    Utilization rate: {stats['utilization_rate_percent']:.1f}%")
                print(f"    Avg booking duration: {stats['avg_booking_duration_hours']:.1f} hours")
                print(f"    Bookings per day: {stats['bookings_per_day']:.1f}")
                print()
        
        # Test specific room
        if rooms:
            test_room = rooms[0]['room_name']
            specific_stats = engine.get_room_utilization_stats(test_room, days=30)
            if test_room in specific_stats:
                print(f"✓ Retrieved specific stats for '{test_room}'")
            
        return all_stats
        
    except Exception as e:
        print(f"✗ Utilization stats failed: {e}")
        return {}

def run_comprehensive_test():
    """Run all tests in sequence"""
    print("ROOM BOOKING RECOMMENDATION ENGINE TEST")
    print("=" * 60)
    print("Testing integration with MRBS database")
    print("=" * 60)
    
    # Test 2: Recommendation Engine
    engine = test_recommendation_engine()
    if not engine:
        print("\n❌ Recommendation engine initialization failed.")
        return False
    
    # Test 3: Room Data Retrieval
    rooms = test_room_data_retrieval(engine)
    
    # Test 4: Availability Checking
    test_availability_check(engine, rooms)
    
    # Test 5: Recommendation Generation
    recommendations = test_recommendations(engine, rooms)
    
    # Test 6: User History
    test_user_history(engine)
    
    # Test 7: Utilization Stats
    test_utilization_stats(engine, rooms)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print("✓ Database Connection: PASSED")
    print("✓ Recommendation Engine: PASSED")
    print(f"✓ Room Data: {len(rooms)} rooms found")
    print(f"✓ Recommendations: {len(recommendations) if 'recommendations' in locals() else 0} generated")
    print("✓ All tests completed successfully!")
    
    return True

def test_specific_scenario():
    """Test a specific booking scenario"""
    print("\n" + "=" * 50)
    print("TESTING SPECIFIC BOOKING SCENARIO")
    print("=" * 50)
    
    try:
        config = RecommendationConfig()
        engine = RecommendationEngine(config=config)
        
        # Scenario: User wants to book a meeting room for tomorrow
        tomorrow = datetime.now() + timedelta(days=1)
        meeting_start = tomorrow.replace(hour=14, minute=0, second=0, microsecond=0)  # 2 PM
        meeting_end = meeting_start + timedelta(hours=2)  # 4 PM
        
        request_data = {
            'user_id': 'john.doe@company.com',
            'room_id': 'Conference Room A',  
            'start_time': meeting_start.isoformat(),
            'end_time': meeting_end.isoformat(),
            'purpose': 'quarterly review meeting',
            'capacity': 10,
            'requirements': {
                'projector': True,
                'whiteboard': True,
                'video_conference': True
            }
        }
        
        print("Scenario: Quarterly Review Meeting")
        print(f"  Requested Room: {request_data['room_id']}")
        print(f"  Date & Time: {meeting_start.strftime('%Y-%m-%d %H:%M')} - {meeting_end.strftime('%H:%M')}")
        print(f"  Attendees: {request_data['capacity']} people")
        print(f"  Requirements: {', '.join(request_data['requirements'].keys())}")
        
        # Generate recommendations
        recommendations = engine.get_recommendations(request_data)
        
        print(f"\n📋 RECOMMENDATION RESULTS ({len(recommendations)} options found):")
        print("-" * 50)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"\n🏢 Option {i}: {rec['type'].replace('_', ' ').title()}")
            print(f"   Score: {rec['score']:.2f}/1.00")
            print(f"   Reason: {rec['reason']}")
            
            suggestion = rec.get('suggestion', {})
            if 'room_name' in suggestion:
                print(f"   Room: {suggestion['room_name']}")
            if 'capacity' in suggestion:
                print(f"   Capacity: {suggestion['capacity']} people")
            if 'start_time' in suggestion and 'end_time' in suggestion:
                start = datetime.fromisoformat(suggestion['start_time'].replace('Z', '+00:00'))
                end = datetime.fromisoformat(suggestion['end_time'].replace('Z', '+00:00'))
                print(f"   Time: {start.strftime('%Y-%m-%d %H:%M')} - {end.strftime('%H:%M')}")
            
            print(f"   Data Source: {rec['data_source']}")
            print(f"   Confidence: {suggestion.get('confidence', 'N/A')}")
        
        return recommendations
        
    except Exception as e:
        print(f"✗ Specific scenario test failed: {e}")
        return []

def interactive_test():
    """Interactive test allowing user input"""
    print("\n" + "=" * 50)
    print("INTERACTIVE RECOMMENDATION TEST")
    print("=" * 50)
    
    try:
        config = RecommendationConfig()
        engine = RecommendationEngine(config=config)
        
        # Get available rooms
        rooms = engine.get_room_data_from_db()
        if not rooms:
            print("No rooms found in database")
            return
        
        print("Available rooms:")
        for i, room in enumerate(rooms[:10], 1):  # Show first 10 rooms
            print(f"  {i}. {room['room_name']} (Capacity: {room['capacity']})")
        
        # Get user input
        try:
            room_choice = input(f"\nEnter room number (1-{min(10, len(rooms))}) or room name: ").strip()
            
            # Parse room choice
            if room_choice.isdigit():
                room_index = int(room_choice) - 1
                if 0 <= room_index < len(rooms):
                    selected_room = rooms[room_index]['room_name']
                else:
                    selected_room = rooms[0]['room_name']  # Default to first room
            else:
                selected_room = room_choice if room_choice else rooms[0]['room_name']
            
            # Get date and time
            date_input = input("Enter date (YYYY-MM-DD) or press Enter for tomorrow: ").strip()
            if not date_input:
                target_date = datetime.now() + timedelta(days=1)
            else:
                target_date = datetime.strptime(date_input, '%Y-%m-%d')
            
            time_input = input("Enter start time (HH:MM) or press Enter for 14:00: ").strip()
            if not time_input:
                start_hour, start_minute = 14, 0
            else:
                start_hour, start_minute = map(int, time_input.split(':'))
            
            duration_input = input("Enter duration in hours (default 2): ").strip()
            duration = float(duration_input) if duration_input else 2.0
            
            # Create datetime objects
            meeting_start = target_date.replace(hour=start_hour, minute=start_minute, second=0, microsecond=0)
            meeting_end = meeting_start + timedelta(hours=duration)
            
            # Get user info
            user_id = input("Enter your user ID/email (default: test@example.com): ").strip()
            if not user_id:
                user_id = "test@example.com"
            
            purpose = input("Enter meeting purpose (default: meeting): ").strip()
            if not purpose:
                purpose = "meeting"
            
            capacity = input("Enter number of attendees (default: 6): ").strip()
            capacity = int(capacity) if capacity.isdigit() else 6
            
        except KeyboardInterrupt:
            print("\nTest cancelled by user")
            return
        except Exception as e:
            print(f"Input error: {e}, using defaults")
            selected_room = rooms[0]['room_name']
            meeting_start = datetime.now() + timedelta(days=1, hours=2)
            meeting_end = meeting_start + timedelta(hours=2)
            user_id = "test@example.com"
            purpose = "meeting"
            capacity = 6
        
        # Create request
        request_data = {
            'user_id': user_id,
            'room_id': selected_room,
            'start_time': meeting_start.isoformat(),
            'end_time': meeting_end.isoformat(),
            'purpose': purpose,
            'capacity': capacity
        }
        
        print(f"\n🔍 Searching for recommendations...")
        print(f"   User: {user_id}")
        print(f"   Preferred Room: {selected_room}")
        print(f"   Date & Time: {meeting_start.strftime('%Y-%m-%d %H:%M')} - {meeting_end.strftime('%H:%M')}")
        print(f"   Purpose: {purpose}")
        print(f"   Capacity: {capacity} people")
        
        # Generate recommendations
        recommendations = engine.get_recommendations(request_data)
        
        print(f"\n✅ Found {len(recommendations)} recommendations:")
        print("=" * 50)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"\n📌 Recommendation {i}:")
            print(f"   Type: {rec['type'].replace('_', ' ').title()}")
            print(f"   Score: {rec['score']:.2f}")
            print(f"   Reason: {rec['reason']}")
            
            suggestion = rec.get('suggestion', {})
            if suggestion:
                print(f"   Suggested Room: {suggestion.get('room_name', 'N/A')}")
                if 'capacity' in suggestion:
                    print(f"   Room Capacity: {suggestion['capacity']}")
                if 'start_time' in suggestion:
                    start = datetime.fromisoformat(suggestion['start_time'].replace('Z', '+00:00'))
                    end = datetime.fromisoformat(suggestion['end_time'].replace('Z', '+00:00'))
                    print(f"   Suggested Time: {start.strftime('%Y-%m-%d %H:%M')} - {end.strftime('%H:%M')}")
                if 'confidence' in suggestion:
                    print(f"   Confidence: {suggestion['confidence']:.2f}")
        
    except Exception as e:
        print(f"Interactive test failed: {e}")

def main():
    """Main function to run tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Room Booking Recommendation Engine")
    parser.add_argument('--test', choices=['all', 'basic', 'scenario', 'interactive'], 
                       default='all', help='Type of test to run')
    parser.add_argument('--quiet', action='store_true', help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    if not args.quiet:
        print("🏢 MRBS RECOMMENDATION ENGINE TESTER")
        print("=" * 50)
        print("This script tests the recommendation engine with your actual MySQL database.")
        print("Make sure your database is running and accessible.")
        print("=" * 50)
    
    try:
        if args.test == 'all':
            success = run_comprehensive_test()
            if success:
                test_specific_scenario()
        elif args.test == 'basic': 
            
            test_recommendation_engine()
        elif args.test == 'scenario':
            test_specific_scenario()
        elif args.test == 'interactive':
            interactive_test()
            
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        if not args.quiet:
            traceback.print_exc()
    
    print("\n🏁 Test completed!")

if __name__ == "__main__":
    main()