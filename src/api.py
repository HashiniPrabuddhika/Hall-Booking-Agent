# from fastapi import APIRouter, HTTPException, Depends
# from pydantic import BaseModel
# from src.deepseek_llm import DeepSeekLLM
# from src.database import get_db
# from src.availability_logic import check_availability, add_booking, check_available_slotes
# import json
# import re
# import logging
# from src.entity_extraction import extract_entities
# from recommendtion.recommendations.core.recommendation_engine import RecommendationEngine

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# router = APIRouter()

# class QuestionRequest(BaseModel):
#     session_id: str
#     question: str

# def get_missing_params(params: dict, required_fields: list[str]) -> list[str]:
#     """Get list of missing required parameters"""
#     return [f for f in required_fields if f not in params or not params[f]]

# # Mapping each action to its required parameters
# REQUIRED_FIELDS = {
#     "check_availability": ["room_name", "date", "start_time", "end_time"],
#     "add_booking": ["room_name", "date", "start_time", "end_time"],
#     "alternatives": ["date", "start_time", "end_time"],
#     "cancel_booking": ["booking_id"],
    
# }

# # Fallback questions per missing parameter
# FALLBACK_QUESTIONS = {
#     "room_name": "Which room would you like to book?",
#     "date": "What date would you like to book it for? Please use YYYY-MM-DD format.",
#     "start_time": "What start time do you want? Please use HH:MM format.",
#     "end_time": "What end time do you want? Please use HH:MM format.",
#     "booking_id": "Please provide the booking ID to cancel.",
# }

# # Simple in-memory session store: session_id -> {"action": ..., "params": {...}, "last_asked": ...}
# session_store = {}

# @router.post("/ask_llm/")
# async def ask_llm(request: QuestionRequest, db=Depends(get_db)):
#     session_id = request.session_id
#     question = request.question
#     # Load session or initialize
#     session = session_store.get(session_id, {"action": None, "params": {}, "last_asked": None})

#     # If last asked a question (waiting for missing param), treat this input as answer
#     if session["last_asked"]:
#         param_name = session["last_asked"]
#         session["params"][param_name] = question
#         session["last_asked"] = None
#         session_store[session_id] = session
#     else:
#     # No pending missing param — call LLM to extract action + parameters
#         llm = DeepSeekLLM()

#     prompt = f"""
# You are an intelligent assistant that helps manage room bookings.

# From the following user request:
# \"{question}\"

# Extract the **action** and its corresponding **parameters** in **strict JSON format**.

# Supported actions:
# - \"check_availability\"
# - \"add_booking\"
# - \"cancel_booking\"
# - \"alternatives\"

# If the request is not related to any of these actions, return:
# {{ "action": "unsupported", "parameters": {{}} }}

# Required JSON structure:
# {{
#   "action": "check_availability" | "add_booking" | "cancel_booking" | "alternatives",
#   "parameters": {{
#     "room_name": "...",
#     "date": "yyyy-mm-dd",
#     "start_time": "HH:MM",
#     "end_time": "HH:MM",
#     "booking_id": "..."  # Only needed for cancel_booking
#   }}
# }}

# Respond in **only JSON format**, without explanations.
# """

#     try:
#         llm_response = llm._call(prompt)
#         print("Raw LLM response:", llm_response)

#         cleaned_response = re.sub(r"^```json|```$", "", llm_response.strip(), flags=re.MULTILINE).strip()
#         print("Cleaned LLM response:", cleaned_response)

#         parsed = json.loads(cleaned_response)


        
#         if "action" not in parsed or "parameters" not in parsed:
#             return {
#                 "status": "llm_response_invalid",
#                 "message": "LLM did not return a valid action or parameters.",
#                 "llm_response": cleaned_response
#             }

#     except json.JSONDecodeError as e:
#         raise HTTPException(status_code=500, detail=f"Error parsing LLM output: {str(e)}")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"LLM call failed: {str(e)}")

#     action = parsed.get("action")
#     params = parsed.get("parameters", {})
    
#     if action == "unsupported":
#         return {
#             "status": "unsupported_action",
#             "message": "I'm here to help with room bookings only. Try asking something like 'Book LT1 tomorrow at 10am'."
#         }
#     extracted = extract_entities(question)
#     for key, value in extracted.items():
#         if key not in params or not params[key]:
#             params[key] = value
            
#         session["action"] = action
#         session["params"] = params
#         session["last_asked"] = None
#         session_store[session_id] = session
            
#      # Check for missing params now
#     required_fields = REQUIRED_FIELDS.get(session["action"], [])
#     missing_fields = get_missing_params(session["params"], required_fields)

#     if missing_fields:
#         # Ask fallback for first missing param
#         next_missing = missing_fields[0]
#         session["last_asked"] = next_missing
#         session_store[session_id] = session
#         return {
#             "status": "missing_parameters",
#             "missing_parameter": next_missing,
#             "message": FALLBACK_QUESTIONS.get(next_missing, f"Please provide {next_missing}."),
#         }

#     # All params present — perform action
#     params = session["params"]
#     action = session["action"]

#     if action == "check_availability":
#         return check_availability(
#             room_name=params["room_name"],
#             date=params["date"],
#             start_time=params["start_time"],
#             end_time=params["end_time"],
#             db=db,
#         )
      
        
#     elif action == "add_booking":
#     # First check availability
#         availability = check_availability(
#         room_name=params["room_name"],
#         date=params["date"],
#         start_time=params["start_time"],
#         end_time=params["end_time"],
#         db=db,
#     )
#         print("Availability response:", availability)

#         if availability["status"] != "available":
#             return {
#             "status": "unavailable",
#             "message": f"{params['room_name']} is NOT available on {params['date']} from {params['start_time']} to {params['end_time']}."
#         }

#     # Then add booking if available
#         return add_booking(
#             room_name=params["room_name"],
#             date=params["date"],
#             start_time=params["start_time"],
#             end_time=params["end_time"],
#             created_by=params.get("created_by", "system"),
#             db=db,
#         )

#     elif action == "alternatives":
#         return check_available_slotes(
#             date=params["date"],
#             start_time=params["start_time"],
#             end_time=params["end_time"],
#             db=db,
#         )
#     elif action == "cancel_booking":
#         # Add actual cancel logic if implemented
#         return {"status": "success", "message": f"Booking {params['booking_id']} cancelled."}

#     return {"status": "error", "message": "Unhandled action."}


# src/api.py
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime
from src.deepseek_llm import DeepSeekLLM
from src.database import get_db
from src.availability_logic import check_availability, add_booking, check_available_slotes
from src.entity_extraction import extract_entities
from recommendtion.recommendations.core.recommendation_engine import RecommendationEngine
from recommendtion.config.recommendation_config import RecommendationConfig
from .availability_logic import handle_unavailable_room, get_smart_recommendations

import json
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

class QuestionRequest(BaseModel):
    session_id: str
    question: str
    
class RecommendationRequest(BaseModel):
    user_id: str
    room_id: Optional[str] = None
    start_time: datetime
    end_time: datetime
    purpose: str
    capacity: int
    requirements: Optional[Dict[str, Any]] = None


def get_missing_params(params: dict, required_fields: list[str]) -> list[str]:
    """Get list of missing required parameters"""
    return [f for f in required_fields if f not in params or not params[f]]

# Mapping each action to its required parameters
REQUIRED_FIELDS = {
    "check_availability": ["room_name", "date", "start_time", "end_time"],
    "add_booking": ["room_name", "date", "start_time", "end_time"],
    "alternatives": ["date", "start_time", "end_time"],
    "cancel_booking": ["booking_id"],
    "get_recommendations": ["date"],
}

# Fallback questions per missing parameter
FALLBACK_QUESTIONS = {
    "room_name": "Which room would you like to book?",
    "date": "What date would you like to book it for? Please use YYYY-MM-DD format.",
    "start_time": "What start time do you want? Please use HH:MM format.",
    "end_time": "What end time do you want? Please use HH:MM format.",
    "booking_id": "Please provide the booking ID to cancel.",
}

# Simple in-memory session store: session_id -> {"action": ..., "params": {...}, "last_asked": ...}
session_store = {}

@router.post("/ask_llm/")
async def ask_llm(request: QuestionRequest, db=Depends(get_db)):
    """
    Main endpoint for processing natural language booking requests with AI recommendations
    """
    session_id = request.session_id
    question = request.question
    
    try:
        # Load session or initialize
        session = session_store.get(session_id, {
            "action": None, 
            "params": {}, 
            "last_asked": None,
            "user_preferences": {}
        })

        # If last asked a question (waiting for missing param), treat this input as answer
        if session["last_asked"]:
            param_name = session["last_asked"]
            session["params"][param_name] = question
            session["last_asked"] = None
            session_store[session_id] = session
        else:
            # No pending missing param — call LLM to extract action + parameters
            llm = DeepSeekLLM()

            prompt = f"""
You are an intelligent assistant that helps manage room bookings with AI-powered recommendations.

From the following user request:
\"{question}\"

Extract the **action** and its corresponding **parameters** in **strict JSON format**.

Supported actions:
- \"check_availability\" - Check if a room is available
- \"add_booking\" - Book a room
- \"cancel_booking\" - Cancel an existing booking  
- \"alternatives\" - Find alternative time slots or rooms
- \"get_recommendations\" - Get AI-powered booking suggestions

If the request is not related to any of these actions, return:
{{ "action": "unsupported", "parameters": {{}} }}

Required JSON structure:
{{
  "action": "check_availability" | "add_booking" | "cancel_booking" | "alternatives" | "get_recommendations",
  "parameters": {{
    "room_name": "...",
    "date": "yyyy-mm-dd",
    "start_time": "HH:MM",
    "end_time": "HH:MM",
    "booking_id": "...",  # Only needed for cancel_booking
    "preferences": {{}}   # Optional user preferences
  }}
}}

Respond in **only JSON format**, without explanations.
"""

            try:
                llm_response = llm._call(prompt)
                logger.info(f"Raw LLM response: {llm_response}")

                cleaned_response = re.sub(r"^```json|```$", "", llm_response.strip(), flags=re.MULTILINE).strip()
                logger.info(f"Cleaned LLM response: {cleaned_response}")

                parsed = json.loads(cleaned_response)

                if "action" not in parsed or "parameters" not in parsed:
                    return {
                        "status": "llm_response_invalid",
                        "message": "LLM did not return a valid action or parameters.",
                        "llm_response": cleaned_response
                    }

            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error parsing LLM output: {str(e)}")
            except Exception as e:
                logger.error(f"LLM call failed: {str(e)}")
                raise HTTPException(status_code=500, detail=f"LLM call failed: {str(e)}")

            action = parsed.get("action")
            params = parsed.get("parameters", {})
            
            if action == "unsupported":
                # Provide helpful suggestions with recommendation context
                try:
                    rec_engine = RecommendationEngine(db)
                    popular_suggestions = await rec_engine.get_popular_suggestions()
                    
                    suggestion_text = ""
                    if popular_suggestions:
                        suggestion_text = f" Popular options include: {', '.join(popular_suggestions[:3])}"
                    
                    return {
                        "status": "unsupported_action",
                        "message": f"I'm here to help with room bookings only. Try asking something like 'Book LT1 tomorrow at 10am' or 'Show me booking recommendations'.{suggestion_text}"
                    }
                except Exception as e:
                    logger.error(f"Error getting popular suggestions: {str(e)}")
                    return {
                        "status": "unsupported_action",
                        "message": "I'm here to help with room bookings only. Try asking something like 'Book LT1 tomorrow at 10am' or 'Show me recommendations'."
                    }
            
            # Extract additional entities using existing entity extraction
            extracted = extract_entities(question)
            for key, value in extracted.items():
                if key not in params or not params[key]:
                    params[key] = value
                    
            session["action"] = action
            session["params"] = params
            session["last_asked"] = None
            session_store[session_id] = session
                
        # Check for missing params now
        required_fields = REQUIRED_FIELDS.get(session["action"], [])
        missing_fields = get_missing_params(session["params"], required_fields)

        if missing_fields:
            # Ask fallback for first missing param
            next_missing = missing_fields[0]
            session["last_asked"] = next_missing
            session_store[session_id] = session
            return {
                "status": "missing_parameters",
                "missing_parameter": next_missing,
                "message": FALLBACK_QUESTIONS.get(next_missing, f"Please provide {next_missing}."),
            }

        # All params present — perform action
        params = session["params"]
        action = session["action"]

        # Execute the requested action
        if action == "check_availability":
            return await handle_check_availability(params, db, session_id)
            
        elif action == "add_booking":
            return await handle_add_booking(params, db, session_id)

        elif action == "alternatives":
            return await handle_alternatives(params, db, session_id)
            
        elif action == "cancel_booking":
            return await handle_cancel_booking(params, db, session_id)
            
        elif action == "get_recommendations":
            return await handle_get_recommendations(params, db, session_id)

        return {"status": "error", "message": "Unhandled action."}
        
    except Exception as e:
        logger.error(f"Unexpected error in ask_llm: {str(e)}")
        return {
            "status": "error",
            "message": "An unexpected error occurred. Please try again.",
            "error_details": str(e) if logger.level == logging.DEBUG else None
        }



@router.post("/recommend")
async def get_recommendation(request: RecommendationRequest, db=Depends(get_db)):
    """
    Direct recommendation endpoint for structured recommendation requests
    """
    try:
        config = RecommendationConfig()
        engine = RecommendationEngine(db=db, config=config)
        
        request_data = request.dict()
        
        recommendations = engine.get_recommendations(request_data)
        
        return {
            "status": "success",
            "count": len(recommendations) if recommendations else 0,
            "recommendations": recommendations or [],
            "request_details": {
                "user_id": request.user_id,
                "purpose": request.purpose,
                "capacity": request.capacity,
                "time_range": f"{request.start_time} - {request.end_time}"
            }
        }
        
    except Exception as e:
        logger.error(f"Error in get_recommendation: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to generate recommendations: {str(e)}"
        )
        
        
async def handle_check_availability(params: dict, db, user_id: str) -> dict:
    """Handle availability checking with recommendations"""
    try:
        availability_result = check_availability(
            room_name=params["room_name"],
            date=params["date"],
            start_time=params["start_time"],
            end_time=params["end_time"],
            db=db,
            user_id=user_id or "anonymous_user"
        )
        
        logger.info(f"Availability response: {availability_result}")

        
        # If unavailable, add intelligent recommendations
        if availability_result["status"] == "unavailable":
            unavailable_response = await handle_unavailable_room(
                params=params,
                db=db,
                user_id=user_id,
                base_message=availability_result.get("message", "Room is not available."),
                request_type="check_availability"
            )
                
            return {
                **availability_result,
                **unavailable_response
            }
            
        elif availability_result["status"] == "available":
            # Room is available - provide proactive suggestions
            try:
                config = RecommendationConfig()
                rec_engine = RecommendationEngine(db=db, config=config)
                
                proactive_suggestions = await rec_engine.get_proactive_suggestions(
                    user_id=user_id,
                    context=params
                )
                
                if proactive_suggestions:
                    availability_result["future_suggestions"] = proactive_suggestions[:3]
                    availability_result["message"] = f"{availability_result.get('message', '')} ✅ Room is available! You might also be interested in these future bookings."
                    availability_result["proactive_help"] = "Based on your booking patterns, here are some suggestions for future bookings."
                    
            except Exception as e:
                logger.error(f"Proactive suggestions failed: {e}")
        
        return availability_result
        
    except Exception as e:
        logger.error(f"Error in handle_check_availability: {str(e)}")
        return {
            "status": "check_failed",
            "message": "We encountered an issue while checking availability. Let us find alternatives for you!",
            **(await handle_unavailable_room(
                params=params,
                db=db,
                user_id=user_id,
                base_message="Availability check failed, but here are some options:",
                request_type="check_availability_error"
            ))
        }


async def handle_add_booking(params: dict, db, user_id: str) -> dict:
    """Handle booking creation with intelligent fallbacks"""
    try:
        # First check availability
        availability = check_availability(
            room_name=params["room_name"],
            date=params["date"],
            start_time=params["start_time"],
            end_time=params["end_time"],
            db=db,
            user_id=user_id
        )
        logger.info(f"Availability response: {availability}")

        if availability["status"] != "available":
            return await handle_unavailable_room(
                params=params,
                db=db,
                user_id=user_id,
                base_message=f"Sorry, {params['room_name']} is not available on {params['date']} from {params['start_time']} to {params['end_time']}.",
                request_type="add_booking"
            )

        # Room is available - proceed with booking
        try:
            booking_result = await add_booking(
                room_name=params["room_name"],
                date=params["date"],
                start_time=params["start_time"],
                end_time=params["end_time"],
                created_by=params.get("created_by", user_id),
                db=db,
                user_id=user_id
            )
            
            # Learn from successful booking and provide future suggestions
            if booking_result.get("status") == "success":
                try:
                    config = RecommendationConfig()
                    rec_engine = RecommendationEngine(db=db, config=config)
                    
                    # Learn from success
                    await rec_engine.learn_from_booking(
                        user_id=user_id,
                        booking_data=params,
                        outcome="success"
                    )
                    
                    # Get future suggestions
                    future_suggestions = await rec_engine.get_proactive_suggestions(
                        user_id=user_id,
                        context={**params, "recent_booking": params}
                    )
                    
                    if future_suggestions:
                        booking_result["future_suggestions"] = future_suggestions[:3]
                        booking_result["message"] = f"🎉 {booking_result.get('message', '')} Booking confirmed! Here are some suggestions for your next booking:"
                        booking_result["success_enhancement"] = "We're learning your preferences to make future bookings even easier!"
                        
                except Exception as e:
                    logger.error(f"Learning from booking failed: {e}")
            
            return booking_result
            
        except Exception as booking_error:
            logger.error(f"Booking creation failed: {booking_error}")
            
            # If booking fails due to technical issues, still provide recommendations
            return {
                "status": "booking_failed",
                "message": "We encountered a technical issue while creating your booking, but here are some alternatives:",
                **(await get_smart_recommendations(
                    params=params,
                    db=db,
                    user_id=user_id,
                    request_type="add_booking",
                    failure_reason="technical_error"
                )),
                "error_type": "technical_error",
                "suggested_action": "Please try booking one of the recommended alternatives, or try your original request again in a few moments.",
                "original_request": params
            }
        
    except Exception as e:
        logger.error(f"Error in handle_add_booking: {str(e)}")
        
        # Even in case of unexpected errors, try to provide some help
        return {
            "status": "booking_failed", 
            "message": "We're having trouble processing your booking right now, but we found some great alternatives for you!",
            **(await get_smart_recommendations(
                params=params,
                db=db,
                user_id=user_id,
                request_type="add_booking",
                failure_reason="unexpected_error"
            )),
            "suggested_action": "Choose one of these available options, or try your original request again in a few moments.",
            "help_text": "If you continue having issues, our support team is here to help.",
            "error_details": str(e) if logger.level == logging.DEBUG else None
        }
        

async def handle_alternatives(params: dict, db, user_id: str) -> dict:
    """Handle alternative suggestions with AI recommendations"""
    try:
        # Use recommendation engine for intelligent alternatives
        rec_engine = RecommendationEngine(db)
        recommendations = await rec_engine.get_recommendations(
            user_id=user_id,
            request_type="comprehensive",
            context={
                **params,
                "original_request": "alternatives",
                "preference_type": "flexible"
            }
        )
        
        if recommendations:
            return {
                "status": "success",
                "message": "Here are intelligent alternatives based on your preferences and availability:",
                "recommendations": recommendations,
                "total_alternatives": len(recommendations)
            }
        else:
            # Fallback to existing alternatives logic
            logger.info("No AI recommendations available, falling back to basic alternatives")
            basic_alternatives = check_available_slotes(
                date=params["date"],
                start_time=params["start_time"],
                end_time=params["end_time"],
                db=db,
            )
            return {
                **basic_alternatives,
                "message": "Here are available alternatives:",
                "recommendation_type": "basic"
            }
            
    except Exception as e:
        logger.error(f"Error in handle_alternatives: {e}")
        # Fallback to existing alternatives logic
        try:
            return check_available_slotes(
                date=params["date"],
                start_time=params["start_time"],
                end_time=params["end_time"],
                db=db,
            )
        except Exception as fallback_error:
            logger.error(f"Fallback alternatives also failed: {fallback_error}")
            return {
                "status": "error",
                "message": "Failed to find alternatives. Please try again."
            }

async def handle_cancel_booking(params: dict, db, user_id: str) -> dict:
    """Handle booking cancellation with learning"""
    try:
       
        booking_id = params.get("booking_id")
        
        # Learn from cancellation pattern
        try:
            rec_engine = RecommendationEngine(db)
            await rec_engine.learn_from_booking(
                user_id=user_id,
                booking_data={"booking_id": booking_id},
                outcome="cancelled"
            )
        except Exception as e:
            logger.error(f"Learning from cancellation failed: {e}")
        
        return {
            "status": "success", 
            "message": f"Booking {booking_id} has been cancelled.",
            "booking_id": booking_id
        }
        
    except Exception as e:
        logger.error(f"Error in handle_cancel_booking: {str(e)}")
        return {
            "status": "error",
            "message": "Failed to cancel booking. Please try again."
        }

async def handle_get_recommendations(params: dict, db, user_id: str) -> dict:
    """Handle proactive recommendation requests"""
    try:
        rec_engine = RecommendationEngine(db)
        
        # Get comprehensive recommendations
        recommendations = await rec_engine.get_proactive_suggestions(
            user_id=user_id,
            context=params
        )
        
        if not recommendations:
            # If no personalized recommendations, get popular options
            popular_recommendations = await rec_engine.get_popular_recommendations(
                context=params
            )
            
            return {
                "status": "success",
                "message": "Here are popular booking options since we're still learning your preferences:",
                "recommendations": popular_recommendations,
                "recommendation_type": "popular",
                "note": "Book more rooms to get personalized recommendations!"
            }
        
        # Get user's booking summary for additional context
        booking_summary = await rec_engine.get_user_booking_summary(user_id)
        
        return {
            "status": "success",
            "message": "Here are personalized recommendations based on your booking patterns:",
            "recommendations": recommendations,
            "recommendation_type": "personalized",
            "user_summary": booking_summary,
            "total_suggestions": len(recommendations)
        }
        
    except Exception as e:
        logger.error(f"Error in handle_get_recommendations: {str(e)}")
        return {
            "status": "error",
            "message": "Failed to generate recommendations. Please try again."
        }

# Additional utility endpoints for the recommendation system

# @router.get("/recommendations/{user_id}")
# async def get_user_recommendations(user_id: str, db=Depends(get_db)):
#     """Get personalized recommendations for a specific user"""
#     try:
#         rec_engine = RecommendationEngine(db)
#         recommendations = await rec_engine.get_proactive_suggestions(
#             user_id=user_id,
#             context={}
#         )
        
#         return {
#             "status": "success",
#             "user_id": user_id,
#             "recommendations": recommendations,
#             "generated_at": rec_engine.get_current_timestamp()
#         }
#     except Exception as e:
#         logger.error(f"Error getting user recommendations: {str(e)}")
#         raise HTTPException(status_code=500, detail="Failed to get recommendations")

# @router.get("/recommendations/{user_id}/summary")
# async def get_user_recommendation_summary(user_id: str, db=Depends(get_db)):
#     """Get user's booking patterns and recommendation performance summary"""
#     try:
#         rec_engine = RecommendationEngine(db)
#         summary = await rec_engine.get_user_booking_summary(user_id)
        
#         return {
#             "status": "success",
#             "user_id": user_id,
#             "summary": summary
#         }
#     except Exception as e:
#         logger.error(f"Error getting user summary: {str(e)}")
#         raise HTTPException(status_code=500, detail="Failed to get user summary")

# @router.post("/recommendations/feedback")
# async def provide_recommendation_feedback(
#     user_id: str,
#     recommendation_id: str,
#     feedback: str,  # 'accepted', 'rejected', 'modified'
#     db=Depends(get_db)
# ):
#     """Provide feedback on recommendation quality"""
#     try:
#         rec_engine = RecommendationEngine(db)
#         await rec_engine.record_recommendation_feedback(
#             user_id=user_id,
#             recommendation_id=recommendation_id,
#             feedback=feedback
#         )
        
#         return {
#             "status": "success",
#             "message": "Feedback recorded successfully",
#             "user_id": user_id,
#             "recommendation_id": recommendation_id,
#             "feedback": feedback
#         }
#     except Exception as e:
#         logger.error(f"Error recording feedback: {str(e)}")
#         raise HTTPException(status_code=500, detail="Failed to record feedback")

# @router.get("/popular-rooms")
# async def get_popular_rooms(db=Depends(get_db)):
#     """Get currently popular rooms and time slots"""
#     try:
#         rec_engine = RecommendationEngine(db)
#         popular_data = await rec_engine.get_popular_recommendations()
        
#         return {
#             "status": "success",
#             "popular_rooms": popular_data.get("rooms", []),
#             "popular_times": popular_data.get("times", []),
#             "trending": popular_data.get("trending", [])
#         }
#     except Exception as e:
#         logger.error(f"Error getting popular rooms: {str(e)}")
#         raise HTTPException(status_code=500, detail="Failed to get popular rooms")