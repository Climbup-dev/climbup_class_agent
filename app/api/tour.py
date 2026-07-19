from fastapi import APIRouter, HTTPException
import os
import json

router = APIRouter()

@router.get("/api/v1/classrooms/{classroom_id}/tour")
async def get_classroom_tour(classroom_id: str):
    """
    Fetches the pre-generated Guided Tour JSON for a classroom.
    """
    tour_path = f"uploads/{classroom_id}_tour.json"
    
    if not os.path.exists(tour_path):
        # Fallback if the tour isn't generated yet or failed
        return {
            "syllabus": ["Introduction", "Main Concepts", "Summary"],
            "lessons": [
                {
                    "topic": "Welcome",
                    "exact_quote": "The Guided Tour is currently generating or unavailable for this PDF.",
                    "explanation": "You can still use the Chat Box to ask any specific questions!"
                }
            ]
        }
        
    try:
        with open(tour_path, "r", encoding="utf-8") as f:
            tour_data = json.load(f)
        return tour_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read tour file: {str(e)}")
