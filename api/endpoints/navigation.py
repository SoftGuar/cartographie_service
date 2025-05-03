from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from typing import List, Dict

from database import get_db
from services.navigation_service import NavigationService
from schemas.navigation import NavigationRequest, NavigationResponse, ActionStep

router = APIRouter(
    prefix="/api/navigation",
    tags=["navigation"],
    responses={404: {"description": "Not found"}},
)

@router.post("", response_model=NavigationResponse)
def navigate(
    request: NavigationRequest,
    db: Session = Depends(get_db)
):
    """
    Calculate navigation path from current position to destination POI.
    
    Args:
        request: Navigation request with current position and destination
        db: Database session
        
    Returns:
        Navigation response with path, actions, and total distance
    """
    try:
        result = NavigationService.get_navigation_path(
            db=db, 
            current_x=request.current_x, 
            current_y=request.current_y, 
            floor_id=request.floor_id, 
            destination_poi_id=request.destination_poi_id
        )
        
        # Convert dict actions to ActionStep objects for response model
        actions = [ActionStep(**action) for action in result["actions"]]
        
        return NavigationResponse(
            path=result["path"],
            actions=actions,
            total_distance=result["total_distance"]
        )
        
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))