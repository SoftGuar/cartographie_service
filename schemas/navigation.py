from pydantic import BaseModel
from typing import List, Dict

class NavigationRequest(BaseModel):
    """Schema for navigation request input."""
    current_x: float  # Real-world x coordinate in meters
    current_y: float  # Real-world y coordinate in meters
    floor_id: str
    destination_poi_id: str

class ActionStep(BaseModel):
    """Schema for a single navigation action step."""
    type: str  # "move" or "rotate"
    value: float  # Distance in meters for move, angle in degrees for rotate
    message: str
    end_position: Dict[str, float]  # {x: float, y: float} in real-world coordinates

class NavigationResponse(BaseModel):
    """Schema for navigation response."""
    path: List[Dict[str, int]]  # List of grid coordinates [{x: int, y: int}]
    actions: List[ActionStep]
    total_distance: float  # Total distance in meters