from pydantic import BaseModel
from typing import Optional, List, Dict

class ZoneBase(BaseModel):
    """Base schema for Zone."""
    name: str
    color: str  # Hex color code
    type_id: str  # Reference to ZoneType
    shape: Dict  # JSON object representing the shape (e.g., polygon, circle)
    floor_id: str  # Reference to Floor

class ZoneCreate(ZoneBase):
    """Schema for creating a Zone."""
    pass

class ZoneUpdate(BaseModel):
    """Schema for updating a Zone."""
    name: Optional[str] = None
    color: Optional[str] = None
    type_id: Optional[str] = None
    shape: Optional[Dict] = None

class ZoneResponse(ZoneBase):
    """Schema for returning a Zone."""
    id: str

    class Config:
        orm_mode = True 