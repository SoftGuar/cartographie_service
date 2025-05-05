from pydantic import BaseModel
from typing import Optional

class POIBase(BaseModel):
    """Base schema for POI."""
    name: str
    description: Optional[str] = None
    category_id: str  # Reference to Category
    point_id: str  # Reference to Point

class POICreate(BaseModel):
    """Schema for creating a POI."""
    name: str
    description: Optional[str] = None
    category_id: str  # Reference to Category
    x: float  # X coordinate for the Point
    y: float  # Y coordinate for the Point
    zone_id: Optional[str] = None  # Optional zone association
    floor_id: str  # Floor ID for the POI

class POIUpdate(BaseModel):
    """Schema for updating a POI."""
    name: Optional[str] = None
    description: Optional[str] = None
    category_id: Optional[str] = None
    point_id: Optional[str] = None

class POIResponse(BaseModel):
    """Schema for returning a POI."""
    id: str
    name: str
    description: Optional[str] = None
    category_id: str
    x: float  # X coordinate of the Point
    y: float  # Y coordinate of the Point

    class Config:
        orm_mode = True