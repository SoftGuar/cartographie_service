from pydantic import BaseModel
from typing import Optional

class POIBase(BaseModel):
    """Base schema for POI."""
    name: str
    description: Optional[str] = None
    category_id: str  # Reference to Category
    point_id: str  # Reference to Point

class POICreate(POIBase):
    """Schema for creating a POI."""
    pass

class POIUpdate(BaseModel):
    """Schema for updating a POI."""
    name: Optional[str] = None
    description: Optional[str] = None
    category_id: Optional[str] = None
    point_id: Optional[str] = None

class POIResponse(POIBase):
    """Schema for returning a POI."""
    id: str

    class Config:
        orm_mode = True  