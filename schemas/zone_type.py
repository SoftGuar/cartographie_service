from pydantic import BaseModel
from typing import Optional, Dict

class ZoneTypeBase(BaseModel):
    """Base schema for ZoneType."""
    name: str
    properties: Optional[Dict] = None  

class ZoneTypeCreate(ZoneTypeBase):
    """Schema for creating a ZoneType."""
    pass

class ZoneTypeUpdate(BaseModel):
    """Schema for updating a ZoneType."""
    name: Optional[str] = None
    properties: Optional[Dict] = None

class ZoneTypeResponse(ZoneTypeBase):
    """Schema for returning a ZoneType."""
    id: str

    class Config:
        orm_mode = True  