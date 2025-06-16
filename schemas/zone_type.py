from pydantic import BaseModel, ConfigDict
from typing import Optional, Dict, Any, Union

class ZoneTypeBase(BaseModel):
    """Base schema for ZoneType."""
    name: str
    properties: Optional[Union[Dict[str, Any], None]] = None

class ZoneTypeCreate(ZoneTypeBase):
    """Schema for creating a ZoneType."""
    pass

class ZoneTypeUpdate(BaseModel):
    """Schema for updating a ZoneType."""
    name: Optional[str] = None
    properties: Optional[Union[Dict[str, Any], None]] = None

class ZoneTypeResponse(ZoneTypeBase):
    """Schema for returning a ZoneType."""
    id: str

    model_config = ConfigDict(
        from_attributes=True,
        arbitrary_types_allowed=True
    )  