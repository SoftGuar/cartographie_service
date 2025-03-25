from pydantic import BaseModel, Field, validator
from typing import List, Optional, Union
import json
import uuid

from schemas.zone import ZoneResponse

class FloorBase(BaseModel):
    name: str
    environment_id: str
    level: int
    width: float
    height: float
    coordinates: str  # JSON string (as stored in DB)
    building: Optional[str] = None

class FloorCreate(FloorBase):
    grid_data: List[List[int]] = Field(..., example=[[0, 1], [1, 0]])
    grid_dimensions: List[int] = Field(..., example=[10, 10])
    
    image_data: Optional[str] = None  # base64 or None
    
    @validator("coordinates")
    def validate_coordinates(cls, v):
        try:
            json.loads(v)
            return v
        except json.JSONDecodeError:
            raise ValueError("Coordinates must be a valid JSON string")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Test Floor",
                "environment_id": "env-123",
                "level": 1,
                "width": 100.0,
                "height": 80.0,
                "coordinates": '{"points": [[0,0], [100,0], [100,80], [0,80]]}',
                "grid_data": [[0, 1], [1, 0]],
                "grid_dimensions": [10, 10],
                "image_data": None,
            }
        }

class FloorUpdate(BaseModel):
    grid_data: List[List[int]]
    grid_dimensions: List[int]
    image_data: Optional[str] = None

class FloorResponse(FloorBase):
    id: str
    grid_data: Optional[List[List[int]]] = None
    grid_dimensions: Optional[List[int]] = None
    image_data: Optional[str] = None
    zones: List[ZoneResponse] = Field(default_factory=list)
    
    class Config:
        from_attributes = True  