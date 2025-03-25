from pydantic import BaseModel, ConfigDict, Field, validator
from typing import List, Literal, Optional, Union
import json

# ------ Individual Shape Types ------
class PolygonShape(BaseModel):
    type: Literal["polygon"]
    coordinates: List[List[List[float]]]  # [[[x,y], [x,y], ...]]

class RectangleShape(BaseModel):
    type: Literal["rectangle"] 
    coordinates: List[List[float]]  # [[x1,y1], [x2,y2]]

class CircleShape(BaseModel):
    type: Literal["circle"]
    center: List[float]  # [x,y]
    radius: float

class ZoneBase(BaseModel):
    """Base schema for Zone."""
    name: str = Field(..., min_length=1, max_length=100)
    color: str = Field(..., pattern=r"^#[0-9a-fA-F]{6}$")  # Hex color validation
    type_id: str = Field(..., min_length=1)
    shape: List[Union[PolygonShape, RectangleShape, CircleShape]]
    floor_id: str = Field(..., min_length=1)

    model_config = ConfigDict(
        json_encoders={
            'shape': lambda v: json.loads(json.dumps(v))  # Ensure JSON serialization
        },
        json_schema_extra={
            "example": {
                "name": "Conference Room",
                "color": "#FF0000",
                "type_id": "zone-type-123",
                "shape": [
                    {
                        "type": "polygon",
                        "coordinates": [[[0,0], [10,0], [10,10], [0,10]]]
                    },
                    {
                        "type": "circle",
                        "center": [5,5],
                        "radius": 2
                    }
                ],
                "floor_id": "floor-123"
            }
        }
    )

class ZoneCreate(ZoneBase):
    """Schema for creating a Zone."""
    @validator('shape')
    def validate_shapes(cls, v):
        if len(v) < 1:
            raise ValueError("At least one shape is required")
        return v
    pass

class ZoneUpdate(BaseModel):
    """Schema for updating a Zone."""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    color: Optional[str] = Field(None, pattern=r"^#[0-9a-fA-F]{6}$")
    type_id: Optional[str] = Field(None, min_length=1)
    shape: Optional[List[Union[PolygonShape, RectangleShape, CircleShape]]] = None
    floor_id: Optional[str] = Field(None, min_length=1)

class ZoneResponse(ZoneBase):
    """Schema for returning a Zone."""
    id: str

    model_config = ConfigDict(from_attributes=True)