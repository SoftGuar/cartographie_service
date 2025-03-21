from pydantic import BaseModel
from typing import List, Optional, Union

class FloorBase(BaseModel):
    name: str
    building: str
    level: int
    width: float
    height: float
    coordinates: List[float]

class FloorCreate(FloorBase):
    id: Optional[str] = None
    environment_id: Optional[str] = None
    grid_data: List[List[int]]
    grid_dimensions: List[int]
    image_data: Optional[str] = None

class FloorUpdate(BaseModel):
    grid_data: List[List[int]]
    grid_dimensions: List[int]
    image_data: Optional[str] = None

class Floor(FloorBase):
    id: str
    environment_id: Optional[str] = None
    grid_data: Optional[List[List[int]]] = None
    grid_dimensions: Optional[List[int]] = None
    image_data: Optional[str] = None
    
    class Config:
        orm_mode = True