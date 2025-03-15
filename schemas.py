from pydantic import BaseModel
from typing import List, Optional, Union

class RoomBase(BaseModel):
    name: str
    building: str
    floor: int
    width: float
    height: float
    coordinates: List[float]

class RoomCreate(RoomBase):
    id: Optional[str] = None  # Make id optional
    company_id: Optional[str] = None
    grid_data: List[List[int]]
    grid_dimensions: List[int]
    image_data: Optional[str] = None

class RoomUpdate(BaseModel):
    grid_data: List[List[int]]
    grid_dimensions: List[int]
    image_data: Optional[str] = None

class Room(RoomBase):
    id: str
    company_id: Optional[str] = None
    grid_data: Optional[List[List[int]]] = None
    grid_dimensions: Optional[List[int]] = None
    image_data: Optional[str] = None  # Add image_data field
    
    class Config:
        orm_mode = True

class CompanyBase(BaseModel):
    name: str
    address: str

class CompanyCreate(CompanyBase):
    id: str

class Company(CompanyBase):
    id: str
    rooms: List[Room] = []
    
    class Config:
        orm_mode = True