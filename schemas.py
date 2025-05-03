from pydantic import BaseModel
from typing import List, Optional, Union


class ZoneShape(BaseModel):
    coordinates: List[List[float]]  # [[x1, y1], [x2, y2]]

class ZoneProperties(BaseModel):
    width: Optional[float] = None
    height: Optional[float] = None

class ZoneBase(BaseModel):
    name: str
    shapes: List[ZoneShape]
    color: str
    properties: ZoneProperties

class ZoneCreate(ZoneBase):
    id: Optional[str] = None  # Make id optional
    room_id: str

class ZoneUpdate(ZoneBase):
    pass

class Zone(ZoneBase):
    id: str
    room_id: str
    
    class Config:
        orm_mode = True

class PointOfInterestBase(BaseModel):
    name: str
    coordinates: List[List[int]]  # Array of [x, y] coordinates

class PointOfInterestCreate(PointOfInterestBase):
    id: Optional[str] = None  # Make id optional
    room_id: str

class PointOfInterestUpdate(PointOfInterestBase):
    pass

class PointOfInterest(PointOfInterestBase):
    id: str
    room_id: str
    
    class Config:
        orm_mode = True

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
    zones: Optional[List[Zone]] = None
    points_of_interest: Optional[List[PointOfInterest]] = None
    
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
