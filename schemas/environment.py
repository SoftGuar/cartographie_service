from pydantic import BaseModel
from typing import List
from .floor import FloorResponse

class EnvironmentBase(BaseModel):
    name: str
    address: str



class EnvironmentCreate(EnvironmentBase):
    pass

class Environment(EnvironmentBase):
    id: str
    floors: List[FloorResponse] = []
    
    class Config:
        orm_mode = True