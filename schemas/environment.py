from pydantic import BaseModel
from typing import List
from .floor import Floor

class EnvironmentBase(BaseModel):
    name: str
    address: str

class EnvironmentCreate(EnvironmentBase):
    id: str

class Environment(EnvironmentBase):
    id: str
    floors: List[Floor] = []
    
    class Config:
        orm_mode = True