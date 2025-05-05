from typing import Optional
from pydantic import BaseModel

class CategoryResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]

    class Config:
        orm_mode = True