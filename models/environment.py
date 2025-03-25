from sqlalchemy import Column, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from database import Base

class Environment(Base):
    __tablename__ = "environments"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    address = Column(String)
    floors = relationship("Floor", back_populates="environment")