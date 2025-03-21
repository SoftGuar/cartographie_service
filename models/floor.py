from sqlalchemy import Column, Integer, String, LargeBinary, ForeignKey, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from database import Base

class Floor(Base):
    __tablename__ = "floors"
    
    id = Column(String, primary_key=True)
    environment_id = Column(String, ForeignKey("environments.id"))
    name = Column(String, nullable=False)
    building = Column(String)
    level = Column(Integer)  # Changed from 'floor' to 'level' to avoid confusion
    width = Column(Float)
    height = Column(Float)
    coordinates = Column(String)  # JSON string for SQLite compatibility
    grid_data = Column(String)    # JSON string for SQLite compatibility
    grid_dimensions = Column(String)  # JSON string for SQLite compatibility
    image_data = Column(LargeBinary, nullable=True)
    
    environment = relationship("Environment", back_populates="floors")