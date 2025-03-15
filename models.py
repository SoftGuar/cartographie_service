from sqlalchemy import Column, Integer, String, JSON, LargeBinary, ForeignKey, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class Company(Base):
    __tablename__ = "companies"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    address = Column(String)
    rooms = relationship("Room", back_populates="company")

class Room(Base):
    __tablename__ = "rooms"
    
    id = Column(String, primary_key=True)
    company_id = Column(String, ForeignKey("companies.id"))
    name = Column(String, nullable=False)
    building = Column(String)
    floor = Column(Integer)
    width = Column(Float)
    height = Column(Float)
    coordinates = Column(String)  # JSON string for SQLite compatibility
    grid_data = Column(String)    # JSON string for SQLite compatibility
    grid_dimensions = Column(String)  # JSON string for SQLite compatibility
    image_data = Column(LargeBinary, nullable=True)
    
    company = relationship("Company", back_populates="rooms")