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

class Zone(Base):
    __tablename__ = "zones"
    
    id = Column(String, primary_key=True)
    room_id = Column(String, ForeignKey("rooms.id"))
    name = Column(String, nullable=False)
    shapes = Column(String)  # JSON string for shapes array
    color = Column(String)
    properties = Column(String)  # JSON string for properties object
    
    room = relationship("Room", back_populates="zones")

class PointOfInterest(Base):
    __tablename__ = "points_of_interest"
    
    id = Column(String, primary_key=True)
    room_id = Column(String, ForeignKey("rooms.id"))
    name = Column(String, nullable=False)
    coordinates = Column(String)  # JSON string for coordinates array
    
    room = relationship("Room", back_populates="points_of_interest")
    @property
    def coordinates_list(self):
        """Deserialize the coordinates JSON string into a Python list."""
        try:
            return json.loads(self.coordinates)
        except (TypeError, ValueError):
            return None

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
    zones = relationship("Zone", back_populates="room", cascade="all, delete-orphan")
    points_of_interest = relationship("PointOfInterest", back_populates="room", cascade="all, delete-orphan")