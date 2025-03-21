from sqlalchemy import Column, String, Integer, ForeignKey, JSON
from sqlalchemy.orm import relationship
from database import Base

class Zone(Base):
    __tablename__ = "zones"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    color = Column(String, nullable=False)  # Hex color code
    type_id = Column(String, ForeignKey("zone_types.id"), nullable=False)  # Reference to ZoneType
    shape = Column(JSON, nullable=False)    # Array of shapes (e.g., rectangles, circles)
    floor_id = Column(String, ForeignKey("floors.id"), nullable=False)

    floor = relationship("Floor", back_populates="zones")
    zone_type = relationship("ZoneType", back_populates="zones")
    pois = relationship("POI", secondary="poi_zone_association", back_populates="zones")