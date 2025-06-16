from sqlalchemy import Column, String, JSON
from sqlalchemy.orm import relationship
from database import Base

class ZoneType(Base):
    __tablename__ = "zone_types"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)  # e.g., "Zone de circulation", "Zone de travail"
    properties = Column(JSON, nullable=True, default=dict)  # Store properties as a JSON object

    zones = relationship("Zone", back_populates="zone_type")