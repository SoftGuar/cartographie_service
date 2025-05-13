from sqlalchemy import Column, String, Text, ForeignKey
from sqlalchemy.orm import relationship
from database import Base
import uuid

class POI(Base):
    __tablename__ = "pois"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    description = Column(Text)
    category_id = Column(String, ForeignKey("categories.id"), nullable=False)
    point_id = Column(String, ForeignKey("points.id"), nullable=False)

    category = relationship("Category", back_populates="pois")
    point = relationship("Point", back_populates="poi")
    zones = relationship("Zone", secondary="poi_zone_association", back_populates="pois")