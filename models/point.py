from sqlalchemy import Column, String, Float
from sqlalchemy.orm import relationship
from database import Base

class Point(Base):
    __tablename__ = "points"

    id = Column(String, primary_key=True)
    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)

    # Relationship to POI
    poi = relationship("POI", back_populates="point") 