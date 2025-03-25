from sqlalchemy import Column, String, Text
from sqlalchemy.orm import relationship
from database import Base

class Category(Base):
    __tablename__ = "categories"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text)

    pois = relationship("POI", back_populates="category")