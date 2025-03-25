from sqlalchemy import Table, Column, String, ForeignKey
from database import Base

poi_zone_association = Table(
    "poi_zone_association",
    Base.metadata,
    Column("poi_id", String, ForeignKey("pois.id"), primary_key=True),
    Column("zone_id", String, ForeignKey("zones.id"), primary_key=True),
)