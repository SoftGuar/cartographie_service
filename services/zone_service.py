from sqlalchemy.orm import Session
from models.zone import Zone
from schemas.zone import ZoneCreate, ZoneUpdate

def create_zone(db: Session, zone: ZoneCreate):
    """Create a new zone."""
    db_zone = Zone(**zone.dict())
    db.add(db_zone)
    db.commit()
    db.refresh(db_zone)
    return db_zone

def get_zone(db: Session, zone_id: str):
    """Get a zone by ID."""
    return db.query(Zone).filter(Zone.id == zone_id).first()

def update_zone(db: Session, zone_id: str, zone: ZoneUpdate):
    """Update a zone."""
    db_zone = db.query(Zone).filter(Zone.id == zone_id).first()
    if db_zone:
        for key, value in zone.dict().items():
            setattr(db_zone, key, value)
        db.commit()
        db.refresh(db_zone)
    return db_zone

def delete_zone(db: Session, zone_id: str):
    """Delete a zone."""
    db_zone = db.query(Zone).filter(Zone.id == zone_id).first()
    if db_zone:
        db.delete(db_zone)
        db.commit()
        return True
    return False

def get_zones_by_floor(db: Session, floor_id: str):
    """Get all zones for a specific floor."""
    return db.query(Zone).filter(Zone.floor_id == floor_id).all()