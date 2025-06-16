from sqlalchemy.orm import Session
from models.zone import Zone
from schemas.zone import ZoneCreate, ZoneUpdate
import uuid
from models.zone_type import ZoneType
from utils.notifications import send_notification
import threading

def create_zone(db: Session, zone: ZoneCreate):
    """Create a new zone."""
    zone_data = zone.dict()
    zone_data["id"] = str(uuid.uuid4()) 

    db_zone = Zone(**zone_data)
    db.add(db_zone)
    db.commit()
    db.refresh(db_zone)
    threading.Thread(
        target=send_notification,
        args=(
            {"name": db_zone.name},
            "/notifications/notify/zone-created"
        ),
        daemon=True
    ).start()
    return db_zone

def get_zone(db: Session, zone_id: str):
    """Get a zone by ID."""
    return db.query(Zone).filter(Zone.id == zone_id).first()

def update_zone(db: Session, zone_id: str, zone: ZoneUpdate):
    """Update a zone."""
    db_zone = db.query(Zone).filter(Zone.id == zone_id).first()
    if not db_zone:
        return None

    # Update only the fields provided in the request body
    for key, value in zone.dict(exclude_unset=True).items():
        setattr(db_zone, key, value)

    db.commit()
    db.refresh(db_zone)
    threading.Thread(
        target=send_notification,
        args=(
            {"name": db_zone.name},
            "/notifications/notify/zone-updated"
        ),
        daemon=True
    ).start()
    return db_zone

def delete_zone(db: Session, zone_id: str):
    """Delete a zone."""
    db_zone = db.query(Zone).filter(Zone.id == zone_id).first()
    if db_zone:
        db.delete(db_zone)
        db.commit()
        return True
    threading.Thread(
        target=send_notification,
        args=(
            {"id": db_zone.id},
            "/notifications/notify/zone-deleted"
        ),
        daemon=True
    ).start()
    return False

def get_zones_by_floor(db: Session, floor_id: str):
    """Get all zones for a specific floor."""
    return db.query(Zone).filter(Zone.floor_id == floor_id).all()

def validate_zone_shapes(shapes: list[dict]) -> bool:
    """Validate the structure of zone shapes"""
    required_fields = {
        "polygon": ["coordinates"],
        "rectangle": ["coordinates"], 
        "circle": ["center", "radius"]
    }
    
    for shape in shapes:
        shape_type = shape.get("type")
        if shape_type not in required_fields:
            raise ValueError(f"Invalid shape type: {shape_type}")
        
        for field in required_fields[shape_type]:
            if field not in shape:
                raise ValueError(f"Missing required field '{field}' for {shape_type}")
    
    return True

def get_all_zone_types(db: Session):
    """Retrieve all ZoneTypes."""
    return db.query(ZoneType).all()