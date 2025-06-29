from sqlalchemy.orm import Session
from models.poi import POI
from models.point import Point
from models.zone import Zone
from schemas.poi import POICreate, POIUpdate

def create_poi(db: Session, poi_data: dict):
    """Create a new POI."""
    db_poi = POI(**poi_data)
    db.add(db_poi)
    db.commit()
    db.refresh(db_poi)
    return db_poi

def get_poi(db: Session, poi_id: str):
    """Get a POI by ID."""
    return db.query(POI).filter(POI.id == poi_id).first()

def update_poi(db: Session, poi_id: str, poi: POIUpdate):
    """Update a POI."""
    db_poi = db.query(POI).filter(POI.id == poi_id).first()
    if not db_poi:
        return None

    # Update the associated Point if x and y are provided
    if poi.x is not None and poi.y is not None:
        db_point = db.query(Point).filter(Point.id == db_poi.point_id).first()
        if db_point:
            db_point.x = poi.x
            db_point.y = poi.y
            db.commit()
            db.refresh(db_point)

    # Update the POI fields
    for key, value in poi.dict(exclude={"x", "y"}).items():
        if value is not None:
            setattr(db_poi, key, value)

    db.commit()
    db.refresh(db_poi)
    return db_poi

def delete_poi(db: Session, poi_id: str):
    """Delete a POI."""
    db_poi = db.query(POI).filter(POI.id == poi_id).first()
    if db_poi:
        db.delete(db_poi)
        db.commit()
        return True
    return False

def search_pois(db: Session, query: str):
    """Search for POIs by name or category."""
    return db.query(POI).filter(POI.name.ilike(f"%{query}%")).all()

def get_pois_by_floor(db: Session, floor_id: str):
    """Retrieve all POIs for a specific floor."""
    return (
        db.query(POI)
        .join(Zone, POI.zones)
        .join(Point, POI.point_id == Point.id)
        .filter(Zone.floor_id == floor_id)
        .all()
    )