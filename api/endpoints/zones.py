from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from database import get_db
from models.zone import Zone
from schemas.zone import ZoneCreate, ZoneUpdate, ZoneResponse
from services.zone_service import (
    create_zone,
    get_zone,
    update_zone,
    delete_zone,
    get_zones_by_floor,
    get_all_zone_types
)
from schemas.zone_type import ZoneTypeResponse

router = APIRouter(prefix="/zones", tags=["zones"])

@router.post("/", response_model=ZoneResponse)
def create_zone_endpoint(zone: ZoneCreate, db: Session = Depends(get_db)):
    """Create a new zone."""
    return create_zone(db, zone)

@router.get("/types", response_model=List[ZoneTypeResponse])
def get_zone_types_endpoint(db: Session = Depends(get_db)):
    """Fetch all ZoneTypes."""
    zone_types = get_all_zone_types(db)
    if not zone_types:
        raise HTTPException(status_code=404, detail="No ZoneTypes found")
    return zone_types

@router.get("/{zone_id}", response_model=ZoneResponse)
def get_zone_endpoint(zone_id: str, db: Session = Depends(get_db)):
    """Get a zone by ID."""
    db_zone = get_zone(db, zone_id)
    if not db_zone:
        raise HTTPException(status_code=404, detail="Zone not found")
    return db_zone

@router.put("/{zone_id}", response_model=ZoneResponse)
def update_zone_endpoint(zone_id: str, zone: ZoneUpdate, db: Session = Depends(get_db)):
    """Update a zone."""
    db_zone = update_zone(db, zone_id, zone)
    if not db_zone:
        raise HTTPException(status_code=404, detail="Zone not found")
    return db_zone

@router.delete("/{zone_id}")
def delete_zone_endpoint(zone_id: str, db: Session = Depends(get_db)):
    """Delete a zone."""
    success = delete_zone(db, zone_id)
    if not success:
        raise HTTPException(status_code=404, detail="Zone not found")
    return {"message": "Zone deleted"}

@router.get("/floor/{floor_id}", response_model=List[ZoneResponse])
def get_zones_by_floor_endpoint(floor_id: str, db: Session = Depends(get_db)):
    """Get all zones for a specific floor."""
    return get_zones_by_floor(db, floor_id)