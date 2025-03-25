from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from database import get_db
from models.poi import POI
from schemas.poi import POICreate, POIUpdate, POIResponse
from services.poi_service import (
    create_poi,
    get_poi,
    update_poi,
    delete_poi,
    search_pois,
)

router = APIRouter(prefix="/pois", tags=["pois"])

@router.post("/", response_model=POIResponse)
def create_poi_endpoint(poi: POICreate, db: Session = Depends(get_db)):
    """Create a new POI."""
    return create_poi(db, poi)

@router.get("/{poi_id}", response_model=POIResponse)
def get_poi_endpoint(poi_id: str, db: Session = Depends(get_db)):
    """Get a POI by ID."""
    db_poi = get_poi(db, poi_id)
    if not db_poi:
        raise HTTPException(status_code=404, detail="POI not found")
    return db_poi

@router.put("/{poi_id}", response_model=POIResponse)
def update_poi_endpoint(poi_id: str, poi: POIUpdate, db: Session = Depends(get_db)):
    """Update a POI."""
    db_poi = update_poi(db, poi_id, poi)
    if not db_poi:
        raise HTTPException(status_code=404, detail="POI not found")
    return db_poi

@router.delete("/{poi_id}")
def delete_poi_endpoint(poi_id: str, db: Session = Depends(get_db)):
    """Delete a POI."""
    success = delete_poi(db, poi_id)
    if not success:
        raise HTTPException(status_code=404, detail="POI not found")
    return {"message": "POI deleted"}

@router.get("/search", response_model=List[POIResponse])
def search_pois_endpoint(query: str, db: Session = Depends(get_db)):
    """Search for POIs by name or category."""
    return search_pois(db, query)