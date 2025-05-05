from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from database import get_db
from models.poi import POI
from schemas.poi import POICreate, POIUpdate, POIResponse
from schemas.Category import CategoryResponse  # Import the response schema
from services.poi_service import (
    create_poi,
    get_poi,
    update_poi,
    delete_poi,
    search_pois,
    get_pois_by_floor
)
from services.category_service import get_all_categories  
from services.point_service import create_point  # Import the service to create a Point


router = APIRouter(prefix="/pois", tags=["pois"])

@router.post("/", response_model=POIResponse)
def create_poi_endpoint(poi: POICreate, db: Session = Depends(get_db)):
    """Create a new POI."""
    # Create a new Point from x and y
    new_point = create_point(db, poi.x, poi.y)

    # Prepare POI data without x and y
    poi_data = poi.dict(exclude={"x", "y"})
    poi_data["point_id"] = new_point.id

    # Create the POI
    return create_poi(db, poi_data)


@router.get("/categories", response_model=List[CategoryResponse])
def get_categories_endpoint(db: Session = Depends(get_db)):
    """Fetch all categories."""
    categories = get_all_categories(db)
    if not categories:
        raise HTTPException(status_code=404, detail="No categories found")
    return categories

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

@router.get("/floor/{floor_id}", response_model=List[POIResponse])
def get_pois_by_floor_endpoint(floor_id: str, db: Session = Depends(get_db)):
    """Get all POIs for a specific floor."""
    pois = get_pois_by_floor(db, floor_id)
    if pois is None:
        raise HTTPException(status_code=404, detail="No POIs found for the specified floor")
    return pois

