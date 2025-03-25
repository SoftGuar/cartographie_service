from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session
from typing import List
import json
import base64
import uuid
import binascii
from models.floor import Floor
from schemas.floor import FloorCreate, FloorUpdate, FloorResponse
from database import get_db
from utils.validation import validate_base64

router = APIRouter(prefix="/floors", tags=["floors"])

@router.get(
    "/",
    response_model=List[FloorResponse],
    summary="Get all floors",
    description="Retrieve a list of all floors in the system.",
    response_description="List of floors with their details, grid data, and images."
)
def get_floors(db: Session = Depends(get_db)):
    """Get all floors"""
    floors = db.query(Floor).all()
    
    for floor in floors:
        floor.coordinates = json.loads(floor.coordinates) if floor.coordinates else None
        floor.grid_data = json.loads(floor.grid_data) if floor.grid_data else None
        floor.grid_dimensions = json.loads(floor.grid_dimensions) if floor.grid_dimensions else None

        if floor.image_data:
            floor.image_data = "data:image/png;base64," + base64.b64encode(floor.image_data).decode('utf-8')
    
    return floors

@router.post(
    "/",
    response_model=FloorResponse,
    summary="Create a new floor",
    description="Create a new floor with floor plan data and optional image.",
    response_description="The created floor with all its details.",
    responses={
        400: {
            "description": "Invalid base64 image data",
            "content": {
                "application/json": {
                    "example": {"detail": "Invalid base64 image data"}
                }
            }
        }
    }
)
def create_floor(floor: FloorCreate, db: Session = Depends(get_db)):
    # Validate base64 image data
    if floor.image_data and not validate_base64(floor.image_data):
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    # Generate UUID for new floor if not provided
    floor_data = floor.dict(exclude={'image_data'})
    if not floor_data.get('id'):
        floor_data['id'] = str(uuid.uuid4())
    
    # Convert objects to JSON strings for SQLite storage
    floor_data['coordinates'] = json.dumps(floor_data['coordinates'])
    floor_data['grid_data'] = json.dumps(floor_data['grid_data'])
    floor_data['grid_dimensions'] = json.dumps(floor_data['grid_dimensions'])
    
    # Convert base64 image data to bytes if present
    image_data = None
    if floor.image_data:
        try:
            if ',' in floor.image_data:
                image_data = base64.b64decode(floor.image_data.split(',')[1])
            else:
                image_data = base64.b64decode(floor.image_data)
        except binascii.Error:
            raise HTTPException(status_code=400, detail="Invalid base64 image data")

    # Check if floor exists
    existing_floor = db.query(Floor).filter(Floor.id == floor_data['id']).first()
    
    if existing_floor:
        # Update existing floor
        for key, value in floor_data.items():
            setattr(existing_floor, key, value)
        if image_data is not None:
            existing_floor.image_data = image_data
        db_floor = existing_floor
    else:
        # Create new floor
        db_floor = Floor(**floor_data, image_data=image_data)
        db.add(db_floor)
    
    try:
        db.commit()
        db.refresh(db_floor)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    
    # Convert JSON strings back to objects for response
    db_floor.coordinates = json.loads(db_floor.coordinates)
    db_floor.grid_data = json.loads(db_floor.grid_data)
    db_floor.grid_dimensions = json.loads(db_floor.grid_dimensions)
    if db_floor.image_data:
        db_floor.image_data = "data:image/png;base64," + base64.b64encode(db_floor.image_data).decode('utf-8')
    
    return db_floor

@router.put(
    "/{floor_id}",
    response_model=FloorResponse,
    summary="Update a floor",
    description="Update an existing floor's grid data and image.",
    responses={
        404: {"description": "Floor not found"},
        400: {"description": "Invalid base64 image data"}
    }
)
def update_floor(floor_id: str, floor_update: FloorUpdate, db: Session = Depends(get_db)):
    # Validate base64 image data
    if floor_update.image_data and not validate_base64(floor_update.image_data):
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    db_floor = db.query(Floor).filter(Floor.id == floor_id).first()
    if not db_floor:
        raise HTTPException(status_code=404, detail="Floor not found")
    
    # Update grid data and dimensions
    db_floor.grid_data = json.dumps(floor_update.grid_data)
    db_floor.grid_dimensions = json.dumps(floor_update.grid_dimensions)
    
    # Update image if provided
    if floor_update.image_data:
        try:
            if ',' in floor_update.image_data:
                db_floor.image_data = base64.b64decode(floor_update.image_data.split(',')[1])
            else:
                db_floor.image_data = base64.b64decode(floor_update.image_data)
        except binascii.Error:
            raise HTTPException(status_code=400, detail="Invalid base64 image data")
    
    try:
        db.commit()
        db.refresh(db_floor)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    
    # Convert JSON strings back to objects for response
    db_floor.grid_data = json.loads(db_floor.grid_data)
    db_floor.grid_dimensions = json.loads(db_floor.grid_dimensions)
    if db_floor.coordinates:
        db_floor.coordinates = json.loads(db_floor.coordinates)
    if db_floor.image_data:
        db_floor.image_data = "data:image/png;base64," + base64.b64encode(db_floor.image_data).decode('utf-8')
    
    return db_floor

@router.get(
    "/{floor_id}",
    response_model=FloorResponse,
    summary="Get a specific floor",
    description="Retrieve details of a specific floor by its ID.",
    responses={404: {"description": "Floor not found"}}
)
def get_floor(floor_id: str, db: Session = Depends(get_db)):
    floor = db.query(Floor).filter(Floor.id == floor_id).first()
    if not floor:
        raise HTTPException(status_code=404, detail="Floor not found")
    
    # Convert JSON strings back to objects
    floor.coordinates = json.loads(floor.coordinates) if floor.coordinates else None
    floor.grid_data = json.loads(floor.grid_data) if floor.grid_data else None
    floor.grid_dimensions = json.loads(floor.grid_dimensions) if floor.grid_dimensions else None
    # Convert image data to base64 if present
    if floor.image_data:
        floor.image_data = "data:image/png;base64," + base64.b64encode(floor.image_data).decode('utf-8')
    return floor

@router.get(
    "/{floor_id}/image",
    summary="Get floor image",
    description="Retrieve the floor plan image for a specific floor.",
    responses={
        200: {
            "content": {"image/png": {}},
            "description": "The floor's floor plan image"
        },
        404: {"description": "Image not found"}
    }
)
def get_floor_image(floor_id: str, db: Session = Depends(get_db)):
    floor = db.query(Floor).filter(Floor.id == floor_id).first()
    if not floor or not floor.image_data:
        raise HTTPException(status_code=404, detail="Image not found")
    return Response(content=floor.image_data, media_type="image/png")