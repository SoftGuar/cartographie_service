import json
from typing import List
import uuid
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from models.environment import Environment as EnvironmentModel  # Use the SQLAlchemy model
from schemas.environment import EnvironmentCreate, Environment  # Use Pydantic schemas for validation
from schemas.floor import FloorResponse as Floor  # Assuming you have a Floor model
from database import get_db
from utils.notifications import send_notification 
import threading

router = APIRouter(prefix="/environments", tags=["environments"])

@router.post("/", response_model=Environment)
def create_environment(environment: EnvironmentCreate, db: Session = Depends(get_db)):
    environment_data = environment.dict()
    if "id" not in environment_data or not environment_data["id"]:
        environment_data["id"] = str(uuid.uuid4())  # Generate a UUID for the id
    
    # Use the SQLAlchemy model to create the database instance
    db_environment = EnvironmentModel(**environment_data)
    db.add(db_environment)
    db.commit()
    db.refresh(db_environment)
    threading.Thread(
        target=send_notification, 
        args=({"name": db_environment.name}, "/notifications/notify/environment_created")
    ).start()
    # Return the SQLAlchemy instance, which will be converted to the Pydantic schema
    return db_environment

@router.get("/", response_model=List[Environment])
def get_environments(db: Session = Depends(get_db)):
    """
    Retrieve all environments.
    """
    environments = db.query(EnvironmentModel).all()
    
    # Deserialize grid_data and grid_dimensions for each floor in each environment
    for environment in environments:
        for floor in environment.floors:
            if isinstance(floor.grid_data, str):
                try:
                    floor.grid_data = json.loads(floor.grid_data)
                except json.JSONDecodeError:
                    raise HTTPException(status_code=500, detail="Invalid JSON in grid_data")
            
            if isinstance(floor.grid_dimensions, str):
                try:
                    floor.grid_dimensions = json.loads(floor.grid_dimensions)
                except json.JSONDecodeError:
                    raise HTTPException(status_code=500, detail="Invalid JSON in grid_dimensions")
    
    return environments

@router.get("/{id}/floors", response_model=List[Floor])
def get_environment_floors(id: str, db: Session = Depends(get_db)):
    """
    Retrieve all floors of a specific environment by its ID.
    """
    environment = db.query(EnvironmentModel).filter(EnvironmentModel.id == id).first()
    if not environment:
        raise HTTPException(status_code=404, detail="Environment not found")
    
    # Assuming the Environment model has a relationship with the Floor model
    return environment.floors

@router.get("/{id}", response_model=Environment)
def get_environment_with_floors(id: str, db: Session = Depends(get_db)):
    """
    Retrieve the details of a specific environment by its ID, including its floors.
    """
    environment = db.query(EnvironmentModel).filter(EnvironmentModel.id == id).first()
    if not environment:
        raise HTTPException(status_code=404, detail="Environment not found")
    
    # Deserialize grid_data and grid_dimensions for each floor
    for floor in environment.floors:
        if isinstance(floor.grid_data, str):
            try:
                floor.grid_data = json.loads(floor.grid_data)
            except json.JSONDecodeError:
                raise HTTPException(status_code=500, detail="Invalid JSON in grid_data")
        
        if isinstance(floor.grid_dimensions, str):
            try:
                floor.grid_dimensions = json.loads(floor.grid_dimensions)
            except json.JSONDecodeError:
                raise HTTPException(status_code=500, detail="Invalid JSON in grid_dimensions")
    
    # Return the environment details along with its floors
    return environment