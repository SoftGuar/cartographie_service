from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from models.environment import Environment
from schemas.environment import EnvironmentCreate, Environment
from database import get_db

router = APIRouter(prefix="/environments", tags=["environments"])

@router.post("/", response_model=Environment)
def create_environment(environment: EnvironmentCreate, db: Session = Depends(get_db)):
    db_environment = Environment(**environment.dict())
    db.add(db_environment)
    db.commit()
    db.refresh(db_environment)
    return db_environment