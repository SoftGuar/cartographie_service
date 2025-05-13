from sqlalchemy.orm import Session
from models.point import Point
import uuid

def create_point(db: Session, x: float, y: float) -> Point:
    """Create a new Point."""
    point = Point(id=str(uuid.uuid4()), x=x, y=y)
    db.add(point)
    db.commit()
    db.refresh(point)
    return point