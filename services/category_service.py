from sqlalchemy.orm import Session
from models.category import Category

def get_all_categories(db: Session):
    """Retrieve all categories."""
    return db.query(Category).all()