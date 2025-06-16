from database import SessionLocal, engine
from sqlalchemy import text
from models.zone_type import ZoneType
from models.zone import Zone  # Ensure Zone is imported so SQLAlchemy can resolve the relationship
from models.floor import Floor  # Ensure Floor is imported so SQLAlchemy can resolve the relationship
from models.association_tables import poi_zone_association  # Ensure association table is imported
from models.poi import POI  # Ensure POI is imported so SQLAlchemy can resolve the relationship
from models.environment import Environment  # Ensure Environment is imported so SQLAlchemy can resolve the relationship
from models.category import Category  # Ensure Category is imported so SQLAlchemy can resolve the relationship
from models.point import Point  # Ensure Point is imported so SQLAlchemy can resolve the relationship
import json

def init_zone_types():
    db = SessionLocal()
    try:
        # Delete existing zones first
        print("Deleting existing zones...")
        db.execute(text("DELETE FROM zones"))
        db.commit()
        print("Existing zones deleted successfully!")
        
        # Delete existing zone types
        print("Deleting existing zone types...")
        db.execute(text("DELETE FROM zone_types"))
        db.commit()
        print("Existing zone types deleted successfully!")
        
        # Create new zone types
        print("Creating new zone types...")
        default_types = [
            {
                "id": "1",
                "name": "walkable",
                "properties": None
            },
            {
                "id": "2",
                "name": "danger",
                "properties": None
            }
        ]
        
        for zone_type in default_types:
            db.execute(
                text("INSERT INTO zone_types (id, name, properties) VALUES (:id, :name, :properties)"),
                zone_type
            )
        db.commit()
        print("Zone types initialized successfully!")
        
        # Verify the created zone types
        print("\nNew zone types:")
        result = db.execute(text("SELECT * FROM zone_types"))
        for row in result:
            print(f"ID: {row.id}, Name: {row.name}, Properties: {row.properties}")
            print(f"Raw properties type: {type(row.properties)}")
            print(f"Raw properties value: {row.properties}")
            
    except Exception as e:
        print(f"Error initializing zone types: {str(e)}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    init_zone_types() 