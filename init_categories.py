from database import SessionLocal, engine
from sqlalchemy import text
import json

def init_categories():
    db = SessionLocal()
    try:
        # Delete existing categories first
        print("Deleting existing categories...")
        db.execute(text("DELETE FROM categories"))
        db.commit()
        print("Existing categories deleted successfully!")
        
        # Create new categories
        print("Creating new categories...")
        default_categories = [
            {
                "id": "1",
                "name": "distributor",
                "description": None
            },
            {
                "id": "2",
                "name": "door",
                "description": None
            },
            {
                "id": "3",
                "name": "default",
                "description": None
            }
        ]
        
        for category in default_categories:
            db.execute(
                text("INSERT INTO categories (id, name, description) VALUES (:id, :name, :description)"),
                category
            )
        db.commit()
        print("Categories initialized successfully!")
        
        # Verify the created categories
        print("\nNew categories:")
        result = db.execute(text("SELECT * FROM categories"))
        for row in result:
            print(f"ID: {row.id}, Name: {row.name}, Description: {row.description}")
            
    except Exception as e:
        print(f"Error initializing categories: {str(e)}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    init_categories() 