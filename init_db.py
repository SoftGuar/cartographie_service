from sqlalchemy.orm import Session
from database import SessionLocal, engine
import models
import json

def init_db():
    # Create tables
    models.Base.metadata.create_all(bind=engine)
    
    db = SessionLocal()
    try:
        # Check if company already exists
        existing = db.query(models.Company).filter_by(id="test-company-1").first()
        if existing:
            print("Test company already exists")
            return
        
        # Create a test company
        test_company = models.Company(
            id="test-company-1",
            name="Tech Office Building",
            address="123 Innovation Street, Silicon Valley"
        )
        db.add(test_company)
        
        # Create some test rooms
        rooms = [
            {
                "id": "room-1",
                "company_id": "test-company-1",
                "name": "Conference Room A",
                "building": "Main Building",
                "floor": 1,
                "width": 10.5,
                "height": 8.0,
                "coordinates": json.dumps([37.7749, -122.4194]),
                "grid_data": json.dumps([[0] * 42 for _ in range(32)]),  # 10.5m x 8m with 25cm cells
                "grid_dimensions": json.dumps([42, 32])
            },
            {
                "id": "room-2",
                "company_id": "test-company-1",
                "name": "Open Office Space",
                "building": "Main Building",
                "floor": 2,
                "width": 15.0,
                "height": 12.0,
                "coordinates": json.dumps([37.7749, -122.4195]),
                "grid_data": json.dumps([[0] * 60 for _ in range(48)]),  # 15m x 12m with 25cm cells
                "grid_dimensions": json.dumps([60, 48])
            }
        ]
        
        for room_data in rooms:
            room = models.Room(**room_data)
            db.add(room)
        
        db.commit()
        print("Test company and rooms created successfully")
        
    except Exception as e:
        print(f"Error creating test data: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    init_db()